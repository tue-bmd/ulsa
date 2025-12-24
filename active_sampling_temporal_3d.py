import argparse
import json
import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse arguments for training DDIM."""
    parser = argparse.ArgumentParser(description="DDIM inference")
    parser.add_argument(
        "--agent_config",
        type=str,
        default="./configs/elevation_3d.yaml",
        help="Path to agent config yaml.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="jax",
        help="ML backend to use",
        choices=["tensorflow", "torch", "jax"],
    )
    parser.add_argument(
        "--target_sequence",
        type=str,
        # default="/mnt/z/Ultrasound-BMd/data/oisin/carotid_img/512_128/test/10_cross_2cm_L_0000.img.hdf5",
        default="{data_root}/USBMD_datasets/echonet/val/0X10A5FC19152B50A5.hdf5",
        help="A hdf5 file containing an ordered sequence of frames to sample from.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="data/image",
        help="The type of data to load from the hdf5 file.",
    )
    parser.add_argument(
        "--image_range",
        type=int,
        nargs=2,
        default=(-60, 0),
        help="Range of pixel values in the images (e.g., --image_range 0 255)",
    )
    parser.add_argument(
        "--slice_az",
        type=int,
        default=None,
        help="Take a slice of azimuth angles around the center rather than the whole volume. Useful for debugging",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="{output_dir}/active_sampling",
        help="Directory in which to save results",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["float32", "mixed_float16", "mixed_bfloat16"],
        default="mixed_float16",
        help="Precision to use for inference: https://keras.io/api/mixed_precision/policy/",
    )
    parser.add_argument(
        "--override_config",
        type=json.loads,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="temp",
        help="Directory to save plots and animations.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["KERAS_BACKEND"] = args.backend
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from zea import init_device

    init_device()

import keras
from keras import ops

if __name__ == "__main__":
    keras.mixed_precision.set_global_policy(args.precision)

import jax
from keras.src import backend

import zea.ops
from active_sampling_temporal import run_active_sampling
from elevation_interpolation.tools import (
    TITLE_LOOKUP,
    animate_slices_from_3d_volumes,
    plot_slices_from_3d_volumes,
    postprocess_3d_data,
)
from ulsa.agent import AgentConfig, reset_agent_state, setup_agent
from zea import File, Pipeline, log, set_data_paths
from zea.func import translate
from zea.io_lib import load_image, save_to_gif
from zea.visualize import plot_biplanes, set_mpl_style


def safe_slice(start, end):
    return slice(start, None if end == 0 else -end)


def read_3d_file(path, n_frames, slice_az, model_input_shape):
    grid = {}
    n_ax, n_elev, _ = model_input_shape
    with File(path, "r") as f:
        grid["rho"] = f["/scan/frustum/rho"][()]
        grid["phi"] = f["/scan/frustum/phi"][()]
        grid["theta"] = f["/scan/frustum/theta"][()]
        _, data_n_ax, data_n_az, data_n_elev = f["/data/image_3D"][()].shape
        crop_ax = (data_n_ax - n_ax) // 2
        crop_elev = (data_n_elev - n_elev) // 2
        crop_az = (data_n_az // 2) - slice_az if slice_az is not None else 0
        data = f["/data/image_3D"][()][
            :n_frames,
            safe_slice(crop_ax, crop_ax),
            safe_slice(crop_az, crop_az) if slice_az is not None else slice(None),
            safe_slice(crop_elev, crop_elev),
        ]

    # permute dims so that n_ax is on the outside, like a 'batch' dim
    data = ops.transpose(data, (0, 2, 1, 3))
    log.info("Loaded file successfully!")
    return data, grid


def make_plots(agent, output, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    postprocess_fn = partial(
        postprocess_3d_data,
        normalization_range=agent.input_range,
        scan_convert_mode="cartesian_phi",
        swap_axes=True,
        scan_convert_kwargs={
            "rho_range": rho_range,
            "theta_range": theta_range,
            "phi_range": phi_range,
            "fill_value": np.nan,
        },
    )

    keys = list(output.keys())
    # sort keys based on title lookup order
    keys = sorted(keys, key=lambda key: list(TITLE_LOOKUP.keys()).index(key))

    output_extracted = {
        key: ops.convert_to_tensor(output[key]) for key in output.keys()
    }

    output_extracted["masks"] = translate(
        output_extracted["masks"], (0, 1), agent.input_range
    )

    titles = [TITLE_LOOKUP[key] for key in keys]

    images = [output_extracted[key] for key in keys]

    # perform on cpu with jax (tensors are already jax)
    device = jax.devices("cpu")[0]  # currently goes OOM on gpu
    with jax.default_device(device):
        # take zero because postprocess returns a tuple of (data, scan_info)
        images = [postprocess_fn(image)[0] for image in images]
        images = [ops.convert_to_numpy(image) for image in images]
        images = np.stack(images)

    np.save(f"{out_dir}/images_cached.npy", images)

    # Add variance and composite images if posterior samples are returned
    plot_slices_cmap = ["gray", "gray", "gray", "gray", "viridis"]
    ## showing multime time frames for a single elevation plane
    frame_idx = range(0, images.shape[1])
    azimuth_idx = images.shape[-2] // 2
    # elevation_idx = range(0, images.shape[-1])
    elevation_idx = range(images.shape[-1] // 2, images.shape[-1] // 2 + 10)
    depth_idx = range(0, images.shape[-3])

    plotting_args = dict(
        aspect="auto",
        background_color="black",
        text_color="white",
        show_frustum=True,
        frustum_kwargs=dict(
            title="",
            rho_range=rho_range,  # depth range in mm
            theta_range=theta_range,  # azimuth range in rad
            phi_range=phi_range,  # elevation range in rad
        ),
        cmap=plot_slices_cmap,
        dpi=300,
    )

    for t in frame_idx:
        plot_slices_from_3d_volumes(
            {title: image for title, image in zip(titles, images)},
            select_dims={"t": t, "a": azimuth_idx},
            broadcast_dims=["e", "d"],
            save_path=f"{out_dir}/slices_t={t}.png",
            **plotting_args,
        )

    acquisitions = images[titles.index("Acquisitions")]
    diffusion = images[titles.index("Diffusion")]
    variance = images[titles.index("Variance")]
    # z, x, y  --> ax, az, el
    # transpose to --> ax, el, az
    _, n_ax, n_az, n_elev = ops.shape(diffusion)
    acquisitions = np.transpose(acquisitions, (0, 1, 3, 2))
    diffusion = np.transpose(diffusion, (0, 1, 3, 2))
    variance = np.transpose(variance, (0, 1, 3, 2))
    var_vmin, var_vmax = np.nanmin(variance), np.nanmax(variance)
    for t in frame_idx:
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        fig, ax1 = plot_biplanes(
            diffusion[t], slice_x=n_elev // 2, slice_y=n_az // 2, fig=fig, ax=ax1
        )
        ax1.set_title(TITLE_LOOKUP["diffusion"], fontsize=18)

        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        fig, ax2 = plot_biplanes(
            acquisitions[t],
            slice_x=n_elev // 2,
            slice_y=n_az // 2,
            cmap="plasma",
            fig=fig,
            ax=ax2,
        )
        ax2.set_title(TITLE_LOOKUP["masks"], fontsize=18)

        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        fig, ax3 = plot_biplanes(
            variance[t],
            slice_x=n_elev // 2,
            slice_y=n_az // 2,
            cmap="plasma",
            fig=fig,
            ax=ax3,
            vmin=var_vmin,
            vmax=var_vmax,
        )
        ax3.set_title(TITLE_LOOKUP["variance"], fontsize=18)

        crop_fraction = 0.8
        crop_start = lambda shape: shape - (shape * crop_fraction)
        crop_end = lambda shape: (shape * crop_fraction)
        for ax, img in zip([ax1, ax2, ax3], [diffusion, acquisitions, variance]):
            ax.set_xlim(crop_start(img.shape[2]), crop_end(img.shape[2]))
            ax.set_ylim(crop_start(img.shape[3]), crop_end(img.shape[3]))
            ax.set_zlim(crop_start(img.shape[1]), crop_end(img.shape[1]))

        # plt.tight_layout()
        path = f"{out_dir}/biplanes_{t}.png"
        plt.savefig(path, bbox_inches="tight", pad_inches=0.2)
        log.info(f"Saved biplane plot to {log.yellow(path)}")
    # make animation
    out_plots = [
        load_image(f"{out_dir}/biplanes_{t}.png", grayscale=False, color_order="BGR")
        for t in frame_idx
    ]
    save_to_gif(out_plots, f"{out_dir}/biplanes.gif", fps=10)
    log.info(f"Saved biplane plot to {log.yellow(f'{out_dir}/biplanes.gif')}")

    azimuth_index = range(images.shape[-2])
    animate_slices_from_3d_volumes(
        {title: image for title, image in zip(titles, images)},
        select_dims={"t": 0, "a": azimuth_index},
        broadcast_dims=["e", "d"],
        animate_dim="a",
        save_path=f"{out_dir}/volume-sweep-azimuth.gif",
        fps=10,
        **plotting_args,
    )
    ## sweep over time
    animate_slices_from_3d_volumes(
        {title: image for title, image in zip(titles, images)},
        select_dims={"t": frame_idx, "a": azimuth_idx},
        broadcast_dims=["e", "d"],
        animate_dim="t",
        save_path=f"{out_dir}/volume-sweep-time.gif",
        fps=10,
        **plotting_args,
    )


if __name__ == "__main__":
    print(f"Using {backend.backend()} backend 🔥")
    data_paths = set_data_paths("users.yaml", local=False)
    set_mpl_style()
    data_root = data_paths["data_root"]
    output_dir = data_paths["output"]
    save_dir = args.save_dir.format(output_dir=output_dir)
    save_dir = Path(save_dir)

    agent_config = AgentConfig.from_yaml(args.agent_config)
    agent_config.fix_paths()
    if args.override_config is not None:
        agent_config.update_recursive(args.override_config)

    target_path = args.target_sequence.format(data_root=data_root)

    n_frames = agent_config.io_config.get("frame_cutoff", "all")

    seed = jax.random.PRNGKey(args.seed)
    agent, agent_state = setup_agent(agent_config, seed=seed)

    validation_sample_frames, grid = read_3d_file(
        target_path, n_frames, args.slice_az, agent.input_shape
    )

    batch_size = ops.shape(validation_sample_frames)[1]
    agent_state = reset_agent_state(agent, seed, batch_size=batch_size)

    rho_range = (ops.min(grid["rho"]), ops.max(grid["rho"]))
    theta_range = (ops.min(grid["theta"]), ops.max(grid["theta"]))
    phi_range = (ops.min(grid["phi"]), ops.max(grid["phi"]))

    expand_dims = zea.ops.Lambda(ops.expand_dims, {"axis": -1})
    # Not using zea.ops.Normalize because that also clips the data!
    normalize = zea.ops.Lambda(
        translate,
        {"range_from": args.image_range, "range_to": agent.input_range},
    )
    pipeline = Pipeline(
        [normalize, expand_dims], jit_options="pipeline", with_batch_dim=False
    )

    results = run_active_sampling(
        agent,
        agent_state,
        validation_sample_frames,
        n_actions=agent_config.action_selection.n_actions,
        pipeline=pipeline,
        hard_project=agent_config.diffusion_inference.hard_project,
    )

    n_az = ops.shape(results.target_imgs)[1]
    # TODO: need to merge usbmd ulsa branch into zea
    make_plots(
        agent,
        output={
            "masks": ops.repeat(ops.expand_dims(results.masks, axis=1), n_az, axis=1),
            "diffusion": results.reconstructions,
            "clean_images": results.target_imgs,
            "variance": ops.var(results.belief_distributions, axis=1),
        },
        out_dir=args.out_dir,
    )
