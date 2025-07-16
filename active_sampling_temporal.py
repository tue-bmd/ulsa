"""
Script for running an ultrasound line-scanning agent that chooses which lines to scan
based on samples from a distribution over full images conditioned on the lines observed
so far.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    """Parse arguments for training DDIM."""
    parser = argparse.ArgumentParser(description="DDIM inference")
    parser.add_argument(
        "--agent_config",
        type=str,
        default="configs/cardiac_112_3_frames.yaml",
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
        "--random_circle_inclusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include a random circle in the target image.",
    )
    parser.add_argument(
        "--target_sequence",
        type=str,
        # 20240701_P3_PLAX_0000
        # default="/mnt/z/Ultrasound-BMd/data/oisin/carotid_img/512_128/test/10_cross_2cm_L_0000.img.hdf5",
        # default="{data_root}/USBMD_datasets/echonet/val/0X10A5FC19152B50A5.hdf5",
        default="{data_root}/USBMD_datasets/2024_USBMD_cardiac_S51/HDF5/20240701_P1_A4CH_0001.hdf5",
        help="A hdf5 file containing an ordered sequence of frames to sample from.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="data/raw_data",
        help="The type of data to load from the hdf5 file.",
    )
    parser.add_argument(
        "--image_range",
        type=int,
        nargs=2,
        default=(-60, 0),
        help=(
            "Range of pixel values in the images (e.g., --image_range 0 255), only used if "
            "data_type is 'data/image'"
        ),
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
        default="float32",
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
from ulsa.agent import Agent, AgentState, setup_agent
from ulsa.downstream_task import (
    DownstreamTask,
    downstream_task_registry,
)
from ulsa.io_utils import (
    animate_overviews,
    make_save_dir,
    map_range,
    plot_belief_distribution_for_presentation,
    plot_frame_overview,
    plot_frames_for_presentation,
)
from ulsa.ops import AntiAliasing
from ulsa.pfield import (
    select_transmits,
    update_scan_for_polar_grid,
)
from zea import Config, File, Pipeline, Probe, Scan, log, set_data_paths
from zea.agent.masks import k_hot_to_indices
from zea.tensor_ops import batched_map, func_with_one_batch_dim
from zea.utils import translate


def simple_scan(f, init, xs, length=None, disable_tqdm=False):
    """Basically ops.scan but not jitted, more GPU memory efficient."""
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in tqdm(xs, leave=False, disable=disable_tqdm):
        carry, y = f(carry, x)
        if isinstance(y, (list, tuple)):
            y = [ops.convert_to_numpy(_y) for _y in y]
        else:
            y = ops.convert_to_numpy(y)
        ys.append(y)
    return carry, [np.stack(tensors) for tensors in zip(*ys)]


def hard_projection(image, masked_measurements):
    """
    Projects an image onto the measurement space by replacing values with measurements
    where they exist.

    Args:
        image (Tensor): The image to project onto
        masked_measurements (Tensor): The masked measurements to project from
            (same shape as image, with zeros where no measurements exist)

    Returns:
        Tensor: The projected image with measurements inserted where they exist
    """
    return ops.where(masked_measurements != 0, masked_measurements, image)


def soft_projection(
    image, measurements, weighting_map, data_range=(-1, 1), mask_range=(0, 1)
):
    _measurements = translate(measurements, data_range, mask_range)
    _measurements *= weighting_map
    _image = translate(image, data_range, mask_range)
    _image *= 1 - weighting_map
    combined = _measurements + _image
    return translate(combined, mask_range, data_range)


def elementwise_append_tuples(t1, t2):
    """
    Elementwise append tuple of tensors

    t2 tensors will be appended to t1 tensors
    """
    assert len(t1) == len(t2)
    combined = []
    for e1, e2 in zip(t1, t2):
        combined.append(ops.concatenate([e1[None, ...], e2]))
    return tuple(combined)


def apply_downstream_task(agent_config, reconstructions):
    downstream_task_key = agent_config.get("downstream_task", None)
    if downstream_task_key is None:
        return None, None

    try:
        downstream_task: DownstreamTask = (
            None
            if downstream_task_key is None
            else downstream_task_registry[downstream_task_key]
        )(batch_size=4)
        log.info(
            log.blue(f"Running downstream task: {log.green(downstream_task.name())}")
        )
        downstream_task_outputs = batched_map(
            downstream_task.call_generic, reconstructions, jit=True, batch_size=4
        )
        return downstream_task.output_type(), downstream_task_outputs
    except Exception as e:
        log.error(
            f"Downstream task {downstream_task_key} not found or failed. "
            f"Skipping downstream task. Original error: {e}"
        )
        return None, None


@dataclass
class AgentResults:
    masks: np.ndarray
    target_imgs: np.ndarray
    reconstructions: np.ndarray
    belief_distributions: np.ndarray  # shape: (n_frames, particles, h, w, 1)
    measurements: np.ndarray

    def squeeze(self, axis=-1):
        return AgentResults(
            np.squeeze(self.masks, axis=axis),
            np.squeeze(self.target_imgs, axis=axis),
            np.squeeze(self.reconstructions, axis=axis),
            np.squeeze(self.belief_distributions, axis=axis),
            np.squeeze(self.measurements, axis=axis),
        )

    def to_uint8(self, input_range=None):
        """
        Convert the results to uint8 format, mapping the input range to (0, 255).
        """

        def map_to_uint8(data):
            return map_range(data, input_range, (0, 255)).astype(np.uint8)

        return AgentResults(
            self.masks,  # keep masks as is
            map_to_uint8(self.target_imgs),
            map_to_uint8(self.reconstructions),
            map_to_uint8(self.belief_distributions),
            map_to_uint8(self.measurements),
        )


def lines_rx_apo(n_tx, n_z, n_x):
    """
    Create a receive apodization for line scanning.
    This is a simple apodization that applies a uniform weight to all elements.

    Returns:
        rx_apo: np.ndarray of shape (n_tx, n_z, n_x)
    """
    assert n_x == n_tx
    rx_apo = np.zeros((n_tx, n_z, n_x), dtype=np.float32)
    for tx in range(n_tx):
        rx_apo[tx, :, tx] = 1.0
    rx_apo = rx_apo.reshape((n_tx, -1))
    return rx_apo[..., None]  # shape (n_tx, n_pix, 1)


def run_active_sampling(
    agent: Agent,
    agent_state: AgentState,
    target_sequence,
    n_actions: int,
    pipeline: Pipeline = None,
    scan: Scan = None,
    probe: Probe = None,
    hard_project=False,
    verbose=True,
    post_pipeline=None,
    pfield: np.ndarray = None,
) -> AgentResults:
    if verbose:
        log.info(log.blue("Running active sampling"))
        agent.print_summary()

    # Prepare acquisition function
    if scan and scan.n_tx > 1:
        disabled_pfield = ops.ones((scan.n_z * scan.n_x, scan.n_tx))
        if pfield is not None:
            flat_pfield = pfield.reshape(scan.n_tx, -1).swapaxes(0, 1)
            flat_pfield = ops.convert_to_tensor(flat_pfield)
        else:
            flat_pfield = disabled_pfield
        rx_apo = lines_rx_apo(scan.n_tx, scan.n_z, scan.n_x)
        base_params = pipeline.prepare_parameters(
            scan=scan, probe=probe, flat_pfield=flat_pfield, rx_apo=rx_apo, factor=6
        )

        # No pfield for target
        target_pipeline_params = base_params | dict(flat_pfield=disabled_pfield)

        def acquire(
            full_data,
            mask,
            selected_lines,
            pipeline_state: dict,
            target_pipeline_state: dict,
        ):
            # Select transmits
            transmits = k_hot_to_indices(selected_lines, n_actions)
            transmits = ops.squeeze(transmits, 0)
            selected_data = full_data[transmits]
            params = pipeline_state | dict(
                t0_delays=base_params["t0_delays"][transmits],
                tx_apodizations=base_params["tx_apodizations"][transmits],
                polar_angles=base_params["polar_angles"][transmits],
                focus_distances=base_params["focus_distances"][transmits],
                initial_times=base_params["initial_times"][transmits],
                flat_pfield=base_params["flat_pfield"][:, transmits],
                n_tx=len(transmits),
                rx_apo=base_params["rx_apo"][transmits],
            )
            params = pipeline.prepare_parameters(**params)

            # Run pipeline with selected lines
            output = pipeline(data=selected_data, **(base_params | params))
            measurements = output["data"]
            pipeline_state = {"maxval": output["maxval"]}

            # Run pipeline with full data
            output = pipeline(
                data=full_data, **(target_pipeline_params | target_pipeline_state)
            )
            target = output["data"]
            target_pipeline_state = {"maxval": output["maxval"]}

            return measurements, target, pipeline_state, target_pipeline_state

    else:

        def acquire(
            full_data,
            mask,
            selected_lines,
            pipeline_state: dict,
            target_pipeline_state: dict,
        ):
            target = pipeline(data=full_data, **pipeline_state)["data"]
            return target * mask, target, {}, {}

    def perception_action_step(agent_state: AgentState, target_data):
        # 1. Acquire measurements
        current_mask = agent_state.mask[..., -1, None]
        selected_lines = agent_state.selected_lines[None]  # (1, n_tx)
        measurements, target_img, pipeline_state, target_pipeline_state = acquire(
            target_data,
            current_mask,
            selected_lines,
            agent_state.pipeline_state,
            agent_state.target_pipeline_state,
        )

        if agent.pfield is None:
            _measurements = measurements * current_mask

        # 2. run perception and action selection via agent.recover
        reconstruction, new_agent_state = agent.recover(_measurements, agent_state)

        if hard_project and agent.pfield is None:
            reconstruction = hard_projection(reconstruction, _measurements)
        elif hard_project and agent.pfield is not None:
            raise NotImplementedError(
                "Hard projection with pfield is not implemented yet. "
                "Please set hard_project=False or use a different agent."
            )
            # TODO: WIP!
            weighting_map = agent.pfield[agent_state.selected_lines].sum(axis=0)

            reconstruction = soft_projection(
                reconstruction, measurements, current_mask, data_range=agent.input_range
            )

        new_agent_state.pipeline_state = pipeline_state
        new_agent_state.target_pipeline_state = target_pipeline_state
        return (
            new_agent_state,
            (
                reconstruction,
                current_mask,
                target_img,
                new_agent_state.belief_distribution,
                measurements,
            ),
        )

    if verbose:
        print(f"Running active sampling for {len(target_sequence)} frames...")

    # Initial recover -> full number of diffusion steps
    # Subsequent percetion_action uses SeqDiff
    start_time = time.perf_counter()
    _, outputs = simple_scan(
        perception_action_step,
        agent_state,
        target_sequence,
        disable_tqdm=not verbose,
    )
    fps = len(target_sequence) / (time.perf_counter() - start_time)
    if verbose:
        print("Done! FPS: ", fps)

    reconstructions, masks, target_imgs, belief_distributions, measurements = outputs

    if post_pipeline:
        reconstructions = post_pipeline(data=reconstructions)["data"]
        masks = post_pipeline(data=masks)["data"]
        target_imgs = post_pipeline(data=target_imgs)["data"]
        belief_distributions = func_with_one_batch_dim(
            lambda data: post_pipeline(data=data)["data"],
            belief_distributions,
            n_batch_dims=2,
            batch_size=belief_distributions.shape[0],
        )
        measurements = post_pipeline(data=measurements)["data"]

    return AgentResults(
        masks,
        target_imgs,
        reconstructions,
        belief_distributions,
        measurements,
    )


def fix_paths(agent_config, data_paths=None):
    if data_paths is None:
        data_paths = set_data_paths("users.yaml", local=False)
    output_dir = data_paths["output"]
    agent_config.diffusion_inference.run_dir = (
        agent_config.diffusion_inference.run_dir.format(output_dir=output_dir)
    )
    return agent_config


def make_pipeline(
    data_type,
    dynamic_range,
    input_range,
    input_shape,
    action_selection_shape,
    jit_options="ops",
) -> Pipeline:
    expand_dims = zea.ops.Lambda(ops.expand_dims, {"axis": -1})
    # Not using zea.ops.Normalize because that also clips the data and we might not want
    # to do that for image data, e.g. the legacy echonet data (which has values outside the range).
    # Also zea.ops.Normalize is used in the default pipeline, so we can't reuse it.
    normalize = zea.ops.Lambda(
        translate,
        {"range_from": dynamic_range, "range_to": input_range},
    )

    downsample_factor = 4  # TODO
    if data_type not in ["data/image", "data/image_3D"]:
        pipeline = Pipeline.from_default(
            with_batch_dim=False,
            num_patches=40,
            jit_options=jit_options,
            pfield=False,
        )
        # pipeline.insert(1, AntiAliasing(axis=-2, complex_channels=True))
        pipeline.insert(1, zea.ops.Downsample(downsample_factor))
        pipeline.append(expand_dims)
        resize = zea.ops.Lambda(
            ops.image.resize,
            {
                "size": action_selection_shape,
                "interpolation": "bilinear",
                "antialias": True,  # TODO: different way of antialiasing?
            },
        )
        pipeline.append(resize)
        pipeline.append(normalize)
        pipeline.append(zea.ops.Clip(*input_range))
        pipeline.append(
            zea.ops.Pad(
                input_shape[:-1],
                axis=(-3, -2),
                pad_kwargs=dict(mode="symmetric"),
                # fail_on_bigger_shape=False,
            )
        )
    else:
        pipeline = Pipeline(
            [normalize, expand_dims], jit_options=jit_options, with_batch_dim=False
        )

    if data_type == "data/image_3D":
        # transpose so that azimuth dimension is on the outside, like a batch.
        # then we simply apply the 2d DM along all azimuthal angles
        pipeline.append(zea.ops.Transpose((1, 0, 2, 3)))
        # we do cropping rather than resizing to maintain elevation focusing
        pipeline.append(
            zea.ops.Lambda(keras.layers.CenterCrop(*action_selection_shape))
        )

    return pipeline


def preload_data(
    file: File,
    n_frames: int,  # if there are less than n_frames, it will load all frames
    data_type="data/image",
    dynamic_range=(-70, -28),
    cardiac=False,
    type="focused",  # 'focused' or 'diverging'
):
    # Get scan and probe from the file
    try:
        scan = file.scan()
    except:
        scan = None

    try:
        probe = file.probe()
    except:
        probe = None

    # TODO: kind of hacky way to update the scan for the cardiac dataset
    if cardiac:
        dynamic_range = (-70, -28)
        select_transmits(scan, type=type)
        update_scan_for_polar_grid(scan, dynamic_range=dynamic_range)

    # slice(None) means all frames.
    if data_type in ["data/raw_data"]:
        validation_sample_frames = file.load_data(
            data_type, [slice(n_frames), scan.selected_transmits]
        )
    else:
        validation_sample_frames = file.load_data(data_type, slice(n_frames))

    # just for debugging
    # if data_type == "data/image_3D":
    #     _, data_n_ax, data_n_az, data_n_elev = validation_sample_frames.shape
    #     slice_az = 2
    #     crop_az = (data_n_az // 2) - slice_az
    #     validation_sample_frames = validation_sample_frames[:,:,crop_az:-crop_az,:]

    return validation_sample_frames, scan, probe


# Example usage
if __name__ == "__main__":
    print(f"Using {backend.backend()} backend 🔥")
    data_paths = set_data_paths("users.yaml", local=False)
    data_root = data_paths["data_root"]
    output_dir = data_paths["output"]
    save_dir = args.save_dir.format(output_dir=output_dir)
    save_dir = Path(save_dir)

    agent_config = Config.from_yaml(args.agent_config)
    agent_config = fix_paths(agent_config, data_paths)
    if args.override_config is not None:
        agent_config.update_recursive(args.override_config)

    dataset_path = args.target_sequence.format(data_root=data_root)
    cardiac = "cardiac" in str(dataset_path)

    if cardiac:
        dynamic_range = (-70, -28)
    else:
        dynamic_range = args.image_range

    n_rays = agent_config.action_selection.shape[-1]
    with File(dataset_path) as file:
        n_frames = agent_config.io_config.get("frame_cutoff", "all")
        validation_sample_frames, scan, probe = preload_data(
            file, n_frames, args.data_type, dynamic_range, cardiac=cardiac
        )

    if scan.theta_range is not None:
        theta_range_deg = np.rad2deg(scan.theta_range)
        log.warning(
            f"Overriding scan conversion angles using the scan object: {theta_range_deg}"
        )
        agent_config.io_config.scan_conversion_angles = list(theta_range_deg)

    if scan.probe_geometry is not None and "pfield" in agent_config.action_selection:
        scan.pfield_kwargs |= agent_config.action_selection.get("pfield", {})
        pfield = scan.pfield
    else:
        pfield = None

    agent, agent_state = setup_agent(
        agent_config,
        seed=jax.random.PRNGKey(args.seed),
        pfield=pfield,
        jit_mode="recover",
    )

    pipeline = make_pipeline(
        data_type=args.data_type,
        dynamic_range=dynamic_range,
        input_range=agent.input_range,
        input_shape=agent.input_shape,
        action_selection_shape=agent_config.action_selection.shape,
    )

    post_pipeline = Pipeline(
        [zea.ops.Lambda(keras.layers.CenterCrop(*agent_config.action_selection.shape))],
        with_batch_dim=True,
    )

    run_dir, run_id = make_save_dir(save_dir)
    log.info(f"Run dir created at {log.yellow(run_dir)}")
    results = run_active_sampling(
        agent,
        agent_state,
        validation_sample_frames,
        n_actions=agent_config.action_selection.n_actions,
        pipeline=pipeline,
        scan=scan,
        probe=probe,
        hard_project=agent_config.diffusion_inference.hard_project,
        post_pipeline=post_pipeline,
        pfield=pfield,
    )

    dst_output_type, downstream_task_outputs = apply_downstream_task(
        agent_config, results.reconstructions
    )

    if not downstream_task_outputs:
        downstream_task_outputs = [None] * len(results.reconstructions)

    # TODO: maybe more io_config to script args? Since this isn't relevant to benchmarking
    if agent_config.io_config.plot_frame_overview:
        postprocess_fn = lambda x: ops.cast(
            map_range(x, agent.input_range, (0, 255)), dtype="uint8"
        )

        for frame_index, (target, recon, beliefs, mask, dst_out) in enumerate(
            zip(
                results.target_imgs,
                results.reconstructions,
                results.belief_distributions,
                results.masks,
                downstream_task_outputs,
            )
        ):
            plot_frame_overview(
                run_dir,
                frame_index,
                postprocess_fn(target),
                postprocess_fn(recon),
                ops.abs(target - recon),
                agent_config.io_config,
                images_from_posterior=postprocess_fn(beliefs),
                mask=mask,
                **({dst_output_type: dst_out} if dst_out is not None else {}),
            )
        if agent_config.io_config.save_animation:
            animate_overviews(run_dir, agent_config.io_config)

    if agent_config.io_config.plot_frames_for_presentation:
        postfix_filename = Path(dataset_path).stem
        squeezed_results = results.squeeze(-1)

        frame_to_plot = 0
        plot_belief_distribution_for_presentation(
            save_dir / run_id,
            squeezed_results.belief_distributions[frame_to_plot],
            squeezed_results.masks[frame_to_plot],
            agent_config.io_config,
            next_masks=squeezed_results.masks[frame_to_plot + 1],
        )

        plot_frames_for_presentation(
            save_dir / run_id,
            squeezed_results.target_imgs,
            squeezed_results.reconstructions,
            squeezed_results.masks,
            squeezed_results.measurements,
            io_config=agent_config.io_config,
            image_range=agent.input_range,
            postfix_filename=postfix_filename,
            **agent_config.io_config.get("plot_frames_for_presentation_kwargs", {}),
        )

    with open(save_dir / run_id / "config.json", "w") as json_file:
        json.dump(agent_config, json_file, indent=4)
