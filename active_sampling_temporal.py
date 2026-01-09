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
from typing import Callable, Optional

import numpy as np
from tqdm import tqdm


def parse_args():
    """Parse arguments for training DDIM."""
    parser = argparse.ArgumentParser(description="DDIM inference")
    parser.add_argument(
        "--agent_config",
        type=str,
        default="configs/echonet_3_frames.yaml",
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
        default=None,
        help="A hdf5 file containing an ordered sequence of frames to sample from.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default=None,
        help="The type of data to load from the hdf5 file (e.g. data/raw_data or data/image).",
    )
    parser.add_argument(
        "--image_range",
        type=int,
        nargs=2,
        default=None,
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
from ulsa import selection  # need to import this to update action selection registry
from ulsa.agent import Agent, AgentConfig, AgentState, hard_projection, setup_agent
from ulsa.downstream_task import downstream_task_registry
from ulsa.io_utils import (
    make_save_dir,
    map_range,
    plot_belief_distribution_for_presentation,
    plot_downstream_task_beliefs,
    plot_downstream_task_output_for_presentation,
    plot_frames_for_presentation,
)
from ulsa.ops import lines_rx_apo
from ulsa.pipeline import make_pipeline
from ulsa.utils import update_scan_for_polar_grid
from zea import File, Pipeline, Scan, log, set_data_paths
from zea.func import func_with_one_batch_dim, vmap
from zea.metrics import Metrics
from zea.utils import FunctionTimer


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


def apply_downstream_task(
    downstream_task: Optional[Callable], agent_config, targets, belief_distributions
):
    if downstream_task is None:
        return None, None, None, None
    else:
        n_frames, n_particles, h, w, c = ops.shape(belief_distributions)
        beliefs_stacked = ops.reshape(
            belief_distributions, (n_frames * n_particles, h, w, c)
        )
        beliefs_dst = vmap(
            downstream_task.call_generic,
            batch_size=agent_config.diffusion_inference.batch_size,
            fn_supports_batch=True,
        )(beliefs_stacked)
        _, h, w, c = ops.shape(beliefs_dst)
        beliefs_dst = ops.reshape(beliefs_dst, (n_frames, n_particles, h, w, c))
        reconstructions_dst = downstream_task.beliefs_to_reconstruction(beliefs_dst)
        targets_dst = vmap(
            downstream_task.call_generic,
            batch_size=agent_config.diffusion_inference.batch_size,
            fn_supports_batch=True,
        )(targets)
        return downstream_task, targets_dst, reconstructions_dst, beliefs_dst


@dataclass
class AgentResults:
    masks: np.ndarray
    target_imgs: np.ndarray
    reconstructions: np.ndarray
    belief_distributions: np.ndarray  # shape: (n_frames, particles, h, w, 1)
    measurements: np.ndarray
    saliency_map: np.ndarray

    def squeeze(self, axis=-1):
        if ops.all(self.saliency_map == None):
            self.saliency_map = ops.zeros_like(self.target_imgs)
        return AgentResults(
            np.squeeze(self.masks, axis=axis),
            np.squeeze(self.target_imgs, axis=axis),
            np.squeeze(self.reconstructions, axis=axis),
            np.squeeze(self.belief_distributions, axis=axis),
            np.squeeze(self.measurements, axis=axis),
            self.squeeze_if_not_none(self.saliency_map, axis=axis),
        )

    @staticmethod
    def squeeze_if_not_none(data, axis=-1):
        """
        Squeeze the data if it is not None.
        """
        if np.any(data == None):
            return None
        return np.squeeze(data, axis=axis)

    def to_uint8(self, input_range=None):
        """
        Convert the results to uint8 format, mapping the input range to (0, 255).
        """

        def map_to_uint8(data):
            if data is None:
                return None
            return map_range(data, input_range, (0, 255)).astype(np.uint8)

        return AgentResults(
            self.masks,  # keep masks as is
            map_to_uint8(self.target_imgs),
            map_to_uint8(self.reconstructions),
            map_to_uint8(self.belief_distributions),
            map_to_uint8(self.measurements),
            self.saliency_map,  # keep saliency map as is
        )


def run_active_sampling(
    agent: Agent,
    agent_state: AgentState,
    target_sequence,
    pipeline: Pipeline = None,
    scan: Scan = None,
    hard_project=False,
    verbose=True,
    post_pipeline: Pipeline = None,
    bandwidth: float = 2e6,
    return_timings=False,
) -> AgentResults:
    if verbose:
        log.info(log.blue("Running active sampling"))
        agent.print_summary()

    # Prepare acquisition function
    if getattr(scan, "n_tx", None) is not None and scan.n_tx > 1:
        rx_apo = lines_rx_apo(scan.n_tx, scan.grid_size_z, scan.grid_size_x)
        bandpass_rf = zea.func.get_band_pass_filter(
            128,
            scan.sampling_frequency,
            scan.demodulation_frequency - bandwidth / 2,
            scan.demodulation_frequency + bandwidth / 2,
        )
        base_params = pipeline.prepare_parameters(
            scan=scan,
            rx_apo=rx_apo,
            bandwidth=bandwidth,
            bandpass_rf=bandpass_rf,
            minval=0,
        )

        def acquire(full_data, mask, pipeline_state: dict):
            # Run pipeline with full data
            output = pipeline(data=full_data, **(base_params | pipeline_state))
            target = output["data"]

            # We use the same maxval & dynamic range for target and measurements.
            # This is based on the first frame of the target sequence and should not change
            # afterwards. You could predetermine it, so it is fine to use the target sequence
            # for it here.
            maxval = output["maxval"]
            dynamic_range = output["dynamic_range"]
            pipeline_state = {"maxval": maxval, "dynamic_range": dynamic_range}

            # This is done to ensure that the measurements are 0 where the mask is 0.
            # Assumes the pipeline beamforms the data to individual lines.
            # In this repo we use `rx_apo` to achieve this.
            measurements = target * mask

            return measurements, target, pipeline_state

    else:
        if scan is not None:
            params = pipeline.prepare_parameters(dynamic_range=scan.dynamic_range)
        else:
            params = {}

        def acquire(
            full_data,
            mask,
            pipeline_state: dict,
        ):
            target = pipeline(data=full_data, **params, **pipeline_state)["data"]
            return target * mask, target, {}

    def perception_action_step(agent_state: AgentState, target_data):
        # 1. Acquire measurements
        current_mask = agent_state.mask[..., -1, None]
        measurements, target_img, pipeline_state = acquire(
            target_data, current_mask, agent_state.pipeline_state
        )

        # 2. run perception and action selection via agent.recover
        reconstruction, new_agent_state = agent.recover(measurements, agent_state)

        if hard_project:
            reconstruction = hard_projection(reconstruction, measurements)

        new_agent_state.pipeline_state = pipeline_state
        return (
            new_agent_state,
            (
                reconstruction,
                current_mask,
                target_img,
                new_agent_state.belief_distribution,
                measurements,
                new_agent_state.saliency_map,
            ),
        )

    if verbose:
        print(f"Running active sampling for {len(target_sequence)} frames...")

    if return_timings:
        timer = FunctionTimer()
        perception_action_step = timer(perception_action_step)

    # Initial recover -> full number of diffusion steps
    # Subsequent percetion_action uses SeqDiff
    _, outputs = simple_scan(
        perception_action_step,
        agent_state,
        target_sequence,
        disable_tqdm=not verbose,
    )

    (
        reconstructions,
        masks,
        target_imgs,
        belief_distributions,
        measurements,
        saliency_map,
    ) = outputs

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

    agent_results = AgentResults(
        masks,
        target_imgs,
        reconstructions,
        belief_distributions,
        measurements,
        saliency_map,
    )

    if not return_timings:
        return agent_results
    else:
        return agent_results, timer.timings["perception_action_step"]


def preload_data(
    file: File,
    n_frames: int,  # if there are less than n_frames, it will load all frames
    data_type="data/image",
    type="focused",  # 'focused' or 'diverging'
):
    # Get scan from file
    try:
        scan = file.scan()
    except:
        scan = Scan(n_tx=1)

    is_in_house_dataset = scan.n_tx and data_type == "data/raw_data"
    if is_in_house_dataset:  # TODO: does echonet still work?
        log.info("Assuming the data file is from the in-house dataset!")
        _type = "unfocused" if type == "diverging" else type
        scan.set_transmits(_type)
        update_scan_for_polar_grid(scan)

    # slice(None) means all frames.
    if data_type in ["data/raw_data"]:
        validation_sample_frames = file.load_data(
            data_type, (slice(n_frames), scan.selected_transmits)
        )
    else:
        validation_sample_frames = file.load_data(data_type, slice(n_frames))

    # just for debugging
    # if data_type == "data/image_3D":
    #     _, data_n_ax, data_n_az, data_n_elev = validation_sample_frames.shape
    #     slice_az = 2
    #     crop_az = (data_n_az // 2) - slice_az
    #     validation_sample_frames = validation_sample_frames[:,:,crop_az:-crop_az,:]

    return validation_sample_frames, scan


def active_sampling_single_file(
    agent_config: str,
    target_sequence: str | Path = None,
    data_type: str = None,
    image_range: tuple = "unset",  # Set to None for auto-dynamic range
    seed: int = 42,
    override_config=None,
    jit_mode="recover",
    return_timings=False,
    map_type="vmap",
    **kwargs,
):
    data_paths = set_data_paths("users.yaml", local=False)
    data_root = data_paths["data_root"]

    agent_config: AgentConfig = AgentConfig.from_yaml(agent_config)
    agent_config.fix_paths()
    if override_config is not None:
        agent_config.update_recursive(override_config)

    if target_sequence is None:
        try:
            target_sequence = agent_config.data.target_sequence
        except:
            raise ValueError(
                "No target_sequence provided and not found in agent_config.data."
            )

    if data_type is None:
        try:
            data_type = agent_config.data.data_type
        except:
            raise ValueError(
                "No data_type provided and not found in agent_config.data."
            )

    if image_range == "unset":
        try:
            image_range = agent_config.data.image_range
        except:
            raise ValueError(
                "No image_range provided and not found in agent_config.data."
            )
    dynamic_range = image_range

    dataset_path = target_sequence.format(data_root=data_root)
    with File(dataset_path) as file:
        n_frames = agent_config.io_config.get("frame_cutoff", "all")
        validation_sample_frames, scan = preload_data(file, n_frames, data_type)
        scan.dynamic_range = dynamic_range
        agent_config.action_selection.set_n_tx(scan.n_tx)

    if getattr(scan, "theta_range", None) is not None:
        theta_range_deg = np.rad2deg(scan.theta_range)
        log.warning(
            f"Overriding scan conversion angles using the scan object: {theta_range_deg}"
        )
        agent_config.io_config.scan_conversion_angles = list(theta_range_deg)

    agent, agent_state = setup_agent(
        agent_config,
        seed=jax.random.PRNGKey(seed),
        jit_mode=jit_mode,
        map_type=map_type,
    )

    pipeline = make_pipeline(
        data_type=data_type,
        output_range=agent.input_range,
        output_shape=agent.input_shape,
        action_selection_shape=agent_config.action_selection.shape,
        **kwargs,
    )

    post_pipeline = Pipeline(
        [zea.ops.Lambda(keras.layers.CenterCrop(*agent_config.action_selection.shape))],
        with_batch_dim=True,
    )

    results = run_active_sampling(
        agent,
        agent_state,
        validation_sample_frames,
        pipeline=pipeline,
        scan=scan,
        hard_project=agent_config.diffusion_inference.hard_project,
        post_pipeline=post_pipeline,
        return_timings=return_timings,
    )

    if return_timings:
        results, timings = results

    if agent_config.downstream_task is not None:
        downstream_task = downstream_task_registry[agent_config.downstream_task](
            batch_size=agent_config.diffusion_inference.batch_size
        )
    else:
        downstream_task = None

    if downstream_task is not None:
        # Load downstream task model and apply to targets and reconstructions for comparison
        targets_normalized = zea.func.translate(
            validation_sample_frames, range_from=dynamic_range, range_to=(-1, 1)
        )
        downstream_task, targets_dst, reconstructions_dst, beliefs_dst = (
            apply_downstream_task(
                downstream_task,
                agent_config,
                targets_normalized[..., None],
                results.belief_distributions,
            )
        )
    else:
        targets_dst = None
        reconstructions_dst = None
        beliefs_dst = None

    to_return = [
        results,
        downstream_task,
        targets_dst,
        reconstructions_dst,
        beliefs_dst,
        agent,
        agent_config,
        dataset_path,
    ]
    if return_timings:
        to_return.append(timings)
    return tuple(to_return)


def compute_metrics(results, agent, metric_keys=["lpips", "psnr"]):
    metrics = Metrics(
        metrics=metric_keys,
        image_range=[0, 255],
    )
    denormalized = results.to_uint8(agent.input_range)
    metrics_results = metrics(denormalized.target_imgs, denormalized.reconstructions)
    print("\nMETRICS:")
    for k, v in metrics_results.items():
        print(f"{k:>8}: {float(v):.4f}")
    print("\n")


def save_results(
    results,
    downstream_task,
    targets_dst,
    reconstructions_dst,
    beliefs_dst,
    agent,
    agent_config,
    dataset_path,
    save_dir,
):
    data_paths = set_data_paths("users.yaml", local=False)
    output_dir = data_paths["output"]
    save_dir = save_dir.format(output_dir=output_dir)
    save_dir = Path(save_dir)
    run_dir, run_id = make_save_dir(save_dir)
    log.info(f"Run dir created at {log.yellow(run_dir)}")

    compute_metrics(results, agent)

    if agent_config.io_config.plot_frames_for_presentation:
        postfix_filename = Path(dataset_path).stem
        squeezed_results = results.squeeze(-1)

        for frame_to_plot in [0]:
            plot_belief_distribution_for_presentation(
                save_dir / run_id,
                squeezed_results.belief_distributions[frame_to_plot],
                squeezed_results.masks[frame_to_plot],
                agent_config.io_config,
                frame_idx=frame_to_plot,
                next_masks=squeezed_results.masks[frame_to_plot + 1],
            )
            if downstream_task is not None:
                plot_downstream_task_beliefs(
                    save_dir / run_id,
                    squeezed_results.belief_distributions[frame_to_plot],
                    np.squeeze(beliefs_dst[frame_to_plot]),
                    downstream_task,
                    squeezed_results.target_imgs[frame_to_plot],
                    np.squeeze(targets_dst)[frame_to_plot],
                    agent_config.io_config,
                    frame_to_plot,
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

        if downstream_task is not None:
            plot_downstream_task_output_for_presentation(
                save_dir / run_id,
                squeezed_results.target_imgs,
                squeezed_results.measurements,
                squeezed_results.reconstructions,
                np.std(
                    squeezed_results.belief_distributions, axis=1
                ),  # posterior std per frame
                downstream_task,
                np.squeeze(reconstructions_dst),  # segmentation masks
                np.squeeze(targets_dst),  # segmentation masks
                np.squeeze(
                    np.log(results.saliency_map + 1e-2)
                ),  # NOTE: tweak the +1e-2 for visualization
                agent_config.io_config,
                image_range=agent.input_range,
            )

    with open(save_dir / run_id / "config.json", "w") as json_file:
        json.dump(agent_config, json_file, indent=4)

    return run_dir, run_id


if __name__ == "__main__":
    print(f"Using {backend.backend()} backend 🔥")
    (
        results,
        downstream_task,
        targets_dst,
        reconstructions_dst,
        beliefs_dst,
        agent,
        agent_config,
        dataset_path,
    ) = active_sampling_single_file(
        args.agent_config,
        args.target_sequence,
        args.data_type,
        args.image_range,
        args.seed,
        args.override_config,
    )
    run_dir, run_id = save_results(
        results,
        downstream_task,
        targets_dst,
        reconstructions_dst,
        beliefs_dst,
        agent,
        agent_config,
        dataset_path,
        args.save_dir,
    )
