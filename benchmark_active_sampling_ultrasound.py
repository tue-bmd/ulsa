"""
Script for running an ultrasound line-scanning agent that chooses which lines to scan
based on samples from a distribution over full images conditioned on the lines observed
so far.
"""

import argparse
import os


def parse_args():
    """Parse arguments for training DDIM."""
    parser = argparse.ArgumentParser(description="DDIM inference")
    parser.add_argument(
        "--agent_config",
        type=str,
        default="./active_sampling/configs/ultrasound/agent_v3_echonet_3_frames.yaml",
        help="Path to agent config yaml.",
    )
    parser.add_argument(
        "--sweep_names",
        nargs="+",  # Allow multiple arguments
        default=["action_selection.n_actions"],
        help="Which config parameters to sweep on (multiple allowed).",
        choices=[
            "action_selection.selection_strategy",
            "action_selection.n_actions",
            "diffusion_inference.initial_step",
        ],
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="jax",
        help="ML backend to use",
        choices=["tensorflow", "torch", "jax"],
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="/mnt/z/Ultrasound-BMd/data/oisin/echonet_val_debug",
        help="A folder containing an ordered sequence of frames to sample from.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./output",
        help="Directory in which to save results",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="data/image",
        help="Data key for loading hdf5",
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
        "--device",
        type=str,
        default="auto:1",
        help="GPU device index",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["KERAS_BACKEND"] = args.backend
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from zea import init_device

    init_device()

import logging
from collections import defaultdict
from itertools import product
from pathlib import Path

import jax
import keras
import numpy as np
import zea
from keras import ops
from zea import Config, Dataset, init_device, log, set_data_paths
from zea.config import Config
from zea.data.augmentations import RandomCircleInclusion

from active_sampling_temporal import (
    apply_downstream_task,
    fix_paths,
    make_pipeline,
    preload_data,
    run_active_sampling,
)
from ulsa.agent import reset_agent_state, setup_agent
from ulsa.downstream_task import compute_dice_score
from ulsa.io_utils import make_save_dir
from ulsa.metrics import Metrics

# Default parameter paths and values
DEFAULT_SWEEP_VALUES = {
    "diffusion_inference.initial_step": [
        400,
        420,
        440,
        460,
        480,
        490,
        492,
        494,
        496,
        498,
    ],
    "action_selection.selection_strategy": [
        "covariance",
        "greedy_entropy",
        "uniform_random",
        "equispaced",
    ],
    "action_selection.n_actions": [28, 14, 7],
    "none": [None],
}


def get_sweep_param_values(config, sweep_name, custom_sweep_values=None):
    """Get parameter values for sweeping.

    Args:
        config: Base configuration
        sweep_name: Name of parameter to sweep
        custom_sweep_values: Optional dict overriding default sweep values
    """
    sweep_values = custom_sweep_values or DEFAULT_SWEEP_VALUES

    if sweep_name == "none":
        return sweep_values["none"]
    if sweep_name == "initial_step":
        num_steps = config.diffusion_sampler.num_steps
        return sweep_values["initial_step"].get(
            num_steps, DEFAULT_SWEEP_VALUES["initial_step"][num_steps]
        )
    if sweep_name in sweep_values:
        return sweep_values[sweep_name]
    raise UserWarning(f"Unhandled sweep parameter: {sweep_name}")


def get_config_value(config, key_path: str):
    """Get value from config using dot notation path"""
    ref = config
    for key in key_path.split("."):
        ref = ref[key]
    return ref


def update_config_value(base_config: Config, key_path: str, value):
    """
    Given a key path like 'a.b.c' and value X
    sets base_config['a']['b']['c'] = X and returns the config

    Args:
        base_config: Config object to update
        key_path: String of dot-separated keys (e.g. 'action_selection.selection_strategy')
        value: Value to set at the specified path

    Raises:
        KeyError: If any part of the key path doesn't exist in the config
    """
    updated_config = base_config.copy()
    ref = updated_config
    key_chain = key_path.split(".")

    # Check if path exists before modifying
    for key in key_chain[:-1]:
        if key not in ref:
            raise KeyError(f"Config path '{key_path}' is invalid: '{key}' not found")
        ref = ref[key]

    if key_chain[-1] not in ref:
        raise KeyError(
            f"Config path '{key_path}' is invalid: '{key_chain[-1]}' not found"
        )

    ref[key_chain[-1]] = value
    return updated_config


def setup_sweep(agent_config: Config, sweep_params):
    """Sets up sweep configurations as cross product of all parameter combinations

    Args:
        agent_config: Base configuration
        sweep_params: Dict mapping parameter paths to lists of values
            e.g. {'action_selection.selection_strategy': ['covariance', 'entropy']}
        save_dir: Directory to save sweep results
    """
    if not sweep_params:
        return [agent_config], None

    param_paths = list(sweep_params.keys())
    param_values = list(sweep_params.values())

    # Generate all combinations
    sweep_configs = []
    for value_combination in product(*param_values):
        config = agent_config.copy()
        # Apply each parameter value in the combination
        for param_path, value in zip(param_paths, value_combination):
            config = update_config_value(config, param_path, value)
        sweep_configs.append(config)

    # Update sweep details to include all parameters
    sweep_details = Config(
        dictionary={
            "sweep_param_names": param_paths,
            "sweep_param_values": [list(values) for values in param_values],
        }
    )

    return sweep_configs, sweep_details


def _setup_sweep_files(save_dir: Path, agent_config: Config, sweep_details: Config):
    base_config_path = save_dir / "base_config.yaml"
    if not base_config_path.exists():
        agent_config.save_to_yaml(base_config_path)

    sweep_details_path = save_dir / "sweep_details.yaml"
    if not sweep_details_path.exists():
        sweep_details.save_to_yaml(sweep_details_path)


def get_target_files(target_dir, limit=None):
    if target_dir.is_dir():
        return [
            os.path.join(target_dir, file_name)
            for file_name in os.listdir(target_dir)
            if file_name.endswith("hdf5")
        ][:limit]
    elif target_dir.is_file():
        return [target_dir]
    else:
        raise UserWarning("❗️ target_dir should point to a file or directory.")


def benchmark(
    agent_config: Config,
    dataset: Dataset,
    dynamic_range: tuple,
    file_indices=None,
    n_frames=30,
    seed=None,
    model=None,
    jit_mode="recover",
    circle_augmentation=None,
    save_dir=None,
    metrics=None,
):
    # Not sure if I have to reinit the agent every time?
    seed, seed_1 = jax.random.split(seed)
    agent, state = setup_agent(
        agent_config, seed=seed_1, model=model, jit_mode=jit_mode
    )

    jit_options = None if jit_mode == "off" else "pipeline"
    pipeline = make_pipeline(
        dataset.key,
        dynamic_range,
        agent.input_range,
        agent.input_shape,
        agent_config.action_selection.shape,
        jit_options=jit_options,
    )
    if agent_config.get("is_3d", False):
        # we don't need any post-cropping for the 3d case
        post_pipeline = None
    else:
        post_pipeline = zea.ops.Pipeline(
            [
                zea.ops.Lambda(
                    keras.layers.CenterCrop(*agent_config.action_selection.shape)
                )
            ],
            with_batch_dim=True,
            jit_options=jit_options,
        )

    if metrics is None:
        metrics = Metrics(
            # ssim can cause OOMs with 3d data...
            # TODO: batchify ssim metric
            metrics=["mae", "mse", "psnr"],  # , "ssim"],
            image_range=[0, 255],
        )

    if file_indices is None:
        file_indices = range(len(dataset))
    if isinstance(file_indices, int):
        file_indices = [file_indices]  # in case the dataset is of size 1

    all_metrics_results = []
    for i, file_index in enumerate(file_indices):
        file = dataset[file_index]
        cardiac = "cardiac" in str(file.path)
        target_sequence, scan, probe = preload_data(
            file,
            n_frames,
            dataset.key,
            dynamic_range,
            cardiac=cardiac,
        )

        if circle_augmentation is not None:
            seed, seed_1 = jax.random.split(seed)
            target_sequence, centers = circle_augmentation(target_sequence, seed=seed_1)

        # 3d batch
        batch_size = (
            None
            if agent_config.get("is_3d", None) is None
            else ops.shape(target_sequence)[2]
        )
        state = reset_agent_state(agent, seed, batch_size=batch_size)
        results = run_active_sampling(
            agent,
            state,
            target_sequence,
            agent_config.action_selection.n_actions,
            pipeline,
            scan,
            probe,
            hard_project=agent_config.diffusion_inference.hard_project,
            verbose=False,
            post_pipeline=post_pipeline,
        )

        dst_output_type, downstream_task_outputs = apply_downstream_task(
            agent_config, results.reconstructions
        )

        denormalized = results.to_uint8(agent.input_range)
        metrics_results = metrics.eval_metrics(
            denormalized.target_imgs, denormalized.reconstructions
        )
        all_metrics_results.append(metrics_results)
        if circle_augmentation is not None:
            recovered_circle_accuracy = (
                circle_augmentation.evaluate_recovered_circle_accuracy(
                    denormalized.reconstructions,
                    centers,
                    recovery_threshold=10.0,  # must be within 10 pixel values
                    fill_value=128.0,  # translate to pixel space
                )
            )
            recovered_circle_kwargs = {
                "recovered_circle": [
                    {
                        "recovery_threshold": 10.0,
                        "fill_value": 128.0,
                        "centers": centers,
                        "accuracy": ops.convert_to_numpy(recovered_circle_accuracy),
                    }
                ],
            }
        else:
            recovered_circle_kwargs = {"recovered_circle": None}

        # Save outputs
        if save_dir is not None:
            run_dir, run_id = make_save_dir(save_dir)
            Config(target_filepath=file.path).save_to_yaml(
                save_dir / run_id / "target_filepath.yaml"
            )
            agent_config.save_to_yaml(save_dir / run_id / "config.yaml")
            outpath = save_dir / run_id / "metrics.npz"
            np.savez(
                outpath,
                targets=denormalized.target_imgs,
                reconstructions=denormalized.reconstructions,
                masks=denormalized.masks,
                measurements=denormalized.measurements,
                belief_distributions=denormalized.belief_distributions,
                **metrics_results,
                **recovered_circle_kwargs,
                **(
                    {dst_output_type: downstream_task_outputs}
                    if downstream_task_outputs is not None
                    else {}
                ),
            )
            log.info(
                f"Saved results for target {i}/{len(file_indices)} (file index: {file_index}) "
                + f"at {log.yellow(outpath)}"
            )

    return (
        all_metrics_results,
        dst_output_type,
        downstream_task_outputs,
        recovered_circle_kwargs,
        results,  # last result of loop
        agent,
    )


def group_by_first(indices):
    grouped = {}
    for k, v in indices:
        grouped.setdefault(k, []).append(v)
    return grouped


def get_shard_indices(shard_index, num_shards, *lengths):
    """
    Returns the indices of the Cartesian product of the iterables assigned to the given shard.
    Each index is a tuple of indices (one per iterable).

    Args:
        shard_index (int): Index of the current shard (0-based).
        num_shards (int): Total number of shards.
        *lengths: Any number of iterables.

    Returns:
        List[Tuple[int, ...]]: List of index tuples for this shard.
    """

    all_indices = product(*(range(n) for n in lengths))

    if shard_index is None or num_shards is None:
        return list(all_indices)

    assert num_shards > 0 and shard_index < num_shards, (
        "num_shards must be > 0 and shard_index must be < num_shards"
    )

    log.info(f"Sharding: {num_shards} shards, current shard index: {shard_index}")

    # Assign each combination to a shard in round-robin fashion
    result = [
        idxs for i, idxs in enumerate(all_indices) if i % num_shards == shard_index
    ]
    assert len(result) > 0, (
        f"No indices found for shard {shard_index} with {num_shards} shards and lengths {lengths}. "
        "Possibly too many shards?"
    )
    return result


def run_benchmark(
    agent_config,
    target_dir,
    save_dir: Path,
    sweep_params,  # Now a single parameter that can be either dict or list
    limit_n_frames=None,  # override agent base config
    limit_n_samples=None,
    image_range=(-60, 0),
    circle_augmentation: RandomCircleInclusion = None,
    initial_run_key=jax.random.PRNGKey(42),
    data_type="data/image",
    num_shards=None,
    shard_index=None,
    validate_dataset=True,
):
    """
    Run benchmarking for ultrasound line-scanning agents.

    Args:
        agent_config: Base configuration for the agent
        target_dir: Directory containing target data files
        save_dir: Directory to save results
        sweep_params: Either:
            - List of parameter names to sweep using DEFAULT_SWEEP_VALUES
            - Dict with parameter names as keys and lists of values to sweep over
    """
    agent_config = fix_paths(agent_config)

    sweep_configs, sweep_details = setup_sweep(agent_config, sweep_params)

    if num_shards is not None and shard_index is not None:
        sweep_save_dir = save_dir
        os.makedirs(sweep_save_dir, exist_ok=True)
    else:
        sweep_save_dir, _ = make_save_dir(save_dir, prefix="sweep")

    if sweep_details is not None:
        _setup_sweep_files(sweep_save_dir, agent_config, sweep_details)

    log.info(
        f"Starting sweep on multiple parameters:\n"
        + "\n".join(
            [
                f"- [{log.green(param_path)}] over values {log.green(values)}"
                for param_path, values in sweep_params.items()
            ]
        )
    )

    # Set up logging
    logging.basicConfig(
        filename=sweep_save_dir / "log.txt",
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    limit_n_frames = (
        limit_n_frames
        if limit_n_frames is not None
        else agent_config.io_config.frame_cutoff
    )

    run_level_subkey = initial_run_key.copy()

    dataset = Dataset(target_dir, key=data_type, validate=validate_dataset)

    if limit_n_samples is None:
        limit_n_samples = len(dataset)

    indices_per_iterable = get_shard_indices(
        shard_index, num_shards, len(sweep_configs), limit_n_samples
    )
    indices_per_iterable = group_by_first(indices_per_iterable)
    # TODO: might be good to store the shard index of each run so you can check
    # if the benchmark is complete later

    for sweep_idx, file_indices in indices_per_iterable.items():
        sweep_config = sweep_configs[sweep_idx]
        out = benchmark(
            sweep_config,
            dataset,
            image_range,
            file_indices,
            limit_n_frames,
            run_level_subkey,
            circle_augmentation=circle_augmentation,
            save_dir=sweep_save_dir,
        )
        all_metrics_results = out[0]

    return sweep_save_dir, all_metrics_results


def extract_sweep_data(
    sweep_dir: str,
    keys_to_extract=["mse", "psnr"],
    x_axis_key="action_selection.n_actions",
    strategies_to_plot=None,
    gt_masks=None,
    include_only_these_files=None,
):
    """Can be used to load all the metrics from the run_benchmark function."""
    sweep_details_path = os.path.join(sweep_dir, "sweep_details.yaml")
    if not os.path.exists(sweep_details_path):
        raise FileNotFoundError(f"Missing sweep_details.yaml in {sweep_dir}")

    sweep_details = Config.from_yaml(sweep_details_path)

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Loop over runs in the sweep directory
    for run_dir in sorted(os.listdir(sweep_dir)):
        # Check if run_dir is a directory
        run_path = os.path.join(sweep_dir, run_dir)
        if not os.path.isdir(run_path):
            continue

        # Print run directory being processed in place
        print(f"Processing run: {run_dir}", end="\r")

        # Check for required files
        config_path = os.path.join(run_path, "config.yaml")
        metrics_path = os.path.join(run_path, "metrics.npz")
        filepath_yaml = os.path.join(run_path, "target_filepath.yaml")
        if not all(
            os.path.exists(p) for p in [config_path, metrics_path, filepath_yaml]
        ):
            continue

        # Load the config, metrics, and target file path
        config = Config.from_yaml(config_path)
        metrics = np.load(metrics_path, allow_pickle=True)
        target_file = Config.from_yaml(filepath_yaml)["target_filepath"]

        # Hack to get snellius results to work locally
        target_file = str(target_file).replace(
            "/projects/0/prjs0966/data", "/mnt/z/Ultrasound-BMd/data"
        )

        # Filter by include_only_these_files if specified
        if (
            include_only_these_files is not None
            and target_file not in include_only_these_files
        ):
            continue

        # Check if the x-axis key exists in the config
        x_value = get_config_value(config, x_axis_key)
        if x_value is None:
            log.warning(f"Skipping {run_path} due to missing x_axis_key: {x_axis_key}.")
            continue

        # Check if the selection strategy is in the config
        selection_strategy = config.get("action_selection", {}).get(
            "selection_strategy"
        )
        if selection_strategy is None:
            log.warning(
                f"Skipping {run_path} due to missing selection_strategy: {selection_strategy}."
            )
            continue

        # Skip if the selection strategy is not in the specified strategies
        # This saves time by not processing runs that are not needed
        if (
            strategies_to_plot is not None
            and selection_strategy not in strategies_to_plot
        ):
            continue

        # Extract prediction masks and images (and skip if not available)
        if "reconstructions" not in metrics or "masks" not in metrics:
            continue
        pred_images = metrics["reconstructions"]

        # Store masks and images in results
        # TODO: what is this used for @OisinNolan?
        # results["masks"][selection_strategy][x_value].append(
        #     {"masks": pred_masks, "x_scan_converted": pred_images, "run_dir": run_path}
        # )

        # Compute DICE if ground truths available
        if gt_masks is not None and target_file in gt_masks:
            pred_masks = metrics["segmentation_mask"]
            gt_sequence = gt_masks[target_file]
            pred_masks = np.array(pred_masks)
            gt_masks_array = np.array(gt_sequence["masks"])
            gt_masks_array = np.squeeze(gt_masks_array, axis=1)
            dice_score = compute_dice_score(pred_masks, gt_masks_array)

            # Store mean DICE over patient frames
            results["dice"][selection_strategy][x_value].append(np.mean(dice_score))

        # Add other metrics
        for metric_name in keys_to_extract:
            if metric_name not in metrics:
                continue  # Skip if metric not found

            metric_values = metrics[metric_name]
            if isinstance(metric_values, np.ndarray) and metric_values.size > 0:
                sequence_means = np.mean(metric_values, axis=-1)
                results[metric_name][selection_strategy][x_value].append(sequence_means)

    return results


if __name__ == "__main__":
    print(f"Using {keras.backend.backend()} backend 🔥")
    data_paths = set_data_paths("users.yaml", local=False)

    agent_config = Config.from_yaml(args.agent_config)
    sweep_save_dir, metrics_results = run_benchmark(
        agent_config=agent_config,
        target_dir=args.target_dir.format(data_root=data_paths["data_root"]),
        save_dir=args.save_dir,
        sweep_params={
            sweep_name: DEFAULT_SWEEP_VALUES[sweep_name]
            for sweep_name in args.sweep_names
        },
        data_type=args.data_type,
        image_range=args.image_range,
        validate_dataset=False,
    )
