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
        default="./configs/echonet_3_frames.yaml",
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
from keras import ops

import zea
from active_sampling_temporal import (
    apply_downstream_task,
    fix_paths,
    make_pipeline,
    preload_data,
    run_active_sampling,
)
from ulsa.agent import reset_agent_state, setup_agent
from ulsa.downstream_task import (
    DownstreamTask,
    EchoNetLVHSegmentation,
    compute_dice_score,
    downstream_task_registry,
)
from ulsa.io_utils import make_save_dir
from ulsa.metrics import Metrics
from zea import Config, Dataset, init_device, log, set_data_paths
from zea.config import Config
from zea.data.augmentations import RandomCircleInclusion

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
        if agent_config.get("is_3d", False):
            metrics_to_compute = ["mae", "mse", "psnr", "lpips"]
        else:
            metrics_to_compute = ["mae", "mse", "psnr", "ssim", "lpips"]
        metrics = Metrics(
            metrics=metrics_to_compute,
            image_range=[0, 255],
        )

    if agent_config.downstream_task is not None:
        downstream_task = downstream_task_registry[agent_config.downstream_task](
            batch_size=agent_config.diffusion_inference.batch_size
        )
    else:
        downstream_task = None

    if file_indices is None:
        file_indices = range(len(dataset))
    if isinstance(file_indices, int):
        file_indices = [file_indices]  # in case the dataset is of size 1

    all_metrics_results = []
    for i, file_index in enumerate(file_indices):
        file = dataset[file_index]
        target_sequence, scan, probe = preload_data(file, n_frames, dataset.key)
        scan.dynamic_range = dynamic_range

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

        target_sequence_preprocessed = zea.utils.translate(
            target_sequence[..., None], dynamic_range, (-1, 1)
        )
        downstream_task, targets_dst, reconstructions_dst, _ = apply_downstream_task(
            downstream_task,
            agent_config,
            target_sequence_preprocessed,
            results.belief_distributions,
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
                    {
                        f"{downstream_task.output_type()}_targets": targets_dst,
                        f"{downstream_task.output_type()}_reconstructions": reconstructions_dst,
                    }
                    if downstream_task is not None
                    else {}
                ),
            )
            log.info(
                f"Saved results for target {i + 1}/{len(file_indices)} (file index: {file_index}) "
                + f"at {log.yellow(outpath)}"
            )

    return (
        all_metrics_results,
        downstream_task.output_type() if downstream_task is not None else None,
        {
            f"{downstream_task.output_type()}_targets": targets_dst,
            f"{downstream_task.output_type()}_reconstructions": reconstructions_dst,
        }
        if downstream_task is not None
        else {},
        recovered_circle_kwargs,
        results,  # last result of loop
        agent,
    )


def group_by_first(indices):
    """
    Groups a list of tuples by the first element of each tuple.

    Args:
        indices (Iterable[Tuple[Any, Any]]): List of tuples to group.

    Returns:
        Dict[Any, List[Any]]: Dictionary where each key is the first element,
        and the value is a list of the second elements from tuples with that key.
    """
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

    all_indices = list(product(*(range(n) for n in lengths)))

    if shard_index is None or num_shards is None:
        return all_indices

    assert num_shards > 0 and shard_index < num_shards, (
        "num_shards must be > 0 and shard_index must be < num_shards"
    )

    log.info(f"Sharding: {num_shards} shards, current shard index: {shard_index}")

    # Calculate the chunk size for each shard
    # Use ceiling division to ensure all indices are covered
    chunk_size = (len(all_indices) + num_shards - 1) // num_shards  # Ceiling division
    start = shard_index * chunk_size
    end = min(start + chunk_size, len(all_indices))
    result = all_indices[start:end]

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
    debug_sweep=False,
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
        zea.log.info(
            f":: Running sweep index: {sweep_idx + 1} / {len(indices_per_iterable)} ::"
        )
        sweep_config = sweep_configs[sweep_idx]

        if debug_sweep:
            log.warning("DEBUGGING SWEEP!")
            limit_n_frames = 4
            file_indices = file_indices[:2]  # only run on first 2 files

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
        agent = out[-1]
        del agent  # for garbage collection

    return sweep_save_dir, all_metrics_results


def extract_sweep_data(
    sweep_dir: str,
    keys_to_extract=["mse", "psnr", "dice"],
    x_axis_key="action_selection.n_actions",
    strategies_to_plot=None,
    include_only_these_files=None,
):
    """Load all metrics from a sweep directory. Now uses segmentation_mask_targets and segmentation_mask_reconstructions from metrics.npz."""
    sweep_details_path = os.path.join(sweep_dir, "sweep_details.yaml")
    if not os.path.exists(sweep_details_path):
        raise FileNotFoundError(f"Missing sweep_details.yaml in {sweep_dir}")

    sweep_details = Config.from_yaml(sweep_details_path)
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for run_dir in sorted(os.listdir(sweep_dir)):
        run_path = os.path.join(sweep_dir, run_dir)
        if not os.path.isdir(run_path):
            continue

        print(f"Processing run: {run_dir}", end="\r")

        config_path = os.path.join(run_path, "config.yaml")
        metrics_path = os.path.join(run_path, "metrics.npz")
        filepath_yaml = os.path.join(run_path, "target_filepath.yaml")
        if not all(
            os.path.exists(p) for p in [config_path, metrics_path, filepath_yaml]
        ):
            continue

        config = Config.from_yaml(config_path)
        metrics = np.load(metrics_path, allow_pickle=True)
        target_file = Config.from_yaml(filepath_yaml)["target_filepath"]

        # Optional file filter
        if (
            include_only_these_files is not None
            and target_file not in include_only_these_files
        ):
            continue

        x_value = get_config_value(config, x_axis_key)
        if x_value is None:
            log.warning(f"Skipping {run_path} due to missing x_axis_key: {x_axis_key}.")
            continue

        selection_strategy = config.get("action_selection", {}).get(
            "selection_strategy"
        )
        if selection_strategy is None:
            log.warning(f"Skipping {run_path} due to missing selection_strategy.")
            continue

        if (
            strategies_to_plot is not None
            and selection_strategy not in strategies_to_plot
        ):
            continue

        # Extract masks and images for comparison and plotting
        if (
            "segmentation_mask_targets" in metrics
            and "segmentation_mask_reconstructions" in metrics
        ):
            # TODO: converting to an array is slow here -- maybe faster way to do it?
            gt_masks = np.array(metrics[f"segmentation_mask_targets"])
            pred_masks = np.array(metrics[f"segmentation_mask_reconstructions"])
            if "dice" in keys_to_extract:
                dice_score = compute_dice_score(pred_masks, gt_masks)
                results["dice"][selection_strategy][x_value].append(np.mean(dice_score))
        if any(key.startswith("heatmap_center_mse_") for key in keys_to_extract):
            # Find which heatmap_center metric is requested
            heatmap_keys = [
                key for key in keys_to_extract if key.startswith("heatmap_center_mse_")
            ]

            for heatmap_key in heatmap_keys:
                # Extract measurement type from key (e.g., "heatmap_center_mse_LVPW" -> "LVPW")
                measurement_type = heatmap_key.replace("heatmap_center_mse_", "")

                # Validate measurement type
                valid_types = ["LVPW", "LVID", "IVS"]
                if measurement_type not in valid_types:
                    log.warning(
                        f"Invalid measurement type '{measurement_type}' in key '{heatmap_key}'. Valid types: {valid_types}"
                    )
                    continue

                gt_bottom_coords, gt_top_coords = (
                    EchoNetLVHSegmentation.outputs_to_coordinates(
                        gt_masks, measurement_type
                    )
                )
                pred_bottom_coords, pred_top_coords = (
                    EchoNetLVHSegmentation.outputs_to_coordinates(
                        pred_masks, measurement_type
                    )
                )
                bottom_mean_euclidean_distance = ops.mean(
                    [
                        ops.sqrt(
                            ops.sum((gt_bottom_coords[i] - pred_bottom_coords[i]) ** 2)
                        )
                        for i in range(len(gt_bottom_coords))
                    ]
                )
                top_mean_euclidean_distance = ops.mean(
                    [
                        ops.sqrt(ops.sum((gt_top_coords[i] - pred_top_coords[i]) ** 2))
                        for i in range(len(gt_top_coords))
                    ]
                )
                # NOTE: currently taking the mean over both points, but could report both
                results[heatmap_key][selection_strategy][x_value].append(
                    float(
                        (bottom_mean_euclidean_distance + top_mean_euclidean_distance)
                        / 2
                    )
                )

            # Store masks and images for plotting
            results["masks"][selection_strategy][x_value].append(
                {
                    "masks": pred_masks,
                    "x_scan_converted": metrics["reconstructions"],
                    "run_dir": run_path,
                    "gt_masks": gt_masks,
                }
            )

        if "heatmap_mse" in keys_to_extract:
            mse = ops.mean((gt_masks - pred_masks) ** 2)
            results["heatmap_mse"][selection_strategy][x_value].append(
                float(np.mean(mse))
            )
        if any(key.startswith("heatmap_entropy_") for key in keys_to_extract):
            # Find which heatmap_entropy metric is requested
            heatmap_entropy_keys = [
                key for key in keys_to_extract if key.startswith("heatmap_entropy_")
            ]

            for heatmap_entropy_key in heatmap_entropy_keys:
                # Extract measurement type from key (e.g., "heatmap_entropy_LVPW" -> "LVPW")
                measurement_type = heatmap_entropy_key.replace("heatmap_entropy_", "")

                # Validate measurement type
                valid_types = ["LVPW", "LVID", "IVS"]
                if measurement_type not in valid_types:
                    log.warning(
                        f"Invalid measurement type '{measurement_type}' in key '{heatmap_entropy_key}'. Valid types: {valid_types}"
                    )
                    continue

                # Get channel indices for this measurement type
                measurements_to_channels = {
                    "LVPW": [0, 1],
                    "LVID": [1, 2],
                    "IVS": [2, 3],
                }
                c1, c2 = measurements_to_channels[measurement_type]

                def compute_heatmap_entropy(heatmap):
                    """Compute entropy of heatmap treating it as categorical distribution"""
                    # Flatten the heatmap
                    flat_heatmap = heatmap.flatten()

                    # Add small epsilon to avoid log(0)
                    epsilon = 1e-10
                    flat_heatmap = flat_heatmap + epsilon

                    # Normalize to probability distribution
                    prob_dist = flat_heatmap / np.sum(flat_heatmap)

                    # Compute entropy: H = -sum(p * log(p))
                    entropy = -np.sum(prob_dist * np.log(prob_dist))
                    return entropy

                batch_entropies = []
                for batch_idx in range(gt_masks.shape[0]):
                    # Compute entropy for both GT and predicted heatmaps
                    gt_bottom_entropy = compute_heatmap_entropy(
                        gt_masks[batch_idx, ..., c1]
                    )
                    gt_top_entropy = compute_heatmap_entropy(
                        gt_masks[batch_idx, ..., c2]
                    )

                    pred_bottom_entropy = compute_heatmap_entropy(
                        pred_masks[batch_idx, ..., c1]
                    )
                    pred_top_entropy = compute_heatmap_entropy(
                        pred_masks[batch_idx, ..., c2]
                    )

                    # Compute absolute difference in entropy between GT and predicted
                    bottom_entropy_diff = abs(gt_bottom_entropy - pred_bottom_entropy)
                    top_entropy_diff = abs(gt_top_entropy - pred_top_entropy)

                    # Average the entropy difference for bottom and top points
                    avg_entropy_diff = (bottom_entropy_diff + top_entropy_diff) / 2
                    batch_entropies.append(avg_entropy_diff)

                results[heatmap_entropy_key][selection_strategy][x_value].append(
                    float(np.mean(batch_entropies))
                )

        if any(key.startswith("gaussian_emd_") for key in keys_to_extract):
            # Find which gaussian_emd metric is requested
            gaussian_emd_keys = [
                key for key in keys_to_extract if key.startswith("gaussian_emd_")
            ]

            for gaussian_emd_key in gaussian_emd_keys:
                # Extract measurement type from key (e.g., "gaussian_emd_LVPW" -> "LVPW")
                measurement_type = gaussian_emd_key.replace("gaussian_emd_", "")

                # Validate measurement type
                valid_types = ["LVPW", "LVID", "IVS"]
                if measurement_type not in valid_types:
                    log.warning(
                        f"Invalid measurement type '{measurement_type}' in key '{gaussian_emd_key}'. Valid types: {valid_types}"
                    )
                    continue

                # Get Gaussian parameters for ground truth and predictions
                gt_gaussian_params = EchoNetLVHSegmentation.outputs_to_gaussians(
                    gt_masks, measurement_type
                )
                pred_gaussian_params = EchoNetLVHSegmentation.outputs_to_gaussians(
                    pred_masks, measurement_type
                )

                # Compute 2D Wasserstein distance between Gaussians
                def gaussian_2d_wasserstein(params1, params2):
                    """Compute 2D Wasserstein distance between two Gaussians"""
                    # Extract parameters
                    mu1 = np.array([params1["center_x"], params1["center_y"]])
                    mu2 = np.array([params2["center_x"], params2["center_y"]])

                    # Create covariance matrices from sigma_x, sigma_y, theta
                    def make_cov_matrix(sigma_x, sigma_y, theta):
                        cos_t, sin_t = np.cos(theta), np.sin(theta)
                        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                        D = np.diag([sigma_x**2, sigma_y**2])
                        return R @ D @ R.T

                    cov1 = make_cov_matrix(
                        params1["sigma_x"], params1["sigma_y"], params1["theta"]
                    )
                    cov2 = make_cov_matrix(
                        params2["sigma_x"], params2["sigma_y"], params2["theta"]
                    )

                    # 2D Wasserstein distance formula for Gaussians
                    mean_diff = mu1 - mu2
                    cov_sqrt = scipy.linalg.sqrtm(cov1)
                    middle_term = scipy.linalg.sqrtm(cov_sqrt @ cov2 @ cov_sqrt)

                    return np.sqrt(
                        np.sum(mean_diff**2) + np.trace(cov1 + cov2 - 2 * middle_term)
                    )

                # Import scipy for matrix operations
                import scipy.linalg

                batch_emds = []
                for batch_idx in range(len(gt_gaussian_params["bottom"]["center_x"])):
                    # Compute EMD for bottom and top points
                    bottom_emd = gaussian_2d_wasserstein(
                        {
                            k: v[batch_idx]
                            for k, v in gt_gaussian_params["bottom"].items()
                        },
                        {
                            k: v[batch_idx]
                            for k, v in pred_gaussian_params["bottom"].items()
                        },
                    )
                    top_emd = gaussian_2d_wasserstein(
                        {k: v[batch_idx] for k, v in gt_gaussian_params["top"].items()},
                        {
                            k: v[batch_idx]
                            for k, v in pred_gaussian_params["top"].items()
                        },
                    )

                    # Average the EMD for bottom and top points
                    batch_emds.append((bottom_emd + top_emd) / 2)

                # DEBUG point
                if False:
                    import matplotlib.patches as patches
                    import matplotlib.pyplot as plt
                    from matplotlib.patches import Ellipse

                    # Create a grid to show a few samples
                    n_samples_to_show = min(4, gt_masks.shape[0])
                    # 6 rows: 3 for bottom channel, 3 for top channel
                    # n_samples_to_show columns
                    fig, axes = plt.subplots(
                        6, n_samples_to_show, figsize=(4 * n_samples_to_show, 18)
                    )
                    if n_samples_to_show == 1:
                        axes = axes.reshape(6, 1)

                    def plot_gaussian_ellipse(ax, params, color="red", alpha=0.7):
                        """Plot an ellipse representing the 2D Gaussian"""
                        center_x = params["center_x"]
                        center_y = params["center_y"]
                        sigma_x = params["sigma_x"]
                        sigma_y = params["sigma_y"]
                        theta = params["theta"]

                        # Convert theta from radians to degrees for matplotlib
                        angle_deg = np.degrees(theta)

                        # Create ellipse (2-sigma bounds for better visibility)
                        ellipse = Ellipse(
                            (center_x, center_y),
                            width=4 * sigma_x,  # 2-sigma in each direction
                            height=4 * sigma_y,
                            angle=angle_deg,
                            facecolor="none",
                            edgecolor=color,
                            linewidth=1.5,
                            alpha=alpha,
                            linestyle="--",
                        )
                        ax.add_patch(ellipse)

                        # Add smaller center point
                        ax.plot(
                            center_x,
                            center_y,
                            "o",
                            color=color,
                            markersize=3,
                            markeredgecolor="white",
                            markeredgewidth=0.5,
                        )

                    for sample_idx in range(n_samples_to_show):
                        # Get the channels for this measurement type
                        measurements_to_channels = {
                            "LVPW": [0, 1],
                            "LVID": [1, 2],
                            "IVS": [2, 3],
                        }
                        c1, c2 = measurements_to_channels[measurement_type]

                        # BOTTOM CHANNEL PLOTS (rows 0, 1, 2)
                        gt_bottom_heatmap = gt_masks[sample_idx, ..., c1]
                        pred_bottom_heatmap = pred_masks[sample_idx, ..., c1]

                        # Row 0: GT Bottom heatmap
                        ax_gt_bottom = axes[0, sample_idx]
                        ax_gt_bottom.imshow(gt_bottom_heatmap, cmap="viridis")
                        ax_gt_bottom.set_title(f"GT Bottom - Sample {sample_idx}")
                        ax_gt_bottom.set_xticks([])
                        ax_gt_bottom.set_yticks([])

                        # Row 1: Pred Bottom heatmap
                        ax_pred_bottom = axes[1, sample_idx]
                        ax_pred_bottom.imshow(
                            pred_bottom_heatmap,
                            cmap="viridis",
                        )
                        ax_pred_bottom.set_title(f"Pred Bottom - Sample {sample_idx}")
                        ax_pred_bottom.set_xticks([])
                        ax_pred_bottom.set_yticks([])

                        # Row 2: Combined Bottom with Gaussians
                        ax_combined_bottom = axes[2, sample_idx]
                        # Show GT as grayscale background
                        ax_combined_bottom.imshow(
                            gt_bottom_heatmap, cmap="gray", alpha=0.6
                        )
                        # Overlay pred as red heatmap
                        if np.max(pred_bottom_heatmap) > 0:
                            ax_combined_bottom.imshow(
                                pred_bottom_heatmap, cmap="Reds", alpha=0.5
                            )

                        # Plot fitted Gaussians
                        gt_bottom_params = {
                            k: v[sample_idx]
                            for k, v in gt_gaussian_params["bottom"].items()
                        }
                        pred_bottom_params = {
                            k: v[sample_idx]
                            for k, v in pred_gaussian_params["bottom"].items()
                        }

                        plot_gaussian_ellipse(
                            ax_combined_bottom,
                            gt_bottom_params,
                            color="cyan",
                            alpha=0.8,
                        )
                        plot_gaussian_ellipse(
                            ax_combined_bottom,
                            pred_bottom_params,
                            color="yellow",
                            alpha=0.8,
                        )

                        # Compute and display EMD
                        bottom_emd = gaussian_2d_wasserstein(
                            gt_bottom_params, pred_bottom_params
                        )
                        ax_combined_bottom.text(
                            0.02,
                            0.98,
                            f"EMD: {bottom_emd:.2f}",
                            transform=ax_combined_bottom.transAxes,
                            verticalalignment="top",
                            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
                            color="white",
                            fontsize=8,
                        )

                        ax_combined_bottom.set_title(
                            f"Combined Bottom - Sample {sample_idx}"
                        )
                        ax_combined_bottom.set_xticks([])
                        ax_combined_bottom.set_yticks([])

                        # TOP CHANNEL PLOTS (rows 3, 4, 5)
                        gt_top_heatmap = gt_masks[sample_idx, ..., c2]
                        pred_top_heatmap = pred_masks[sample_idx, ..., c2]

                        # Row 3: GT Top heatmap
                        ax_gt_top = axes[3, sample_idx]
                        ax_gt_top.imshow(gt_top_heatmap, cmap="viridis")
                        ax_gt_top.set_title(f"GT Top - Sample {sample_idx}")
                        ax_gt_top.set_xticks([])
                        ax_gt_top.set_yticks([])

                        # Row 4: Pred Top heatmap
                        ax_pred_top = axes[4, sample_idx]
                        ax_pred_top.imshow(pred_top_heatmap, cmap="viridis")
                        ax_pred_top.set_title(f"Pred Top - Sample {sample_idx}")
                        ax_pred_top.set_xticks([])
                        ax_pred_top.set_yticks([])

                        # Row 5: Combined Top with Gaussians
                        ax_combined_top = axes[5, sample_idx]
                        # Show GT as grayscale background
                        ax_combined_top.imshow(gt_top_heatmap, cmap="gray", alpha=0.6)
                        # Overlay pred as red heatmap
                        if np.max(pred_top_heatmap) > 0:
                            ax_combined_top.imshow(
                                pred_top_heatmap, cmap="Reds", alpha=0.5
                            )

                        # Plot fitted Gaussians
                        gt_top_params = {
                            k: v[sample_idx]
                            for k, v in gt_gaussian_params["top"].items()
                        }
                        pred_top_params = {
                            k: v[sample_idx]
                            for k, v in pred_gaussian_params["top"].items()
                        }

                        plot_gaussian_ellipse(
                            ax_combined_top, gt_top_params, color="cyan", alpha=0.8
                        )
                        plot_gaussian_ellipse(
                            ax_combined_top, pred_top_params, color="yellow", alpha=0.8
                        )

                        # Compute and display EMD
                        top_emd = gaussian_2d_wasserstein(
                            gt_top_params, pred_top_params
                        )
                        ax_combined_top.text(
                            0.02,
                            0.98,
                            f"EMD: {top_emd:.2f}",
                            transform=ax_combined_top.transAxes,
                            verticalalignment="top",
                            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
                            color="white",
                            fontsize=8,
                        )

                        ax_combined_top.set_title(f"Combined Top - Sample {sample_idx}")
                        ax_combined_top.set_xticks([])
                        ax_combined_top.set_yticks([])

                    # Add row labels on the left
                    row_labels = [
                        "GT Bottom",
                        "Pred Bottom",
                        "Combined Bottom",
                        "GT Top",
                        "Pred Top",
                        "Combined Top",
                    ]
                    for i, label in enumerate(row_labels):
                        axes[i, 0].set_ylabel(
                            label, rotation=90, fontsize=12, fontweight="bold"
                        )

                    plt.tight_layout()
                    plt.savefig(
                        f"gaussian_fit_debug_{measurement_type}.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                    plt.close()

                    print(
                        f"Saved Gaussian fitting visualization for {measurement_type} to gaussian_fit_debug_{measurement_type}.png"
                    )

                results[gaussian_emd_key][selection_strategy][x_value].append(
                    float(np.mean(batch_emds))
                )

        # Add other metrics
        for metric_name in keys_to_extract:
            if metric_name == "dice":
                continue  # Already handled above
            if metric_name not in metrics:
                continue
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
