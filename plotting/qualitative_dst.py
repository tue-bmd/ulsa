"""Generate qualitative plots for downstream task results."""

import argparse
import os
import pickle
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import ops

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from zea import init_device

    init_device()


from plot_downstream_task_results import get_config_value

import zea.utils
from benchmark_active_sampling_ultrasound import extract_sweep_data
from ulsa.downstream_task import EchoNetLVHMeasurement
from ulsa.io_utils import (
    plot_downstream_task_output_for_presentation,
    postprocess_agent_results,
    postprocess_heatmap,
)
from zea import Config

# Strategy names for display
STRATEGY_NAMES = {
    "downstream_task_selection": "Measurement Information Gain",
    "greedy_entropy": "Tissue Information Gain",
    "uniform_random": "Uniform Random",
    "equispaced": "Equispaced",
}

# Measurement type names and colors
MEASUREMENT_NAMES = {
    "LVPW": "LVPW",
    "LVID": "LVID",
    "IVS": "IVS",
}

MEASUREMENT_COLORS = {
    "LVPW": "#B8860B",  # Dark goldenrod (dark yellow)
    "LVID": "#8B008B",  # Dark magenta
    "IVS": "#008B8B",  # Dark cyan (dark teal)
}


def extract_configs_only(
    sweep_paths, strategies_to_plot=None, x_value=None, target_measurement_type="LVID"
):
    """Extract only config information to quickly find suitable runs."""
    suitable_runs = []

    for sweep_path in sweep_paths:
        try:
            print(f"Scanning configs in sweep: {sweep_path}")

            sweep_details_path = os.path.join(sweep_path, "sweep_details.yaml")
            if not os.path.exists(sweep_details_path):
                print(f"Missing sweep_details.yaml in {sweep_path}")
                continue

            for run_dir in sorted(os.listdir(sweep_path)):
                run_path = os.path.join(sweep_path, run_dir)
                if not os.path.isdir(run_path):
                    continue

                config_path = os.path.join(run_path, "config.yaml")
                metrics_path = os.path.join(run_path, "metrics.npz")
                filepath_yaml = os.path.join(run_path, "target_filepath.yaml")

                # Check if all required files exist
                if not all(
                    os.path.exists(p)
                    for p in [config_path, metrics_path, filepath_yaml]
                ):
                    continue

                try:
                    config = Config.from_yaml(config_path)
                    target_file = Config.from_yaml(filepath_yaml)["target_filepath"]

                    # Extract basic info
                    x_val = get_config_value(config, "action_selection.n_actions")
                    if x_val is None or (x_value is not None and x_val != x_value):
                        continue

                    selection_strategy = config.get("action_selection", {}).get(
                        "selection_strategy"
                    )
                    if selection_strategy is None:
                        continue

                    if (
                        strategies_to_plot is not None
                        and selection_strategy not in strategies_to_plot
                    ):
                        continue

                    # Extract measurement type for downstream_task_selection
                    measurement_type = None
                    if selection_strategy == "downstream_task_selection":
                        measurement_type = (
                            config.get("action_selection", {})
                            .get("kwargs", {})
                            .get("measurement_type", None)
                        )
                        # Only include if it matches target measurement type
                        if measurement_type != target_measurement_type:
                            continue

                    suitable_runs.append(
                        {
                            "selection_strategy": selection_strategy,
                            "measurement_type": measurement_type,
                            "x_value": x_val,
                            "filepath": target_file,
                            "filename": Path(target_file).stem,
                            "run_path": run_path,
                            "config_path": config_path,
                            "metrics_path": metrics_path,
                        }
                    )

                except Exception as e:
                    print(f"Error processing config {config_path}: {e}")
                    continue

        except Exception as e:
            print(f"Failed to process sweep {sweep_path}: {e}")

    return pd.DataFrame(suitable_runs)


def select_representative_run_fast(
    suitable_runs_df, strategy, target_measurement_type="LVID", seed=42
):
    """Select a random run"""
    strategy_runs = suitable_runs_df[
        suitable_runs_df["selection_strategy"] == strategy
    ].copy()

    if strategy_runs.empty:
        return None

    print(f"Found {len(strategy_runs)} candidate runs for {strategy}")

    # Set seed for reproducible random selection
    np.random.seed(seed)

    # Randomly select one run
    selected_idx = np.random.choice(strategy_runs.index)
    selected_run_info = strategy_runs.loc[selected_idx]

    print(f"Randomly selected run: {selected_run_info['run_path']}")
    return selected_run_info["run_path"]


def select_representative_runs_fast(
    suitable_runs_df, strategy, target_measurement_type="LVID", seed=42, n_runs=2
):
    """Select multiple random runs for comparison"""
    strategy_runs = suitable_runs_df[
        suitable_runs_df["selection_strategy"] == strategy
    ].copy()

    if strategy_runs.empty:
        return []

    print(f"Found {len(strategy_runs)} candidate runs for {strategy}")

    # Set seed for reproducible random selection
    np.random.seed(seed)

    # Randomly select n_runs
    n_available = len(strategy_runs)
    n_to_select = min(n_runs, n_available)

    selected_indices = np.random.choice(
        strategy_runs.index, size=n_to_select, replace=False
    )
    selected_runs = [strategy_runs.loc[idx]["run_path"] for idx in selected_indices]

    print(f"Selected {len(selected_runs)} runs for {strategy}")
    for i, run_path in enumerate(selected_runs):
        print(f"  Run {i + 1}: {run_path}")

    return selected_runs


def load_run_data_from_metrics(run_path):
    """Load all necessary data from metrics.npz for visualization."""
    metrics_path = os.path.join(run_path, "metrics.npz")
    config_path = os.path.join(run_path, "config.yaml")
    target_file_path = os.path.join(run_path, "target_filepath.yaml")

    if not all(os.path.exists(p) for p in [metrics_path, config_path]):
        return None

    config = Config.from_yaml(config_path)
    metrics = np.load(metrics_path, allow_pickle=True)
    target_file_hash = Path(Config.from_yaml(target_file_path)["target_filepath"]).stem

    # Extract data from metrics.npz
    data = {
        "config": config,
        "targets": metrics.get("targets"),
        "target_file_hash": target_file_hash,
        "reconstructions": metrics.get("reconstructions"),
        "measurements": metrics.get("measurements"),
        "masks": metrics.get("masks"),
        "belief_distributions": metrics.get("belief_distributions"),
        "segmentation_mask_targets": metrics.get("segmentation_mask_targets"),
        "segmentation_mask_reconstructions": metrics.get(
            "segmentation_mask_reconstructions"
        ),
        "segmentation_mask_beliefs": metrics.get("segmentation_mask_beliefs"),
    }

    return data


def create_downstream_task_visualization_from_metrics(
    run_path, save_dir, downstream_task, agent_config=None, target_file_hash=None
):
    """Create downstream task visualization using data from metrics.npz."""
    # Load data from metrics
    data = load_run_data_from_metrics(run_path)
    if data is None:
        print(f"Failed to load data from {run_path}")
        return None

    # Extract relevant arrays
    targets = data["targets"]
    reconstructions = data["reconstructions"]
    measurements = data["measurements"]
    masks = data["masks"]
    belief_distributions = data["belief_distributions"]

    # Downstream task outputs
    targets_dst = data["segmentation_mask_targets"]
    reconstructions_dst = data["segmentation_mask_reconstructions"]
    beliefs_dst = data["segmentation_mask_beliefs"]

    if targets_dst is None or reconstructions_dst is None or beliefs_dst is None:
        print(f"Missing downstream task outputs in {run_path}")
        return None

    # Convert to proper format and squeeze where needed
    targets = np.squeeze(targets, axis=-1) if targets.ndim > 3 else targets
    measurements = (
        np.squeeze(measurements, axis=-1) if measurements.ndim > 3 else measurements
    )
    reconstructions = (
        np.squeeze(reconstructions, axis=-1)
        if reconstructions.ndim > 3
        else reconstructions
    )
    masks = np.squeeze(masks, axis=-1) if masks.ndim > 3 else masks

    # For belief distributions, we need the posterior mean as reconstructions
    # and posterior std for uncertainty visualization
    if belief_distributions.ndim == 5:  # (frames, particles, h, w, c)
        posterior_mean = np.mean(belief_distributions, axis=1)  # Average over particles
        posterior_std = np.std(belief_distributions, axis=1)  # Std over particles
        reconstructions = (
            np.squeeze(posterior_mean, axis=-1)
            if posterior_mean.ndim > 3
            else posterior_mean
        )
        posterior_std = (
            np.squeeze(posterior_std, axis=-1)
            if posterior_std.ndim > 3
            else posterior_std
        )
    else:
        posterior_std = np.zeros_like(reconstructions)

    # Squeeze downstream task outputs
    targets_dst = np.squeeze(targets_dst)
    reconstructions_dst = np.squeeze(reconstructions_dst)
    beliefs_dst = np.squeeze(beliefs_dst)

    # Create dummy saliency maps (zeros) since they're not in metrics.npz
    saliency_maps = np.zeros_like(targets)

    # Use config from data if agent_config not provided
    if agent_config is None:
        agent_config = data["config"]

    # Normalize targets to expected range for downstream task input
    # Assuming targets are in uint8 range [0, 255], convert to [-1, 1]
    targets_normalized = zea.utils.translate(
        targets, range_from=(0, 255), range_to=(-1, 1)
    )
    reconstructions_normalized = zea.utils.translate(
        reconstructions, range_from=(0, 255), range_to=(-1, 1)
    )
    measurements_normalized = zea.utils.translate(
        measurements, range_from=(0, 255), range_to=(-1, 1)
    )

    # Call the visualization function
    plot_downstream_task_output_for_presentation(
        save_dir,
        targets_normalized,  # shape (num_frames, H, W)
        measurements_normalized,  # shape (num_frames, H, W)
        reconstructions_normalized,  # shape (num_frames, H, W)
        masks,
        posterior_std,  # shape (num_frames, H, W)
        downstream_task,
        reconstructions_dst,  # segmentation masks
        beliefs_dst,  # shape (frames, particles, h, w, c)
        targets_dst,  # segmentation masks
        saliency_maps,  # shape (num_frames, H, W)
        agent_config.io_config,
        dpi=150,
        scan_convert_order=0,
        scan_convert_resolution=1,
        interpolation_matplotlib="nearest",
        image_range=(-1, 1),
        # context="styles/ieee-tmi.mplstyle",
        gif_name="downstream_task_output_from_metrics.gif",
        drop_first_n_frames=0,
        no_measurement_color="gray",
        show_reconstructions_in_timeseries=True,
        target_file_hash=data["target_file_hash"],
    )

    print(f"Created downstream task visualization in {save_dir}")
    return save_dir


def process_run_for_visualization(run_path, io_config):
    """Process a single run's data for visualization."""
    # Load run data
    data = load_single_run_data(run_path)
    if data is None:
        return None

    # Initialize downstream task
    downstream_task = EchoNetLVHMeasurement()

    # Get targets and reconstructions
    targets = np.squeeze(data["targets"])
    reconstructions = np.squeeze(
        np.mean(data["belief_distributions"], axis=1)
    )  # Use posterior mean
    measurements = np.squeeze(data["measurements"])
    masks = np.squeeze(data["masks"])

    # Process segmentation outputs
    targets_dst = np.squeeze(data["segmentation_mask_targets"])
    reconstructions_dst = np.squeeze(data["segmentation_mask_reconstructions"])
    beliefs_dst = np.squeeze(data["segmentation_mask_beliefs"])

    if targets_dst is None or reconstructions_dst is None or beliefs_dst is None:
        print("Missing downstream task outputs")
        return None

    # Convert to tensor format for downstream task
    targets_dst = ops.convert_to_tensor(targets_dst)
    reconstructions_dst = ops.convert_to_tensor(reconstructions_dst)
    beliefs_dst = ops.convert_to_tensor(beliefs_dst)

    # Compute measurement time series
    target_line_lengths, recon_line_lengths, line_stds = (
        compute_measurement_time_series(
            data["target_file_hash"],
            targets_dst,
            reconstructions_dst,
            beliefs_dst,
            downstream_task,
        )
    )

    # Select middle frame for visualization
    num_frames = targets.shape[0]
    frame_idx = num_frames // 2

    # Extract single frame while preserving batch dimension
    targets_single = targets[frame_idx : frame_idx + 1]  # Shape: (1, ...)
    reconstructions_single = reconstructions[
        frame_idx : frame_idx + 1
    ]  # Shape: (1, ...)
    measurements_single = measurements[frame_idx : frame_idx + 1]  # Shape: (1, ...)
    masks_single = masks[frame_idx : frame_idx + 1]

    # Apply postprocessing
    targets_with_mask = downstream_task.postprocess_for_visualization(
        targets_single, targets_dst[frame_idx : frame_idx + 1]
    )
    reconstructions_with_mask = downstream_task.postprocess_for_visualization(
        reconstructions_single, reconstructions_dst[frame_idx : frame_idx + 1]
    )

    return {
        "target_line_lengths": target_line_lengths,
        "recon_line_lengths": recon_line_lengths,
        "line_stds": line_stds,
        "targets_with_mask": targets_with_mask,
        "measurements": measurements_single,
        "reconstructions_with_mask": reconstructions_with_mask,
        "frame_idx": frame_idx,
    }


def load_single_run_data(run_path):
    """Load detailed data from a single run for qualitative visualization."""
    config_path = os.path.join(run_path, "config.yaml")
    metrics_path = os.path.join(run_path, "metrics.npz")
    target_file_path = os.path.join(run_path, "target_filepath.yaml")

    if not all(os.path.exists(p) for p in [config_path, metrics_path]):
        return None

    config = Config.from_yaml(config_path)
    target_file_hash = Path(Config.from_yaml(target_file_path)["target_filepath"]).stem
    metrics = np.load(metrics_path, allow_pickle=True)

    # Extract relevant data
    data = {
        "config": config,
        "target_file_hash": target_file_hash,
        "targets": metrics.get("targets"),
        "reconstructions": metrics.get("reconstructions"),
        "measurements": metrics.get("measurements"),
        "masks": metrics.get("masks"),
        "belief_distributions": metrics.get("belief_distributions"),
        "segmentation_mask_targets": metrics.get("segmentation_mask_targets"),
        "segmentation_mask_reconstructions": metrics.get(
            "segmentation_mask_reconstructions"
        ),
        "segmentation_mask_beliefs": metrics.get("segmentation_mask_beliefs"),
    }

    return data


def compute_line_length(target_file_hash, dst_model, bottom_coords, top_coords):
    """Compute Euclidean distance between bottom and top coordinates."""
    if bottom_coords is None or top_coords is None:
        return np.nan
    return dst_model.get_distance_in_cm(target_file_hash, bottom_coords, top_coords)


def compute_measurement_time_series(
    target_file_hash, targets_dst, reconstructions_dst, beliefs_dst, downstream_task
):
    """Compute measurement line lengths over time for all measurement types."""
    measurement_types = ["LVPW", "LVID", "IVS"]

    target_line_lengths = {}
    recon_line_lengths = {}
    line_stds = {}

    for measurement_type in measurement_types:

        def compute_line_length_single(dst_output):
            bottom, top = downstream_task.outputs_to_coordinates(
                dst_output[None, ...], measurement_type
            )
            return compute_line_length(target_file_hash, downstream_task, bottom, top)

        target_line_lengths[measurement_type] = ops.vectorized_map(
            compute_line_length_single,
            targets_dst,
        )
        recon_line_lengths[measurement_type] = ops.vectorized_map(
            compute_line_length_single,
            reconstructions_dst,
        )
        line_stds[measurement_type] = ops.vectorized_map(
            lambda belief_heatmaps: ops.std(
                ops.vectorized_map(compute_line_length_single, belief_heatmaps), axis=0
            ),
            beliefs_dst,
        )

    return target_line_lengths, recon_line_lengths, line_stds


def create_time_series_plot(
    target_line_lengths,
    recon_line_lengths,
    save_path,
    strategy_name,
    line_stds=None,
    context="styles/ieee-tmi.mplstyle",
    measurement_types_to_plot=None,
):
    """Create standalone time series plot with optional uncertainty bands."""
    if measurement_types_to_plot is None:
        measurement_types_to_plot = ["LVPW", "LVID", "IVS"]

    num_measurements = len(measurement_types_to_plot)
    num_frames = len(target_line_lengths[measurement_types_to_plot[0]])
    frame_indices = np.arange(num_frames)

    with plt.style.context(context):
        fig, axes = plt.subplots(
            num_measurements, 1, figsize=(8, 2 * num_measurements), sharex=True
        )

        # Handle single subplot case
        if num_measurements == 1:
            axes = [axes]

        for idx, measurement_type in enumerate(measurement_types_to_plot):
            ax = axes[idx]
            color = MEASUREMENT_COLORS[measurement_type]

            # Plot target line lengths
            target_lengths = np.array(target_line_lengths[measurement_type])
            valid_target = ~np.isnan(target_lengths)
            ax.plot(
                frame_indices[valid_target],
                target_lengths[valid_target],
                color=color,
                linestyle="-",
                linewidth=2,
                label="Target",
                alpha=0.9,
            )

            # Plot reconstruction line lengths
            recon_lengths = np.array(recon_line_lengths[measurement_type])
            valid_recon = ~np.isnan(recon_lengths)
            ax.plot(
                frame_indices[valid_recon],
                recon_lengths[valid_recon],
                color=color,
                linestyle="--",
                linewidth=2,
                label="Reconstruction",
                alpha=0.9,
            )

            # Add uncertainty bands if standard deviations are provided
            if line_stds is not None and measurement_type in line_stds:
                std_values = np.array(line_stds[measurement_type])
                valid_std = ~np.isnan(std_values) & ~np.isnan(recon_lengths)

                if np.any(valid_std):
                    # Plot ±1 standard deviation band
                    ax.fill_between(
                        frame_indices[valid_std],
                        recon_lengths[valid_std] - std_values[valid_std],
                        recon_lengths[valid_std] + std_values[valid_std],
                        color=color,
                        alpha=0.2,
                        label="±1σ",
                    )

            # Set labels and formatting
            ax.set_ylabel(f"{measurement_type} [cm]", fontsize=12)
            ax.legend(loc="right", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, num_frames - 1)

            # Only show x-axis label on bottom subplot
            if idx == len(measurement_types_to_plot) - 1:
                ax.set_xlabel("Frame", fontsize=12)

            # Set y-limits based on this measurement type's data including uncertainty
            all_lengths = []
            all_lengths.extend([l for l in target_lengths if not np.isnan(l)])
            all_lengths.extend([l for l in recon_lengths if not np.isnan(l)])

            # Include uncertainty bounds in y-limit calculation
            if line_stds is not None and measurement_type in line_stds:
                std_values = np.array(line_stds[measurement_type])
                valid_std = ~np.isnan(std_values) & ~np.isnan(recon_lengths)
                if np.any(valid_std):
                    all_lengths.extend(
                        recon_lengths[valid_std] - 2 * std_values[valid_std]
                    )
                    all_lengths.extend(
                        recon_lengths[valid_std] + 2 * std_values[valid_std]
                    )

            if all_lengths:
                y_min, y_max = min(all_lengths), max(all_lengths)
                y_range = y_max - y_min
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for title
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved time series plot to {save_path}")


def create_combined_time_series_plot(
    strategy_data_dict,
    save_path,
    context="styles/ieee-tmi.mplstyle",
    measurement_types_to_plot=None,
):
    """Create side-by-side time series plots for strategy comparison."""
    if measurement_types_to_plot is None:
        measurement_types_to_plot = ["LVPW", "LVID", "IVS"]

    # Get data for both strategies
    strategies = list(strategy_data_dict.keys())
    if len(strategies) != 2:
        raise ValueError("Expected exactly 2 strategies for comparison")

    # Reorder strategies: greedy_entropy (left), downstream_task_selection (right)
    if "greedy_entropy" in strategies and "downstream_task_selection" in strategies:
        strategy1, strategy2 = "greedy_entropy", "downstream_task_selection"
    else:
        # Fallback to original order if the expected strategies aren't present
        strategy1, strategy2 = strategies

    data1 = strategy_data_dict[strategy1]
    data2 = strategy_data_dict[strategy2]

    # Get strategy display names
    strategy1_name = STRATEGY_NAMES.get(strategy1, strategy1)
    strategy2_name = STRATEGY_NAMES.get(strategy2, strategy2)

    num_measurements = len(measurement_types_to_plot)
    num_frames = len(data1["target_line_lengths"][measurement_types_to_plot[0]])
    frame_indices = np.arange(num_frames)

    with plt.style.context(context):
        # Adjust figure size based on number of measurements
        # For single measurement, use smaller height
        height = 1.2 * num_measurements
        fig, axes = plt.subplots(
            num_measurements, 2, figsize=(7.16, height), sharex=True
        )

        # Handle single row case
        if num_measurements == 1:
            axes = axes.reshape(1, -1)

        # Store legend information for each row
        row_legend_data = []

        for row_idx, measurement_type in enumerate(measurement_types_to_plot):
            color = MEASUREMENT_COLORS[measurement_type]

            # Get data for both strategies to calculate shared y-limits and MAE
            target_lengths1 = np.array(data1["target_line_lengths"][measurement_type])
            recon_lengths1 = np.array(data1["recon_line_lengths"][measurement_type])
            target_lengths2 = np.array(data2["target_line_lengths"][measurement_type])
            recon_lengths2 = np.array(data2["recon_line_lengths"][measurement_type])

            # Calculate MAE for both strategies
            def compute_mae(target, recon):
                valid_mask = ~(np.isnan(target) | np.isnan(recon))
                if np.any(valid_mask):
                    return np.mean(np.abs(target[valid_mask] - recon[valid_mask]))
                else:
                    return np.nan

            mae1 = compute_mae(target_lengths1, recon_lengths1)
            mae2 = compute_mae(target_lengths2, recon_lengths2)

            # Calculate shared y-limits for this measurement type
            all_lengths_for_ylim = []
            all_lengths_for_ylim.extend([l for l in target_lengths1 if not np.isnan(l)])
            all_lengths_for_ylim.extend([l for l in recon_lengths1 if not np.isnan(l)])
            all_lengths_for_ylim.extend([l for l in target_lengths2 if not np.isnan(l)])
            all_lengths_for_ylim.extend([l for l in recon_lengths2 if not np.isnan(l)])

            # Include uncertainty bounds in y-limit calculation
            if "line_stds" in data1 and data1["line_stds"] is not None:
                std_values1 = np.array(data1["line_stds"][measurement_type])
                valid_std1 = ~np.isnan(std_values1) & ~np.isnan(recon_lengths1)
                if np.any(valid_std1):
                    all_lengths_for_ylim.extend(
                        recon_lengths1[valid_std1] - std_values1[valid_std1]
                    )
                    all_lengths_for_ylim.extend(
                        recon_lengths1[valid_std1] + std_values1[valid_std1]
                    )

            if "line_stds" in data2 and data2["line_stds"] is not None:
                std_values2 = np.array(data2["line_stds"][measurement_type])
                valid_std2 = ~np.isnan(std_values2) & ~np.isnan(recon_lengths2)
                if np.any(valid_std2):
                    all_lengths_for_ylim.extend(
                        recon_lengths2[valid_std2] - std_values2[valid_std2]
                    )
                    all_lengths_for_ylim.extend(
                        recon_lengths2[valid_std2] + std_values2[valid_std2]
                    )

            # Calculate shared y-limits
            if all_lengths_for_ylim:
                y_min, y_max = min(all_lengths_for_ylim), max(all_lengths_for_ylim)
                y_range = y_max - y_min
                shared_ylim = (y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            else:
                shared_ylim = (0, 1)

            # Plot for strategy 1 (left column) - now greedy_entropy
            ax1 = axes[row_idx, 0]

            # Target and reconstruction for strategy 1
            valid_target1 = ~np.isnan(target_lengths1)
            valid_recon1 = ~np.isnan(recon_lengths1)

            line1 = ax1.plot(
                frame_indices[valid_target1],
                target_lengths1[valid_target1],
                color=color,
                linestyle="-",
                linewidth=1.5,
                alpha=0.9,
                marker=None,
                markersize=0,
            )[0]

            line2 = ax1.plot(
                frame_indices[valid_recon1],
                recon_lengths1[valid_recon1],
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                marker=None,
                markersize=0,
            )[0]

            # Add uncertainty bands for strategy 1
            fill_patch = None
            if "line_stds" in data1 and data1["line_stds"] is not None:
                std_values1 = np.array(data1["line_stds"][measurement_type])
                valid_std1 = ~np.isnan(std_values1) & ~np.isnan(recon_lengths1)

                if np.any(valid_std1):
                    fill_patch = ax1.fill_between(
                        frame_indices[valid_std1],
                        recon_lengths1[valid_std1] - std_values1[valid_std1],
                        recon_lengths1[valid_std1] + std_values1[valid_std1],
                        color=color,
                        alpha=0.2,
                    )

            # Add MAE text for strategy 1
            if not np.isnan(mae1):
                ax1.text(
                    0.04,
                    0.95,
                    f"MAE: {mae1:.3f}",
                    transform=ax1.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                )

            # Plot for strategy 2 (right column) - now downstream_task_selection
            ax2 = axes[row_idx, 1]

            # Target and reconstruction for strategy 2
            valid_target2 = ~np.isnan(target_lengths2)
            valid_recon2 = ~np.isnan(recon_lengths2)

            ax2.plot(
                frame_indices[valid_target2],
                target_lengths2[valid_target2],
                color=color,
                linestyle="-",
                linewidth=1.5,
                alpha=0.9,
                marker=None,
                markersize=0,
            )
            ax2.plot(
                frame_indices[valid_recon2],
                recon_lengths2[valid_recon2],
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                marker=None,
                markersize=0,
            )

            # Add uncertainty bands for strategy 2
            if "line_stds" in data2 and data2["line_stds"] is not None:
                std_values2 = np.array(data2["line_stds"][measurement_type])
                valid_std2 = ~np.isnan(std_values2) & ~np.isnan(recon_lengths2)

                if np.any(valid_std2):
                    ax2.fill_between(
                        frame_indices[valid_std2],
                        recon_lengths2[valid_std2] - std_values2[valid_std2],
                        recon_lengths2[valid_std2] + std_values2[valid_std2],
                        color=color,
                        alpha=0.2,
                    )

            # Add MAE text for strategy 2
            if not np.isnan(mae2):
                ax2.text(
                    0.04,
                    0.95,
                    f"MAE: {mae2:.2f}",
                    transform=ax2.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                )

            # Configure both axes with shared y-limits
            for ax_idx, ax in enumerate([ax1, ax2]):
                ax.set_xlim(0, num_frames - 1)
                ax.set_ylim(shared_ylim)
                ax.grid(True, alpha=0.3)

                # Y-axis label only on left column
                if ax_idx == 0:
                    ax.set_ylabel(f"{measurement_type} [cm]", fontsize=9)

                # X-axis label only on bottom row
                if row_idx == len(measurement_types_to_plot) - 1:
                    ax.set_xlabel("Frame", fontsize=9)

                # Format y-axis to show max 1 decimal place
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))

            # Store legend data for this row
            legend_handles = [line1, line2]
            legend_labels = ["Target", "Reconstruction"]
            if fill_patch is not None:
                legend_handles.append(fill_patch)
                legend_labels.append("±1σ")

            row_legend_data.append((legend_handles, legend_labels))

        # Add column titles
        axes[0, 0].set_title(strategy1_name, fontsize=9, pad=10)
        axes[0, 1].set_title(strategy2_name, fontsize=9, pad=10)

        # Adjust layout parameters based on number of measurements
        if num_measurements == 1:
            plt.tight_layout()
            plt.subplots_adjust(top=0.85, wspace=0.15, right=0.75)
        else:
            plt.tight_layout()
            plt.subplots_adjust(top=0.90, wspace=0.15, hspace=0.35, right=0.8)

        # Now add legends after layout is finalized
        for row_idx, (legend_handles, legend_labels) in enumerate(row_legend_data):
            # Get the actual position of the row after layout adjustment
            ax_left = axes[row_idx, 0]
            ax_right = axes[row_idx, 1]

            # Calculate the center y-position of this row in figure coordinates
            pos_left = ax_left.get_position()
            pos_right = ax_right.get_position()

            # Use the average of both subplot centers for the row center
            row_center_y = (pos_left.y0 + pos_left.y1 + pos_right.y0 + pos_right.y1) / 4

            # Adjust bbox_to_anchor based on number of measurements
            bbox_x = 0.77 if num_measurements == 1 else 0.82

            fig.legend(
                legend_handles,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(bbox_x, row_center_y),
                fontsize=8,
                framealpha=0.9,
            )

        # Save with high DPI for publication quality
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved combined time series comparison to {save_path}")


def create_combined_time_series_plot_multiple_sequences(
    strategy_data_dict,
    save_path,
    context="styles/ieee-tmi.mplstyle",
    measurement_types_to_plot=None,
):
    """Create side-by-side time series plots for strategy comparison with multiple sequences."""
    if measurement_types_to_plot is None:
        measurement_types_to_plot = ["LVPW", "LVID", "IVS"]

    # Get data for both strategies
    strategies = list(strategy_data_dict.keys())
    if len(strategies) != 2:
        raise ValueError("Expected exactly 2 strategies for comparison")

    # Reorder strategies: greedy_entropy (left), downstream_task_selection (right)
    if "greedy_entropy" in strategies and "downstream_task_selection" in strategies:
        strategy1, strategy2 = "greedy_entropy", "downstream_task_selection"
    else:
        # Fallback to original order if the expected strategies aren't present
        strategy1, strategy2 = strategies

    # Get data for both strategies - now expecting lists of runs
    data1_list = strategy_data_dict[strategy1]  # List of viz_data dicts
    data2_list = strategy_data_dict[strategy2]  # List of viz_data dicts

    # Get strategy display names
    strategy1_name = STRATEGY_NAMES.get(strategy1, strategy1)
    strategy2_name = STRATEGY_NAMES.get(strategy2, strategy2)

    # Determine number of sequences to plot (minimum of available runs)
    n_sequences = min(len(data1_list), len(data2_list))
    if n_sequences == 0:
        print("No data available for plotting")
        return

    # For now, plot only the first measurement type if multiple are specified
    if len(measurement_types_to_plot) > 1:
        print(
            f"Multiple measurement types specified, plotting only {measurement_types_to_plot[0]}"
        )
        measurement_type = measurement_types_to_plot[0]
    else:
        measurement_type = measurement_types_to_plot[0]

    with plt.style.context(context):
        # Adjust figure size based on number of sequences
        height = 1 * n_sequences
        # Remove sharex=True to allow independent x-axes for each row
        fig, axes = plt.subplots(n_sequences, 2, figsize=(7.16, height))

        # Handle single row case
        if n_sequences == 1:
            axes = axes.reshape(1, -1)

        # Store legend information for each row
        row_legend_data = []

        color = MEASUREMENT_COLORS[measurement_type]

        for row_idx in range(n_sequences):
            data1 = data1_list[row_idx]
            data2 = data2_list[row_idx]

            # Get frame indices for both sequences (can be different lengths)
            num_frames1 = len(data1["target_line_lengths"][measurement_type])
            num_frames2 = len(data2["target_line_lengths"][measurement_type])
            frame_indices1 = np.arange(num_frames1)
            frame_indices2 = np.arange(num_frames2)

            # Get data for both strategies to calculate shared y-limits and MAE
            target_lengths1 = np.array(data1["target_line_lengths"][measurement_type])
            recon_lengths1 = np.array(data1["recon_line_lengths"][measurement_type])
            target_lengths2 = np.array(data2["target_line_lengths"][measurement_type])
            recon_lengths2 = np.array(data2["recon_line_lengths"][measurement_type])

            # Calculate MAE for both strategies
            def compute_mae(target, recon):
                valid_mask = ~(np.isnan(target) | np.isnan(recon))
                if np.any(valid_mask):
                    return np.mean(np.abs(target[valid_mask] - recon[valid_mask]))
                else:
                    return np.nan

            mae1 = compute_mae(target_lengths1, recon_lengths1)
            mae2 = compute_mae(target_lengths2, recon_lengths2)

            # Calculate shared y-limits for this measurement type across both sequences
            all_lengths_for_ylim = []
            all_lengths_for_ylim.extend([l for l in target_lengths1 if not np.isnan(l)])
            all_lengths_for_ylim.extend([l for l in recon_lengths1 if not np.isnan(l)])
            all_lengths_for_ylim.extend([l for l in target_lengths2 if not np.isnan(l)])
            all_lengths_for_ylim.extend([l for l in recon_lengths2 if not np.isnan(l)])

            # Include uncertainty bounds in y-limit calculation
            if "line_stds" in data1 and data1["line_stds"] is not None:
                std_values1 = np.array(data1["line_stds"][measurement_type])
                valid_std1 = ~np.isnan(std_values1) & ~np.isnan(recon_lengths1)
                if np.any(valid_std1):
                    all_lengths_for_ylim.extend(
                        recon_lengths1[valid_std1] - std_values1[valid_std1]
                    )
                    all_lengths_for_ylim.extend(
                        recon_lengths1[valid_std1] + std_values1[valid_std1]
                    )

            if "line_stds" in data2 and data2["line_stds"] is not None:
                std_values2 = np.array(data2["line_stds"][measurement_type])
                valid_std2 = ~np.isnan(std_values2) & ~np.isnan(recon_lengths2)
                if np.any(valid_std2):
                    all_lengths_for_ylim.extend(
                        recon_lengths2[valid_std2] - std_values2[valid_std2]
                    )
                    all_lengths_for_ylim.extend(
                        recon_lengths2[valid_std2] + std_values2[valid_std2]
                    )

            # Calculate shared y-limits
            if all_lengths_for_ylim:
                y_min, y_max = min(all_lengths_for_ylim), max(all_lengths_for_ylim)
                y_range = y_max - y_min
                shared_ylim = (y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            else:
                shared_ylim = (0, 1)

            # Plot for strategy 1 (left column) - greedy_entropy
            ax1 = axes[row_idx, 0]

            # Target and reconstruction for strategy 1
            valid_target1 = ~np.isnan(target_lengths1)
            valid_recon1 = ~np.isnan(recon_lengths1)

            line1 = ax1.plot(
                frame_indices1[valid_target1],
                target_lengths1[valid_target1],
                color=color,
                linestyle="-",
                linewidth=1.5,
                alpha=0.9,
                marker=None,
                markersize=0,
            )[0]

            line2 = ax1.plot(
                frame_indices1[valid_recon1],
                recon_lengths1[valid_recon1],
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                marker=None,
                markersize=0,
            )[0]

            # Add uncertainty bands for strategy 1
            fill_patch = None
            if "line_stds" in data1 and data1["line_stds"] is not None:
                std_values1 = np.array(data1["line_stds"][measurement_type])
                valid_std1 = ~np.isnan(std_values1) & ~np.isnan(recon_lengths1)

                if np.any(valid_std1):
                    fill_patch = ax1.fill_between(
                        frame_indices1[valid_std1],
                        recon_lengths1[valid_std1] - std_values1[valid_std1],
                        recon_lengths1[valid_std1] + std_values1[valid_std1],
                        color=color,
                        alpha=0.2,
                    )

            # Add MAE text for strategy 1
            if not np.isnan(mae1):
                ax1.text(
                    0.04,
                    0.95,
                    f"MAE: {mae1:.3f}",
                    transform=ax1.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                )

            # Plot for strategy 2 (right column) - downstream_task_selection
            ax2 = axes[row_idx, 1]

            # Target and reconstruction for strategy 2
            valid_target2 = ~np.isnan(target_lengths2)
            valid_recon2 = ~np.isnan(recon_lengths2)

            ax2.plot(
                frame_indices2[valid_target2],
                target_lengths2[valid_target2],
                color=color,
                linestyle="-",
                linewidth=1.5,
                alpha=0.9,
                marker=None,
                markersize=0,
            )
            ax2.plot(
                frame_indices2[valid_recon2],
                recon_lengths2[valid_recon2],
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                marker=None,
                markersize=0,
            )

            # Add uncertainty bands for strategy 2
            if "line_stds" in data2 and data2["line_stds"] is not None:
                std_values2 = np.array(data2["line_stds"][measurement_type])
                valid_std2 = ~np.isnan(std_values2) & ~np.isnan(recon_lengths2)

                if np.any(valid_std2):
                    ax2.fill_between(
                        frame_indices2[valid_std2],
                        recon_lengths2[valid_std2] - std_values2[valid_std2],
                        recon_lengths2[valid_std2] + std_values2[valid_std2],
                        color=color,
                        alpha=0.2,
                    )

            # Add MAE text for strategy 2
            if not np.isnan(mae2):
                ax2.text(
                    0.04,
                    0.95,
                    f"MAE: {mae2:.2f}",
                    transform=ax2.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                )

            # Configure both axes with their respective x-limits and shared y-limits
            # Set x-limits based on the respective sequence length
            ax1.set_xlim(0, num_frames1 - 1)
            ax2.set_xlim(0, num_frames2 - 1)

            # Both axes share the same y-limits for comparison
            ax1.set_ylim(shared_ylim)
            ax2.set_ylim(shared_ylim)

            # Grid and formatting
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)

            # Y-axis label only on left column
            ax1.set_ylabel(f"{measurement_type} [cm]", fontsize=9)

            # X-axis label on all axes since they have independent scales
            ax1.set_xlabel("Frame", fontsize=10)
            ax2.set_xlabel("Frame", fontsize=10)

            # Format y-axis to show max 1 decimal place
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))

            # Store legend data for this row (only need to do this once)
            if row_idx == 0:
                legend_handles = [line1, line2]
                legend_labels = ["Target", "Reconstruction"]
                if fill_patch is not None:
                    legend_handles.append(fill_patch)
                    legend_labels.append("±1σ")
                row_legend_data.append((legend_handles, legend_labels))

        # Add column titles
        axes[0, 0].set_title(strategy1_name, fontsize=10, pad=10)
        axes[0, 1].set_title(strategy2_name, fontsize=10, pad=10)

        # Adjust layout parameters based on number of sequences
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, wspace=0.15, hspace=0.4, right=0.8, left=0.1)

        # Add legend if we have legend data
        if row_legend_data:
            legend_handles, legend_labels = row_legend_data[0]

            # Position legend in the center right
            fig.legend(
                legend_handles,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(0.82, 0.5),
                fontsize=10,
                framealpha=0.9,
            )

        # Save with high DPI for publication quality
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(
            f"Saved combined time series comparison (multiple sequences) to {save_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-root", type=str, default="./qualitative_dst_results")
    parser.add_argument(
        "--x-value", type=int, default=5, help="Number of scan lines to analyze"
    )
    parser.add_argument(
        "--target-measurement-type",
        type=str,
        default="LVID",
        choices=["LVPW", "LVID", "IVS"],
        help="Measurement type that downstream_task_selection was optimized for",
    )
    parser.add_argument(
        "--measurement-to-plot",
        type=str,
        default="LVID",
        choices=["LVPW", "LVID", "IVS", "all"],
        help="Which measurement type to plot (default: LVID, 'all' plots all three)",
    )
    parser.add_argument(
        "--n-sequences",
        type=int,
        default=2,
        help="Number of different sequences to plot for comparison (default: 2)",
    )
    args = parser.parse_args()

    # Create save directory
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    # Define sweep paths - include the path with measurement-specific sweeps
    SWEEP_PATHS = [
        "/mnt/z/Ultrasound-BMd/data/oisin/ULSA_out_dst/echonetlvh_downstream_task/27_08_25_run1/sweep_2025_08_27_234604_304986",
        "/mnt/z/Ultrasound-BMd/data/oisin/ULSA_out_dst/echonetlvh_downstream_task/27_08_25_run1/sweep_2025_08_28_005533_658322",
        "/mnt/z/Ultrasound-BMd/data/oisin/ULSA_out_dst/echonetlvh_downstream_task/27_08_25_run1/sweep_2025_08_28_020347_562409",
        "/mnt/z/Ultrasound-BMd/data/oisin/ULSA_out_dst/echonetlvh_downstream_task/27_08_25_run1/sweep_2025_08_28_085132_599359",
    ]

    # Strategies to compare
    strategies_to_compare = ["downstream_task_selection", "greedy_entropy"]

    # Fast config-only scan to find suitable runs
    print("Scanning configs to find suitable runs...")
    suitable_runs_df = extract_configs_only(
        SWEEP_PATHS,
        strategies_to_plot=strategies_to_compare,
        x_value=args.x_value,
        target_measurement_type=args.target_measurement_type,
    )

    if suitable_runs_df.empty:
        print("No suitable runs found for the specified parameters")
        exit(1)

    print(f"Found {len(suitable_runs_df)} suitable runs")
    print(f"Strategies: {suitable_runs_df['selection_strategy'].unique()}")
    print(f"Measurement types: {suitable_runs_df['measurement_type'].unique()}")

    # IO configuration for processing
    io_config = Config(
        {"scan_convert": True, "scan_conversion_angles": [-45, 45], "gif_fps": 10}
    )

    # Determine which measurements to plot
    if args.measurement_to_plot == "all":
        measurement_types_to_plot = ["LVPW", "LVID", "IVS"]
    else:
        measurement_types_to_plot = [args.measurement_to_plot]

    # Process each strategy - now selecting multiple runs
    strategy_data_dict = {}
    for strategy in strategies_to_compare:
        print(f"\nProcessing strategy: {strategy}")

        # Select multiple representative runs
        run_paths = select_representative_runs_fast(
            suitable_runs_df,
            strategy,
            target_measurement_type=args.target_measurement_type,
            seed=3,
            n_runs=args.n_sequences,
        )

        if not run_paths:
            print(f"No suitable runs found for strategy {strategy}")
            continue

        # Process each run
        strategy_viz_data_list = []
        for i, run_path in enumerate(run_paths):
            print(f"Processing run {i + 1}/{len(run_paths)} for {strategy}...")
            viz_data = process_run_for_visualization(run_path, io_config)

            if viz_data is None:
                print(f"Failed to process run data for {run_path}")
                continue

            strategy_viz_data_list.append(viz_data)

        if not strategy_viz_data_list:
            print(f"No valid visualization data for strategy {strategy}")
            continue

        strategy_data_dict[strategy] = strategy_viz_data_list

        # Get strategy display name
        strategy_display_name = STRATEGY_NAMES.get(strategy, strategy)

        # Create individual time series plots for each sequence
        for i, viz_data in enumerate(strategy_viz_data_list):
            time_series_path = (
                save_root
                / f"time_series_{strategy}_seq{i + 1}_x{args.x_value}_{args.measurement_to_plot}.pdf"
            )
            create_time_series_plot(
                viz_data["target_line_lengths"],
                viz_data["recon_line_lengths"],
                time_series_path,
                f"{strategy_display_name} (Seq. {i + 1})",
                line_stds=viz_data["line_stds"],
                context="styles/ieee-tmi.mplstyle",
                measurement_types_to_plot=measurement_types_to_plot,
            )

    # Create combined time series plot for strategy comparison across multiple sequences
    if len(strategy_data_dict) == 2:
        combined_time_series_path = (
            save_root
            / f"combined_time_series_multisequence_x{args.x_value}_{args.measurement_to_plot}.pdf"
        )
        create_combined_time_series_plot_multiple_sequences(
            strategy_data_dict,
            combined_time_series_path,
            context="styles/ieee-tmi.mplstyle",
            measurement_types_to_plot=measurement_types_to_plot,
        )

    print(f"\nQualitative plots saved to {save_root}")
    print(f"Compared strategies: {[STRATEGY_NAMES[s] for s in strategies_to_compare]}")
    print(f"X-value: {args.x_value}")
    print(f"Target measurement type: {args.target_measurement_type}")
    print(f"Plotted measurement type(s): {measurement_types_to_plot}")
    print(f"Number of sequences: {args.n_sequences}")
