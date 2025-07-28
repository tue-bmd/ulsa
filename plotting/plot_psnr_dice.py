"""
Makes violin plots of PSNR and DICE scores for the various scan line selection strategies.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd  # pip install pandas
import scipy.ndimage
from rich.console import Console
from rich.table import Table

from zea import Config, init_device, log

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "numpy"
    init_device("cpu")
    sys.path.append("/ulsa")


from benchmark_active_sampling_ultrasound import compute_dice_score, get_config_value
from plotting.plot_utils import ViolinPlotter, natural_sort

DATA_ROOT = "/mnt/z/Ultrasound-BMD/Ultrasound-BMd/data"
DATA_FOLDER = Path(DATA_ROOT) / "Wessel/output/lud/ULSA_benchmarks"

STRATEGY_COLORS = {
    "downstream_propagation_summed": "#d62728",  # Red
    "greedy_entropy": "#1f77b4",  # Blue
    "equispaced": "#2ca02c",  # Green
    "uniform_random": "#ff7f0e",  # Orange
}

STRATEGY_NAMES = {
    "downstream_propagation_summed": "Measurement Information Gain",
    # "greedy_entropy": "Tissue Information Gain",
    "greedy_entropy": "Active Perception",
    "uniform_random": "Random",
    "equispaced": "Equispaced",
}

# Canonical strategy mapping
STRATEGY_CANONICAL_MAP = {
    "downstream_propagation": "downstream_propagation_summed",
    # Add more mappings if needed
}

STRATEGIES_TO_PLOT = [
    # "downstream_propagation_summed",
    "greedy_entropy",
    "uniform_random",
    "equispaced",
    # Add/remove as needed
]

METRIC_NAMES = {
    "dice": "DICE [-]",
    "psnr": "PSNR [dB]",
    "ssim": "SSIM [-]",
}

# Add this near the top of the file where other constants are defined
AXIS_LABEL_MAP = {
    "n_actions": "# Scan Lines (out of 112)",
    # Add more mappings as needed
}


def extract_sweep_data(
    sweep_dir: str,
    keys_to_extract=["mse", "psnr"],
    x_axis_key="action_selection.n_actions",
    strategies_to_plot=None,
    gt_masks=None,
    include_only_these_files=None,
    ef_lookup=None,
):
    """Can be used to load all the metrics from the run_benchmark function."""
    sweep_details_path = os.path.join(sweep_dir, "sweep_details.yaml")
    if not os.path.exists(sweep_details_path):
        raise FileNotFoundError(f"Missing sweep_details.yaml in {sweep_dir}")

    results = []

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

        # Get EF value
        filename = Path(target_file).stem
        if ef_lookup is not None:
            if filename not in ef_lookup:
                log.warning(f"No EF data found for {filename}")
            ef_value = ef_lookup[filename]
        else:
            ef_value = None

        # Compute DICE if ground truths available
        if gt_masks is not None and target_file in gt_masks:
            pred_masks = metrics["segmentation_mask"]
            gt_sequence = gt_masks[target_file]
            pred_masks = np.array(pred_masks)
            gt_masks_array = np.array(gt_sequence["masks"])
            gt_masks_array = np.squeeze(gt_masks_array, axis=1)
            dice_score = compute_dice_score(pred_masks, gt_masks_array)
            mean_dice = np.mean(dice_score)
        else:
            mean_dice = None

            # Store mean DICE over patient frames
            # results["dice"][selection_strategy][x_value].append(mean_dice)

        # Add other metrics
        metric_results = {}
        for metric_name in keys_to_extract:
            if metric_name not in metrics:
                continue  # Skip if metric not found

            metric_values = metrics[metric_name]
            if isinstance(metric_values, np.ndarray) and metric_values.size > 0:
                sequence_means = np.mean(metric_values, axis=-1)
                metric_results[metric_name] = sequence_means

        results.append(
            {
                "EF": ef_value,
                "selection_strategy": selection_strategy,
                "x_value": x_value,
                "filepath": target_file,
                "filename": filename,
                "dice": mean_dice,
                **metric_results,
            }
        )

    return pd.DataFrame(results)


def get_ground_truth_masks(fully_observed_path):
    """Build dictionary mapping target filepaths to ground truth masks."""
    gt_masks = {}

    for run_dir in sorted(os.listdir(fully_observed_path)):
        run_path = os.path.join(fully_observed_path, run_dir)
        if not os.path.isdir(run_path):
            continue

        filepath_yaml = os.path.join(run_path, "target_filepath.yaml")
        metrics_path = os.path.join(run_path, "metrics.npz")

        if not (os.path.exists(filepath_yaml) and os.path.exists(metrics_path)):
            continue

        target_file = Config.from_yaml(filepath_yaml)["target_filepath"]

        # Load ground truth masks and corresponding images
        metrics = np.load(metrics_path, allow_pickle=True)

        # Create a dictionary for this sequence containing both masks and images
        metadata = metrics["metadata"].item()
        sequence_data = {
            "masks": [mask[None] for mask in metadata["masks"]],
            "x_scan_converted": [
                x[None] for x in metadata["x_scan_converted"][..., None]
            ],
            "run_dir": Path(fully_observed_path) / run_dir,
        }

        gt_masks[target_file] = sequence_data

    return gt_masks


def filter_gt_masks_by_blobs(gt_masks, max_blobs=1, max_bad_frames=5, verbose=True):
    """
    Remove ground truth sequences where the number of blobs in the mask exceeds max_blobs
    for more than max_bad_frames frames.

    Args:
        gt_masks (dict): Mapping from target_filepath to sequence_data.
        max_blobs (int): Maximum allowed blobs per frame.
        max_bad_frames (int): Maximum allowed frames exceeding max_blobs.
        verbose (bool): Print filtered files and summary.

    Returns:
        filtered_gt_masks (dict): Filtered gt_masks.
    """
    filtered_gt_masks = {}
    filtered_out = []

    for target_file, sequence_data in gt_masks.items():
        masks = sequence_data["masks"]  # List of (1, H, W, 1) arrays
        bad_frame_count = 0
        for mask in masks:
            mask_arr = np.squeeze(mask)  # (H, W)
            # Count connected components (blobs)
            labeled, num_blobs = scipy.ndimage.label(mask_arr > 0)
            if num_blobs > max_blobs:
                bad_frame_count += 1
        if bad_frame_count > max_bad_frames:
            filtered_out.append((target_file, bad_frame_count))
        else:
            filtered_gt_masks[target_file] = sequence_data

    if verbose:
        print(
            f"\nFiltered out {len(filtered_out)} ground truth sequences due to excessive blobs:"
        )
        for target_file, bad_count in filtered_out:
            print(f"  {target_file} (bad frames: {bad_count})")
        print(f"Remaining ground truth sequences: {len(filtered_gt_masks)}")

    return filtered_gt_masks


def get_axis_label(key):
    """Get friendly label for axis keys."""
    base_key = key.split(".")[-1]
    return AXIS_LABEL_MAP.get(base_key, base_key.replace("_", " ").title())


def sort_by_names(combined_results, names):
    """Sort combined results by strategy names."""
    return {k: combined_results[k] for k in names if k in combined_results}


def extract_and_combine_sweep_data(sweep_dirs, *args, **kwargs):
    combined_results = []

    for sweep_dir in sweep_dirs:
        try:
            combined_results.append(extract_sweep_data(sweep_dir, *args, **kwargs))

        except Exception as e:
            print(f"Failed to process {sweep_dir}: {e}")
    return pd.concat(combined_results, ignore_index=True)  # ignore_index?


def df_to_dict(df: pd.DataFrame, metric_name: str, filter_nan=True):
    """Convert DataFrame to a nested dictionary for plotting.

    Args:
        df (pd.DataFrame): DataFrame containing the results.
        metric_name (str): Name of the metric to extract.
        filter_nan (bool): Whether to filter out NaN and None values.
            Used in our case to drop the failed ground truth segmentation.
    Returns:
        dict: Nested dictionary with selection strategies as keys and x_values as sub-keys.
    """
    result = {}
    for _, row in df.iterrows():
        strategy = row["selection_strategy"]
        x_value = row["x_value"]
        value = row[metric_name]
        if filter_nan and (value is None or np.isnan(value).any()):
            continue

        if strategy not in result:
            result[strategy] = {}
        if x_value not in result[strategy]:
            result[strategy][x_value] = []
        result[strategy][x_value].append(value)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--x-axis",
        type=str,
        default="action_selection.n_actions",
        help="Config key to use for x-axis (dot notation)",
    )
    args = parser.parse_args()

    TEMP_FILE = Path("./plot_psnr_dice.pkl")

    if TEMP_FILE.exists():
        print(f"Loading existing combined results from {str(TEMP_FILE)}")
        combined_results = pd.read_pickle(TEMP_FILE)
    else:
        FULLY_OBSERVED_PATH = DATA_FOLDER / "fully_observed_echonet_seg"
        SUBSAMPLED_PATHS = [
            DATA_FOLDER / "sharding_sweep_2025-05-30_08-56-07",
            DATA_FOLDER / "sharding_sweep_2025-06-04_13-52-43",
        ]

        # Get ground truth masks from fully observed runs
        gt_masks = get_ground_truth_masks(FULLY_OBSERVED_PATH)

        # Filter ground truth masks by blobs
        gt_masks = filter_gt_masks_by_blobs(gt_masks, max_blobs=1, max_bad_frames=2)

        combined_results = extract_and_combine_sweep_data(
            SUBSAMPLED_PATHS,
            gt_masks=gt_masks,
            keys_to_extract=["mse", "psnr", "dice"],
            x_axis_key=args.x_axis,
        )
        combined_results.to_pickle(TEMP_FILE)

    plotter = ViolinPlotter(
        xlabel=get_axis_label(args.x_axis),
        group_names=STRATEGY_NAMES,
        legend_loc="top",
        scatter_kwargs={"alpha": 0.05, "s": 7},
        context="styles/ieee-tmi.mplstyle",
    )

    # DICE plot
    metric_name = "dice"
    x_values = [2, 4, 7, 14]
    formatted_metric_name = METRIC_NAMES.get(metric_name, metric_name.upper())
    plotter.plot(
        df_to_dict(combined_results, metric_name),
        save_path=f"./echonet_{metric_name}_violin_plot.pdf",
        x_label_values=x_values,
        metric_name=formatted_metric_name,
        groups_to_plot=["greedy_entropy", "uniform_random", "equispaced"],
        ylim=[0.58, 1.02],
    )

    # PSNR plot
    metric_name = "psnr"
    x_values = [7, 14, 28]
    formatted_metric_name = METRIC_NAMES.get(metric_name, metric_name.upper())
    plotter.plot(
        df_to_dict(combined_results, metric_name),
        save_path=f"./{metric_name}_violin_plot.pdf",
        x_label_values=x_values,
        metric_name=formatted_metric_name,
    )

    # Print results in a table format
    for metric_name in ["dice", "psnr"]:
        table = Table(title=f"{metric_name.upper()} Results", show_lines=True)
        table.add_column("Strategy", style="cyan", no_wrap=True)
        table.add_column(get_axis_label(args.x_axis), style="magenta")
        table.add_column("Mean", style="green")
        table.add_column("Std", style="yellow")
        table.add_column("Count", style="white")

        results = df_to_dict(combined_results, metric_name)
        for group in results.keys():
            for x_value in natural_sort(results[group].keys()):
                values = results[group][x_value]
                mean = np.mean(values)
                std = np.std(values)
                count = len(values)
                table.add_row(
                    str(STRATEGY_NAMES.get(group, group)),
                    str(x_value),
                    f"{mean:.2f}",
                    f"{std:.2f}",
                    str(count),
                )

        console = Console()
        console.print(table)
