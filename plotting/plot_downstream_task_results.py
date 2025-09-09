import argparse
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
import yaml
from matplotlib.colors import LinearSegmentedColormap

from zea.visualize import set_mpl_style

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from zea import init_device

    init_device()

from plot_psnr_dice import df_to_dict

from benchmark_active_sampling_ultrasound import extract_sweep_data
from plotting.plot_utils import ViolinPlotter

STRATEGY_COLORS = {
    "downstream_task_selection": "#d62728",  # Blue
    "greedy_entropy": "#ff7f0e",  # Orange
    "equispaced": "#2ca02c",  # Green
    "uniform_random": "#1f77b4",
}

STRATEGY_NAMES = {
    "downstream_task_selection": "Measurement Information Gain",
    "greedy_entropy": "Tissue Information Gain",
    # "greedy_entropy": "Active Perception",
    "uniform_random": "Uniform Random",
    "equispaced": "Equispaced",
}

# Canonical strategy mapping
STRATEGY_CANONICAL_MAP = {
    "downstream_task_selection": "downstream_task_selection",
    # Add more mappings if needed
}

STRATEGIES_TO_PLOT = [
    "downstream_task_selection",
    "greedy_entropy",
    "uniform_random",
    "equispaced",
    # Add/remove as needed
]

METRIC_NAMES = {
    "dice": "DICE [-]",
    "psnr": "PSNR [dB]",
    "ssim": "SSIM [-]",
    "measurement_length_mse_lvid": "LVID (pixels) MSE",
    "measurement_length_mse_lvpw": "LVPW (pixels) MSE",
    "measurement_length_mse_ivs": "IVS (pixels) MSE",
    "measurement_length_mae_lvid": "LVID [cm] MAE",
    "measurement_length_mae_lvpw": "LVPW [cm] MAE",
    "measurement_length_mae_ivs": "IVS [cm] MAE",
}


def canonical_strategy_key(strategy):
    """Map legacy strategy names to canonical keys."""
    return STRATEGY_CANONICAL_MAP.get(strategy, strategy)


# Add this near the top of the file where other constants are defined
AXIS_LABEL_MAP = {
    "n_actions": "# Scan Lines (out of 256)",
    # Add more mappings as needed
}


def load_yaml(filepath):
    """Load YAML file."""
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


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

        target_file = load_yaml(filepath_yaml)["target_filepath"]

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


def get_config_value(config, key_path):
    """Get value from config using dot notation."""
    current = config
    for key in key_path.split("."):
        if key not in current:
            return None
        current = current[key]
    return current


def get_axis_label(key):
    """Get friendly label for axis keys."""
    base_key = key.split(".")[-1]
    return AXIS_LABEL_MAP.get(base_key, base_key.replace("_", " ").title())


def plot_all_sweeps(
    sweep_results,
    save_root=None,
    x_axis_key="action_selection.n_actions",
    file_ext=".pdf",
):
    """Modified to handle configurable x-axis parameter."""

    # Find unique (agent_type, selection_strategy) pairs
    unique_pairs = set()
    for results, agent_type in sweep_results.values():
        for strategy in results[next(iter(results))]:  # Get sample metric's strategies
            unique_pairs.add((agent_type, strategy))

    # Assign unique colors and markers per (agent_type, selection_strategy) pair
    colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#17becf",  # Cyan
        "#bcbd22",  # Olive
    ]
    markers = ["o", "s", "D", "^", "v", "P", "*", "X"]  # A variety of markers
    pair_style_map = {
        pair: (colors[i], markers[i % len(markers)])
        for i, pair in enumerate(unique_pairs)
    }

    for metric_name in next(iter(sweep_results.values()))[0]:
        plt.figure(figsize=(8, 6))

        # Collect all x values across all strategies
        all_x_values = set()
        for _, (results, _) in sweep_results.items():
            for _, x_value_dict in results[metric_name].items():
                all_x_values.update(x_value_dict.keys())
        all_x_values = sorted(all_x_values)

        for sweep_name, (results, agent_type) in sweep_results.items():
            for strategy, x_value_dict in results[metric_name].items():
                mean_values = []
                sem_values = []
                valid_x_values = []

                for x_val in all_x_values:
                    if x_val in x_value_dict:
                        try:
                            metric_values = np.array(
                                x_value_dict[x_val], dtype=np.float64
                            )
                            if metric_values.ndim == 1:
                                mean_values.append(np.mean(metric_values))
                                sem_values.append(
                                    np.std(metric_values) / np.sqrt(len(metric_values))
                                )
                                valid_x_values.append(x_val)
                            else:
                                print(
                                    f"Skipping {sweep_name} - {strategy} for x_value {x_val} due to shape mismatch."
                                )
                        except Exception as e:
                            print(
                                f"Skipping {sweep_name} - {strategy} for x_value {x_val} due to error: {e}"
                            )

                if valid_x_values:
                    color, marker = pair_style_map[(agent_type, strategy)]
                    plt.errorbar(
                        valid_x_values,
                        mean_values,
                        yerr=sem_values,
                        label=f"{agent_type} - {strategy}",
                        marker=marker,
                        capsize=3,
                        color=color,
                        linestyle="-",
                    )

        # Update x-axis label and ticks
        x_label = get_axis_label(x_axis_key)
        plt.xlabel(x_label)
        plt.ylabel(metric_name.upper())
        plt.legend(fontsize=8, loc="best", title="Agent Type - Selection Strategy")

        # Set x-ticks to show all values
        plt.xticks(all_x_values)

        save_path = os.path.join(save_root, f"{metric_name}_combined_plot{file_ext}")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")


def plot_volume_over_time(sweep_results, gt_masks, save_root, max_videos=3):
    """Plot volume over time comparing ground truth with each method for individual videos."""
    # Define base colors with ground truth always black
    base_colors = {
        "ground_truth": "#000000",  # Black for ground truth
        # Additional colors for any other strategies
        "colors": [
            "#1f77b4",  # Blue
            "#ff7f0e",  # Orange
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#9467bd",  # Purple
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#17becf",  # Cyan
        ],
    }

    gt_sequences = list(gt_masks.values())[:max_videos]

    for video_idx, gt_sequence in enumerate(gt_sequences):
        plt.figure(figsize=(5, 3.5))

        # Plot ground truth volume
        gt_masks_seq = np.array(gt_sequence["masks"])
        gt_volumes = np.sum(gt_masks_seq, axis=(2, 3, 4, 5))  # Sum over N, H, W, C
        gt_volumes = np.mean(gt_volumes, axis=1)  # Average over particles

        plt.plot(
            gt_volumes,
            color=base_colors["ground_truth"],
            label="Ground Truth",
            linewidth=2,
            linestyle="--",
        )

        # Plot each strategy's volume for this video
        for sweep_name, (results, agent_type) in sweep_results.items():
            strategy_keys = list(results["masks"].keys())
            strategy_colors = {
                strategy: STRATEGY_COLORS[
                    strategy
                ]  # base_colors["colors"][i % len(base_colors["colors"])]
                for i, strategy in enumerate(strategy_keys)
            }

            for strategy in strategy_keys:
                if strategy != "downstream_propagation":
                    continue
                first_x_value = min(results["masks"][strategy].keys())

                if video_idx < len(results["masks"][strategy][first_x_value]):
                    video_data = results["masks"][strategy][first_x_value][video_idx]
                    mask_sequence = np.array(video_data["masks"])

                    # Handle same shape as ground truth
                    volumes = np.sum(
                        mask_sequence, axis=(2, 3, 4)
                    )  # Sum over N, H, W, C
                    volumes = np.mean(volumes, axis=1)  # Average over particles

                    plt.plot(
                        volumes,
                        color=strategy_colors[strategy],
                        label=f"{STRATEGY_NAMES[strategy]}",
                        linewidth=1.5,
                    )

        plt.xlabel("Frame")
        plt.ylabel("Mask Volume (pixels)")
        # plt.title(f"Segmentation Volume Over Time - Video {video_idx + 1}")
        # Move legend outside plot to top left
        plt.legend(loc="center left", bbox_to_anchor=(0.0, 1.2))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(
            save_root, f"volume_over_time_video_{video_idx + 1}.{FILE_EXT}"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved volume plot for video {video_idx + 1} to {save_path}")


def plot_image_mask_comparisons(sweep_results, gt_masks, save_root, max_videos=3):
    """Create side-by-side comparisons of images and masks."""

    colors = [(1, 0, 0, 0), (1, 0, 0, 0.5)]  # red with alpha=0 for 0, alpha=0.5 for 1
    segmentation_cmap = LinearSegmentedColormap.from_list("custom_red", colors)

    for sweep_name, (results, agent_type) in sweep_results.items():
        strategies = list(results["masks"].keys())

        # Get list of video sequences from ground truth
        gt_sequences = list(gt_masks.values())[:max_videos]

        for video_idx, gt_sequence in enumerate(gt_sequences):
            fig, axes = plt.subplots(
                1, len(strategies) + 1, figsize=(4 * (len(strategies) + 1), 4)
            )
            fig.suptitle(f"Video {video_idx + 1} Comparison (Middle Frame)")

            # Plot ground truth - handle gt_masks as dictionary
            gt_images = np.array(gt_sequence["x_scan_converted"])  # (T, 1, H, W, 1)
            gt_masks_seq = np.array(gt_sequence["masks"])  # (T, 1, 1, H, W, 1)

            # Get middle frame
            middle_frame = len(gt_images) // 2

            # Plot ground truth - squeeze extra dimensions
            gt_image = np.squeeze(gt_images[middle_frame])  # (H, W)
            gt_mask = np.squeeze(gt_masks_seq[middle_frame])  # (H, W)

            axes[0].imshow(gt_image, cmap="gray")
            axes[0].imshow(gt_mask, alpha=0.5, cmap=segmentation_cmap)
            axes[0].set_title("Ground Truth")
            axes[0].axis("off")

            # Plot each strategy
            for i, strategy in enumerate(strategies, 1):
                first_x_value = min(results["masks"][strategy].keys())
                pred_sequence = results["masks"][strategy][first_x_value][video_idx]

                pred_image = np.array(pred_sequence["x_scan_converted"][middle_frame])
                pred_mask = np.array(pred_sequence["masks"][middle_frame])

                # Squeeze extra dimensions, average over particles for mask if needed
                pred_image = np.squeeze(pred_image)  # (H, W)
                pred_mask = np.squeeze(pred_mask)  # (H, W)
                pred_image = pred_image[0]
                pred_mask = pred_mask[0]

                axes[i].imshow(pred_image, cmap="gray")
                axes[i].imshow(pred_mask, alpha=0.5, cmap=segmentation_cmap)
                axes[i].set_title(STRATEGY_NAMES[strategy])
                axes[i].axis("off")

            plt.tight_layout()
            save_path = os.path.join(
                save_root, f"comparison_video_{video_idx + 1}.{FILE_EXT}"
            )
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved comparison plot to {save_path}")


def plot_violin_sweeps(
    all_results,
    metric_names,
    save_root=None,
    x_axis_key="action_selection.n_actions",
    x_values=None,
    file_ext=".pdf",
    **kwargs,
):
    """Create violin plots showing distribution of patient means for each strategy."""

    for metric_name in metric_names:
        if metric_name == "masks":
            continue

        formatted_metric_name = METRIC_NAMES.get(
            metric_name.lower(), metric_name.upper()
        )

        plotter = ViolinPlotter(
            xlabel=get_axis_label(x_axis_key),
            ylabel=formatted_metric_name,
            group_names=STRATEGY_NAMES,
            legend_loc="top",
            scatter_kwargs={"alpha": 0.05, "s": 7},
            **kwargs,
        )

        plotter.plot(
            df_to_dict(all_results, metric_name),
            save_path=os.path.join(save_root, f"{metric_name}_violin_plot{file_ext}"),
            x_label_values=x_values,
            metric_name=formatted_metric_name,
        )


def generate_latex_table(results):
    """Generate LaTeX table of DICE scores with mean and standard error."""
    dice_results = results["dice"]

    # Get all strategies and num_lines
    strategies = sorted(dice_results.keys())
    all_num_lines = sorted(set().union(*[dice_results[s].keys() for s in strategies]))

    # Start LaTeX table
    latex_str = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{l" + "c" * len(all_num_lines) + "}",
        "\\toprule",
        "Strategy & " + " & ".join([f"{n} lines" for n in all_num_lines]) + " \\\\",
        "\\midrule",
    ]

    # Add rows for each strategy
    for strategy in strategies:
        row = [strategy.replace("_", "\\_")]  # Escape underscores for LaTeX

        for num_lines in all_num_lines:
            if num_lines in dice_results[strategy]:
                values = dice_results[strategy][num_lines]
                mean = np.mean(values)
                stderr = np.std(values) / np.sqrt(len(values))
                row.append(f"{mean:.3f} $\\pm$ {stderr:.3f}")
            else:
                row.append("-")

        latex_str.append(" & ".join(row) + " \\\\")

    # Close table
    latex_str.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{DICE scores (mean $\\pm$ standard error) for different sampling strategies and numbers of lines.}",
            "\\label{tab:dice_scores}",
            "\\end{table}",
        ]
    )

    return "\n".join(latex_str)


def compute_volume_errors(sweep_results, gt_masks, save_root):
    """Compute mean absolute error in pixels for volumes over time across all videos and sampling rates."""
    volume_errors = defaultdict(lambda: defaultdict(list))
    gt_sequences = list(gt_masks.values())

    # Get all x values (num_lines_to_sample)
    all_x_values = set()
    for _, (results, _) in sweep_results.items():
        for strategy in results["masks"]:
            all_x_values.update(results["masks"][strategy].keys())
    all_x_values = sorted(all_x_values)

    # Compute errors for each strategy, x_value, and video
    for video_idx, gt_sequence in enumerate(gt_sequences):
        # Calculate ground truth volumes
        gt_masks_seq = np.array(gt_sequence["masks"])
        gt_volumes = np.sum(gt_masks_seq, axis=(2, 3, 4, 5))  # Sum over N, H, W, C
        gt_volumes = np.mean(gt_volumes, axis=1)  # Average over particles

        # Calculate volumes for each strategy and x_value
        for sweep_name, (results, agent_type) in sweep_results.items():
            for strategy in results["masks"].keys():
                for x_value in all_x_values:
                    if x_value in results["masks"][strategy]:
                        if video_idx < len(results["masks"][strategy][x_value]):
                            video_data = results["masks"][strategy][x_value][video_idx]
                            mask_sequence = np.array(video_data["masks"])

                            # Calculate volumes
                            volumes = np.sum(
                                mask_sequence, axis=(2, 3, 4)
                            )  # Sum over N, H, W, C
                            volumes = np.mean(volumes, axis=1)  # Average over particles

                            # Calculate MAE
                            mae = np.mean(np.abs(volumes - gt_volumes))
                            volume_errors[strategy][x_value].append(mae)

    # Create violin plot
    plt.figure(figsize=(5, 3))

    # Create equally spaced positions for x values
    plot_positions = np.arange(len(all_x_values))
    x_value_to_pos = dict(zip(all_x_values, plot_positions))

    width = 0.5
    strategy_offset = np.linspace(-width / 2, width / 2, len(volume_errors))

    # Plot violins for each strategy
    for strategy_idx, (strategy, x_value_dict) in enumerate(volume_errors.items()):
        violin_positions = []
        violin_data = []

        for x_val in all_x_values:
            if x_val in x_value_dict:
                pos = x_value_to_pos[x_val] + strategy_offset[strategy_idx]
                violin_positions.append(pos)
                violin_data.append(x_value_dict[x_val])

        if violin_data:
            parts = plt.violinplot(
                violin_data, positions=violin_positions, widths=width / 4
            )

            # Use consistent colors from STRATEGY_COLORS
            strategy_color = STRATEGY_COLORS[strategy]
            for pc in parts["bodies"]:
                pc.set_facecolor(strategy_color)
                pc.set_alpha(0.7)
            parts["cbars"].set_color(strategy_color)
            parts["cmins"].set_color(strategy_color)
            parts["cmaxes"].set_color(strategy_color)

            # Add scatter points
            for pos, data in zip(violin_positions, violin_data):
                plt.scatter(
                    [pos] * len(data), data, color=strategy_color, alpha=0.4, s=20
                )

            plt.scatter(
                [], [], color=strategy_color, label=f"{STRATEGY_NAMES[strategy]}"
            )

    plt.xlabel("# Scan Lines Acquired")
    plt.ylabel("Volume MAE (pixels)")
    plt.legend(loc="center left", bbox_to_anchor=(0.0, 1.20))
    plt.grid(True, alpha=0.3)
    plt.xticks(plot_positions, [str(x) for x in all_x_values])

    save_path = os.path.join(save_root, f"volume_mae_violin_plot.{FILE_EXT}")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Volume MAE violin plot saved to {save_path}")

    latex_str = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{l" + "c" * len(all_x_values) + "}",
        "\\toprule",
        "Strategy & " + " & ".join([f"{n} lines" for n in all_x_values]) + " \\\\",
        "\\midrule",
    ]

    # Add rows for each strategy
    for strategy in sorted(volume_errors.keys()):
        row = [STRATEGY_NAMES[strategy]]  # Use strategy display names

        for x_val in all_x_values:
            if x_val in volume_errors[strategy]:
                values = volume_errors[strategy][x_val]
                mean = np.mean(values)
                stderr = np.std(values) / np.sqrt(len(values))
                row.append(f"{mean:.1f} $\\pm$ {stderr:.1f}")
            else:
                row.append("-")

        latex_str.append(" & ".join(row) + " \\\\")

    # Close table
    latex_str.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Volume mean absolute error in pixels (mean $\\pm$ standard error) for different sampling strategies and numbers of lines.}",
            "\\label{tab:volume_mae}",
            "\\end{table}",
        ]
    )

    # Save LaTeX table
    table_path = os.path.join(save_root, "volume_mae_table.tex")
    with open(table_path, "w") as f:
        f.write("\n".join(latex_str))
    print(f"Volume MAE table saved to {table_path}")

    return volume_errors


def get_target_file_from_run(run_dir):
    """Load the target_file from a run directory."""
    filepath_yaml = Path(run_dir) / "target_filepath.yaml"
    if not filepath_yaml.exists():
        return None
    with open(filepath_yaml, "r") as f:
        return yaml.safe_load(f)["target_filepath"]


def replay_dice_outliers(
    combined_results,
    gt_masks,
    top_k=3,
    replay_script="/latent-ultrasound-diffusion/active_sampling/ultrasound_line_scanning_agent/plotting/replay_result_frames.py",
    strategy=None,
    x_value=None,
):
    """
    Find the worst DICE outlier frames and replay both predicted and ground truth results.
    """
    # Gather all DICE scores with their run/frame/strategy/x_value info
    dice_entries = []
    for strat in combined_results["dice"]:
        if strategy is not None and strat != strategy:
            continue
        for x_val in combined_results["dice"][strat]:
            if x_value is not None and x_val != x_value:
                continue
            for run_idx, dice in enumerate(combined_results["dice"][strat][x_val]):
                pred_run = combined_results["masks"][strat][x_val][run_idx]
                pred_run_dir = pred_run.get("run_dir", None)
                if pred_run_dir is None:
                    continue
                # Get target_file for this run
                target_file = get_target_file_from_run(pred_run_dir)
                # Find the ground truth run_dir from gt_masks
                gt_entry = gt_masks.get(target_file, None)
                gt_run_dir = None
                if gt_entry is not None and "run_dir" in gt_entry:
                    gt_run_dir = gt_entry["run_dir"]
                dice_entries.append(
                    {
                        "strategy": strat,
                        "x_value": x_val,
                        "run_idx": run_idx,
                        "dice": dice,
                        "pred_run_dir": pred_run_dir,
                        "gt_run_dir": gt_run_dir,
                        "target_file": target_file,
                    }
                )

    # Sort by DICE (ascending: lowest/worst first)
    dice_entries = [
        e
        for e in dice_entries
        if e["pred_run_dir"] is not None and e["gt_run_dir"] is not None
    ]
    dice_entries.sort(key=lambda d: d["dice"])
    outliers = dice_entries[:top_k]

    print("\nWorst DICE score frames:")
    for entry in outliers:
        print(
            f"Strategy: {entry['strategy']}, x_value: {entry['x_value']}, run_idx: {entry['run_idx']}, "
            f"DICE: {entry['dice']:.4f}, pred_run_dir: {entry['pred_run_dir']}, gt_run_dir: {entry['gt_run_dir']}, target_file: {entry['target_file']}"
        )

    # For each outlier, replay both predicted and ground truth
    for entry in outliers:
        pred_metrics_path = Path(entry["pred_run_dir"]) / "metrics.npz"
        gt_metrics_path = Path(entry["gt_run_dir"]) / "metrics.npz"

        # Predicted
        pred_save_dir = (
            Path(pred_metrics_path).parent
            / f"replay_predicted_{entry['strategy']}_x{entry['x_value']}_run{entry['run_idx']}"
        )
        subprocess.run(
            [
                sys.executable,
                replay_script,
                "--results_path",
                str(pred_metrics_path),
                "--save_dir",
                str(pred_save_dir),
                "--make_gif",
                "True",
            ]
        )

        # Ground truth
        if gt_metrics_path.exists():
            gt_save_dir = (
                Path(gt_metrics_path).parent
                / f"replay_groundtruth_{entry['strategy']}_x{entry['x_value']}_run{entry['run_idx']}"
            )
            subprocess.run(
                [
                    sys.executable,
                    replay_script,
                    "--results_path",
                    str(gt_metrics_path),
                    "--save_dir",
                    str(gt_save_dir),
                    "--make_gif",
                    "True",
                ]
            )
        else:
            print(f"Ground truth metrics.npz not found for {entry['gt_run_dir']}")


def compute_volume_mae(sweep_results, gt_masks, save_root):
    """Compute mean absolute error in pixels for volumes over time across all videos and sampling rates."""
    volume_mae = defaultdict(lambda: defaultdict(list))
    gt_sequences = list(gt_masks.values())

    # Get all x values (num_lines_to_sample)
    all_x_values = set()
    for _, (results, _) in sweep_results.items():
        for strategy in results["masks"]:
            all_x_values.update(results["masks"][strategy].keys())
    all_x_values = sorted(all_x_values)

    # Compute errors for each strategy, x_value, and video
    for video_idx, gt_sequence in enumerate(gt_sequences):
        gt_masks_seq = np.array(gt_sequence["masks"])
        gt_volumes = np.sum(gt_masks_seq, axis=(2, 3, 4, 5))
        gt_volumes = np.mean(gt_volumes, axis=1)

        for sweep_name, (results, agent_type) in sweep_results.items():
            for strategy in results["masks"].keys():
                for x_value in all_x_values:
                    if x_value in results["masks"][strategy]:
                        if video_idx < len(results["masks"][strategy][x_value]):
                            video_data = results["masks"][strategy][x_value][video_idx]
                            mask_sequence = np.array(video_data["masks"])
                            volumes = np.sum(mask_sequence, axis=(2, 3, 4))
                            volumes = np.mean(volumes, axis=1)
                            mae = np.mean(np.abs(volumes - gt_volumes))
                            volume_mae[strategy][x_value].append(mae)

    # Create violin plot
    plt.figure(figsize=(5, 3))
    plot_positions = np.arange(len(all_x_values))
    x_value_to_pos = dict(zip(all_x_values, plot_positions))
    width = 0.5
    strategy_keys = list(STRATEGY_NAMES.keys())
    strategy_offset = np.linspace(-width / 2, width / 2, len(strategy_keys))

    # Plot violins for each strategy in the order of STRATEGY_NAMES
    for strategy_idx, strategy in enumerate(strategy_keys):
        x_value_dict = volume_mae.get(strategy, {})
        violin_positions = []
        violin_data = []

        for x_val in all_x_values:
            if x_val in x_value_dict:
                pos = x_value_to_pos[x_val] + strategy_offset[strategy_idx]
                violin_positions.append(pos)
                violin_data.append(x_value_dict[x_val])

        if violin_data:
            parts = plt.violinplot(
                violin_data, positions=violin_positions, widths=width / 4
            )
            strategy_color = STRATEGY_COLORS[strategy]
            for pc in parts["bodies"]:
                pc.set_facecolor(strategy_color)
                pc.set_alpha(0.7)
            parts["cbars"].set_color(strategy_color)
            parts["cmins"].set_color(strategy_color)
            parts["cmaxes"].set_color(strategy_color)
            for pos, data in zip(violin_positions, violin_data):
                plt.scatter(
                    [pos] * len(data), data, color=strategy_color, alpha=0.4, s=20
                )
            plt.scatter(
                [], [], color=strategy_color, label=f"{STRATEGY_NAMES[strategy]}"
            )

    plt.xlabel("# Scan Lines Acquired")
    plt.ylabel("Volume MAE (pixels)")
    plt.legend(loc="center left", bbox_to_anchor=(0.0, 1.20))
    plt.grid(True, alpha=0.3)
    plt.xticks(plot_positions, [str(x) for x in all_x_values])

    save_path = os.path.join(save_root, f"volume_mae_violin_plot.{FILE_EXT}")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Volume MAE violin plot saved to {save_path}")

    return volume_mae


def plot_strategy_comparison_scatter(
    combined_results,
    metric,
    strategy_x,
    strategy_y,
    save_root=".",
    x_axis_label=None,
    y_axis_label=None,
    plot_title=None,
    file_ext=".pdf",
):
    """
    Scatter plot comparing metric values for two strategies across all samples.
    X-axis: metric for strategy_x
    Y-axis: metric for strategy_y
    """
    import matplotlib.pyplot as plt

    # Get metric values for both strategies
    x_dict = combined_results[metric].get(strategy_x, {})
    y_dict = combined_results[metric].get(strategy_y, {})

    # Find all x_values present in both strategies
    x_values = sorted(set(x_dict.keys()) & set(y_dict.keys()))
    if not x_values:
        print(
            f"No overlapping x_values for {strategy_x} and {strategy_y} in metric '{metric}'"
        )
        return

    # For each x_value, plot all samples
    for x_val in x_values:
        x_samples = x_dict[x_val]
        y_samples = y_dict[x_val]
        # Only plot pairs where both strategies have a value for the same sample index
        n = min(len(x_samples), len(y_samples))
        if n == 0:
            continue

        plt.figure(figsize=(5, 5))
        plt.scatter(
            x_samples[:n],
            y_samples[:n],
            alpha=0.6,
            color="#1f77b4",
            s=20,
            edgecolors="none",
        )

        # Plot y=x reference line
        min_val = min(min(x_samples[:n]), min(y_samples[:n]))
        max_val = max(max(x_samples[:n]), max(y_samples[:n]))
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "k--",
            alpha=0.7,
            linewidth=1,
            label="y = x",
        )

        # Calculate correlation coefficient
        import scipy.stats

        r, p_value = scipy.stats.pearsonr(x_samples[:n], y_samples[:n])

        plt.xlabel(
            x_axis_label
            or f"{STRATEGY_NAMES.get(strategy_x, strategy_x)}\n{METRIC_NAMES.get(metric, metric)}",
            fontsize=9,
        )
        plt.ylabel(
            y_axis_label
            or f"{STRATEGY_NAMES.get(strategy_y, strategy_y)}\n{METRIC_NAMES.get(metric, metric)}",
            fontsize=9,
        )
        plt.title(
            plot_title
            or f"{METRIC_NAMES.get(metric, metric)} Comparison (n_lines={x_val})\nr = {r:.3f}, p = {p_value:.3f}",
            fontsize=10,
        )
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis="both", labelsize=8)
        plt.tight_layout()

        save_path = os.path.join(
            save_root,
            f"{metric}_scatter_{strategy_x}_vs_{strategy_y}_x{x_val}{file_ext}",
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved scatter plot to {save_path}")


def plot_combined_measurement_violin(
    all_results,
    save_root=None,
    x_axis_key="action_selection.n_actions",
    x_values=None,
    file_ext=".pdf",
    context="styles/ieee-tmi.mplstyle",
    measurement_types_to_plot=None,  # New parameter
):
    """Create combined violin plots for all measurement types in a single row."""

    # Use the same measurement metrics as defined in METRICS (with uppercase endings)
    all_measurement_metrics = [
        "measurement_length_mae_LVPW",
        "measurement_length_mae_LVID",
        "measurement_length_mae_IVS",
    ]

    all_subplot_titles = ["LVPW", "LVID", "IVS"]

    # Filter based on measurement_types_to_plot
    if measurement_types_to_plot is None:
        measurement_metrics = all_measurement_metrics
        subplot_titles = all_subplot_titles
    else:
        # Map measurement types to metrics and titles
        type_to_metric = {
            "LVPW": "measurement_length_mae_LVPW",
            "LVID": "measurement_length_mae_LVID",
            "IVS": "measurement_length_mae_IVS",
        }
        type_to_title = {"LVPW": "LVPW", "LVID": "LVID", "IVS": "IVS"}

        measurement_metrics = [
            type_to_metric[mt]
            for mt in measurement_types_to_plot
            if mt in type_to_metric
        ]
        subplot_titles = [
            type_to_title[mt] for mt in measurement_types_to_plot if mt in type_to_title
        ]

    num_measurements = len(measurement_metrics)

    # Define colors to match default violin plots (blue and orange)
    default_colors = {
        "downstream_task_selection": "#1f77b4",  # Blue
        "greedy_entropy": "#ff7f0e",  # Orange
        "uniform_random": "#2ca02c",  # Green
        "equispaced": "#d62728",  # Red
    }

    with plt.style.context(context):
        # Adjust figure size based on number of measurements
        if num_measurements == 1:
            figsize = (3.5, 2.5)  # Single column width for one measurement
        else:
            figsize = (7.16, 2.5)  # IEEE two-column width for multiple measurements

        fig, axes = plt.subplots(1, num_measurements, figsize=figsize, sharey=False)

        # Handle single subplot case
        if num_measurements == 1:
            axes = [axes]

        # Create ViolinPlotter instance with default colors
        plotter = ViolinPlotter(
            xlabel=get_axis_label(x_axis_key),
            ylabel="MAE [cm]",
            group_names=STRATEGY_NAMES,
            group_colors=default_colors,  # Use default blue/orange colors
            legend_loc="top",
            scatter_kwargs={"alpha": 0.05, "s": 7},
        )

        for idx, (metric_name, title) in enumerate(
            zip(measurement_metrics, subplot_titles)
        ):
            ax = axes[idx]

            # Check if metric exists in results
            if metric_name not in all_results.columns:
                print(f"Warning: {metric_name} not found in results")
                print(f"Available columns: {list(all_results.columns)}")
                continue

            # Get data for this metric using the same approach as plot_violin_sweeps
            metric_data = df_to_dict(all_results, metric_name)

            if not metric_data:
                print(f"Warning: No data found for {metric_name}")
                continue

            # Use the ViolinPlotter's _plot_core method to plot on this specific axis
            plotter._plot_core(
                ax,
                metric_data,
                x_label_values=x_values,
                metric_name=None,  # We'll set our own y-label
                groups_to_plot=None,  # Use all available groups
                ylim=None,
                order_by="mean",
                reverse_order=True,
                width=0.5,
            )

            # Customize this specific subplot
            if num_measurements > 1:
                ax.set_title(title, fontsize=9, pad=8)

            # Set x-axis label - for single measurement, always show; for multiple, show on middle
            if num_measurements == 1 or idx == num_measurements // 2:
                ax.set_xlabel(get_axis_label(x_axis_key), fontsize=10)
            else:
                ax.set_xlabel("")  # Clear x-label for other subplots

            # Set y-axis label only on leftmost subplot
            if idx == 0:
                ax.set_ylabel("MAE [cm]", fontsize=10)
            else:
                ax.set_ylabel("")  # Clear y-label for other subplots

            # Update font sizes to match IEEE requirements
            ax.tick_params(axis="both", labelsize=10)

            # Set y-axis to start from 0 for better comparison
            current_ylim = ax.get_ylim()
            ax.set_ylim(bottom=0, top=current_ylim[1])

        # Add legend above the plots
        # Collect legend handles and labels from the first subplot
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.45, 0.94),  # Centered above the plots
                # ncol=len(handles),  # Arrange horizontally
                ncol=1,  # Arrange vertically
                fontsize=10,
                framealpha=0.9,
            )

        # Adjust layout to accommodate legend above
        plt.tight_layout()

        # Adjust spacing based on number of measurements
        if num_measurements == 1:
            plt.subplots_adjust(top=0.72)  # No need for wspace with single plot
        else:
            plt.subplots_adjust(top=0.72, wspace=0.2)  # Leave space at top for legend

        # Save plot with measurement type info in filename
        if measurement_types_to_plot is None:
            filename_suffix = "combined_measurement_mae_violin"
        elif len(measurement_types_to_plot) == 1:
            filename_suffix = f"{measurement_types_to_plot[0]}_measurement_mae_violin"
        else:
            filename_suffix = (
                f"{'_'.join(measurement_types_to_plot)}_measurement_mae_violin"
            )

        save_path = os.path.join(save_root, f"{filename_suffix}{file_ext}")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved combined measurement MAE violin plot to {save_path}")


# Add this import at the top with other imports
import pickle

# Replace the main section starting from "TEMP_FILE = Path..." with this:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--x-axis",
        type=str,
        default="action_selection.n_actions",
        help="Config key to use for x-axis (dot notation)",
    )
    parser.add_argument("--save-root", type=str, default=".")
    parser.add_argument(
        "--measurement-to-plot",
        type=str,
        default="LVID",
        choices=["LVPW", "LVID", "IVS", "all"],
        help="Which measurement type to plot (default: all)",
    )
    args = parser.parse_args()

    # set_mpl_style()  # darkmode

    SWEEP_PATHS = [
        # "/mnt/z/Ultrasound-BMd/data/oisin/ULSA_out_dst/echonetlvh_downstream_task/26_08_25_run4/sweep_2025_08_26_213849_625395",
        # "/mnt/z/Ultrasound-BMd/data/oisin/ULSA_out_dst/echonetlvh_downstream_task/26_08_25_run4/sweep_2025_08_26_170059_245130",
        "/mnt/z/Ultrasound-BMd/data/oisin/ULSA_out_dst/echonetlvh_downstream_task/27_08_25_run1/sweep_2025_08_28_165310_165677",
        "/mnt/z/Ultrasound-BMd/data/oisin/ULSA_out_dst/echonetlvh_downstream_task/27_08_25_run1/sweep_2025_08_29_154154_763351",
    ]

    METRICS = [
        "measurement_length_mae_LVPW",
        "measurement_length_mae_LVID",
        "measurement_length_mae_IVS",
    ]

    TEMP_FILE = Path("/tmp/plot_downstream_task_results.pkl")

    # Check if cached results exist
    if TEMP_FILE.exists():
        print(f"Loading existing combined results from {str(TEMP_FILE)}")
        with open(TEMP_FILE, "rb") as f:
            combined_results = pickle.load(f)
    else:
        print("No cached results found. Extracting data from sweep paths...")
        # Aggregate results from all sweep paths
        combined_results = []
        for sweep_path in SWEEP_PATHS:
            try:
                print(f"Processing sweep: {sweep_path}")
                results = extract_sweep_data(
                    sweep_path,
                    keys_to_extract=METRICS,
                    x_axis_key=args.x_axis,
                )
                combined_results.append(results)
            except Exception as e:
                print(f"Failed to process {sweep_path}: {e}")
        combined_results = pd.concat(combined_results)
        # Save combined results to cache
        print(f"Saving combined results to {str(TEMP_FILE)}")
        with open(TEMP_FILE, "wb") as f:
            combined_results.to_pickle(TEMP_FILE)

    if combined_results is not None:
        # Determine which measurements to plot
        if args.measurement_to_plot == "all":
            measurement_types_to_plot = None  # Plot all measurements
        else:
            measurement_types_to_plot = [args.measurement_to_plot]

        # Plot combined measurement violin plots with measurement filtering
        for file_ext in [".png", ".pdf"]:
            plot_combined_measurement_violin(
                combined_results,
                save_root=args.save_root,
                x_axis_key=args.x_axis,
                file_ext=file_ext,
                context="styles/ieee-tmi.mplstyle",
                measurement_types_to_plot=measurement_types_to_plot,
            )

        # Plot individual violin sweeps for each metric (optionally filtered)
        if args.measurement_to_plot != "all":
            # Filter METRICS based on selected measurement type
            type_to_metric = {
                "LVPW": "measurement_length_mae_LVPW",
                "LVID": "measurement_length_mae_LVID",
                "IVS": "measurement_length_mae_IVS",
            }
            filtered_metrics = [type_to_metric[args.measurement_to_plot]]
        else:
            filtered_metrics = METRICS

        for file_ext in [".png", ".pdf"]:
            plot_violin_sweeps(
                combined_results,
                filtered_metrics,
                save_root=args.save_root,
                x_axis_key=args.x_axis,
                file_ext=file_ext,
                # context="styles/ieee-tmi.mplstyle",
            )

        # Create strategy comparison scatter plots (optionally filtered)
        strategies_to_compare = [
            ("downstream_task_selection", "greedy_entropy"),
            ("downstream_task_selection", "uniform_random"),
            ("greedy_entropy", "uniform_random"),
        ]

        # Convert combined_results DataFrame to the format expected by plot_strategy_comparison_scatter
        combined_results_dict = {}
        for metric in filtered_metrics:
            combined_results_dict[metric] = df_to_dict(combined_results, metric)

        for strategy_x, strategy_y in strategies_to_compare:
            for metric in filtered_metrics:
                plot_strategy_comparison_scatter(
                    combined_results_dict,
                    metric=metric,
                    strategy_x=strategy_x,
                    strategy_y=strategy_y,
                    save_root=args.save_root,
                )

    else:
        print("No valid results found in any of the provided paths.")
