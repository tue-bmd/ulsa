"""
Create a combined plot showing omega and n_frames sweeps for LPIPS and PSNR metrics.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zea import Config, init_device, log

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "numpy"
    init_device("cpu")
    sys.path.append("/ulsa")

from plotting.index import extract_sweep_data
from plotting.plot_n_frames_sweep import (
    N_FRAMES_CACHE,
    add_n_frames_column,
)

STRATEGY_NAMES = {
    "greedy_entropy": "Active Perception",
    "uniform_random": "Random",
    "equispaced": "Equispaced",
}

METRIC_NAMES = {
    "psnr": "PSNR (→) [dB]",
    "lpips": "LPIPS (←) [-]",
}

# Colors for different n_actions values
N_ACTIONS_COLORS = {
    2: "#1f77b4",  # Blue
    4: "#ff7f0e",  # Orange
    7: "#2ca02c",  # Green
    14: "#d62728",  # Red
    28: "#9467bd",  # Purple
}

N_ACTIONS_MARKERS = {
    2: "o",  # Circle
    4: "s",  # Square
    7: "^",  # Triangle up
    14: "D",  # Diamond
    28: "v",  # Triangle down
}


def extract_metric_data(df, metric_name, x_column, strategy="greedy_entropy"):
    """Extract mean and standard error for a metric grouped by n_actions and x_value.

    Args:
        df: DataFrame containing results
        metric_name: Name of metric to extract
        x_column: Column name to use for x-axis grouping
        strategy: Selection strategy to filter for

    Returns:
        dict: {n_actions: (x_values, means, standard_errors)}
    """
    df_strategy = df[df["selection_strategy"] == strategy].copy()

    if df_strategy.empty:
        log.error(f"No data found for strategy: {strategy}")
        return {}

    n_actions_values = sorted(df_strategy["n_actions"].dropna().unique())

    results = {}
    for n_actions in n_actions_values:
        df_subset = df_strategy[df_strategy["n_actions"] == n_actions]
        x_values_unique = sorted(df_subset[x_column].dropna().unique())

        means = []
        standard_errors = []
        x_used = []

        for x_val in x_values_unique:
            df_n = df_subset[df_subset[x_column] == x_val]

            if len(df_n) == 0:
                continue

            values = df_n[metric_name].values
            values = values[~np.isnan(values)]

            if len(values) > 0:
                mean = np.mean(values)
                # Standard error = std / sqrt(n)
                std = np.std(values, ddof=1)  # Use sample std with ddof=1
                stderr = std / np.sqrt(len(values))

                means.append(mean)
                standard_errors.append(stderr)
                x_used.append(x_val)

        if len(means) > 0:
            results[n_actions] = (x_used, means, standard_errors)

    return results


def plot_combined_sweeps(
    omega_df: pd.DataFrame,
    n_frames_df: pd.DataFrame,
    save_path: str,
    strategy: str = "greedy_entropy",
    context="styles/ieee-tmi.mplstyle",
):
    """Create a 2x2 grid plot with LPIPS and PSNR for omega and n_frames sweeps.

    Args:
        omega_df: DataFrame with omega sweep results
        n_frames_df: DataFrame with n_frames sweep results
        save_path: Path to save the figure
        strategy: Selection strategy to plot
        context: Matplotlib style context
    """
    with plt.style.context(context):
        # Create 2x2 subplot grid - single column width
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(3.5, 2.5),  # Single IEEE column width
            sharex="col",
            sharey="row",
        )

        # Extract data for all combinations
        lpips_n_frames = extract_metric_data(n_frames_df, "lpips", "n_frames", strategy)
        lpips_omega = extract_metric_data(omega_df, "lpips", "x_value", strategy)
        psnr_n_frames = extract_metric_data(n_frames_df, "psnr", "n_frames", strategy)
        psnr_omega = extract_metric_data(omega_df, "psnr", "x_value", strategy)

        # Plot LPIPS vs n_frames (top-left)
        ax = axes[0, 0]
        for n_actions, (x_vals, means, stds) in lpips_n_frames.items():
            color = N_ACTIONS_COLORS.get(int(n_actions), "#000000")
            marker = N_ACTIONS_MARKERS.get(int(n_actions), "o")
            means_arr = np.array(means)
            stds_arr = np.array(stds)

            ax.plot(
                x_vals,
                means,
                marker=marker,
                linewidth=1,
                markersize=3,
                color=color,
                label=f"{int(n_actions)} lines",
            )
            ax.fill_between(
                x_vals,
                means_arr - stds_arr,
                means_arr + stds_arr,
                alpha=0.2,
                color=color,
            )

        ax.set_ylabel(METRIC_NAMES["lpips"])
        ax.grid(True, alpha=0.3)

        # Plot LPIPS vs omega (top-right)
        ax = axes[0, 1]
        for n_actions, (x_vals, means, stds) in lpips_omega.items():
            color = N_ACTIONS_COLORS.get(int(n_actions), "#000000")
            marker = N_ACTIONS_MARKERS.get(int(n_actions), "o")
            means_arr = np.array(means)
            stds_arr = np.array(stds)

            ax.plot(
                x_vals,
                means,
                marker=marker,
                linewidth=1,
                markersize=3,
                color=color,
                label=f"{int(n_actions)} lines",
            )
            ax.fill_between(
                x_vals,
                means_arr - stds_arr,
                means_arr + stds_arr,
                alpha=0.2,
                color=color,
            )

        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

        # Plot PSNR vs n_frames (bottom-left)
        ax = axes[1, 0]
        for n_actions, (x_vals, means, stds) in psnr_n_frames.items():
            color = N_ACTIONS_COLORS.get(int(n_actions), "#000000")
            marker = N_ACTIONS_MARKERS.get(int(n_actions), "o")
            means_arr = np.array(means)
            stds_arr = np.array(stds)

            ax.plot(
                x_vals, means, marker=marker, linewidth=1, markersize=3, color=color
            )
            ax.fill_between(
                x_vals,
                means_arr - stds_arr,
                means_arr + stds_arr,
                alpha=0.2,
                color=color,
            )

        ax.set_xlabel(r"$W$")
        ax.set_ylabel(METRIC_NAMES["psnr"])
        ax.grid(True, alpha=0.3)

        # Plot PSNR vs omega (bottom-right)
        ax = axes[1, 1]
        for n_actions, (x_vals, means, stds) in psnr_omega.items():
            color = N_ACTIONS_COLORS.get(int(n_actions), "#000000")
            marker = N_ACTIONS_MARKERS.get(int(n_actions), "o")
            means_arr = np.array(means)
            stds_arr = np.array(stds)

            ax.plot(
                x_vals, means, marker=marker, linewidth=1, markersize=3, color=color
            )
            ax.fill_between(
                x_vals,
                means_arr - stds_arr,
                means_arr + stds_arr,
                alpha=0.2,
                color=color,
            )

        ax.set_xlabel(r"$\gamma$")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

        # Format x-axis for omega plots
        for ax in [axes[0, 1], axes[1, 1]]:
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.get_xaxis().set_minor_formatter(plt.NullFormatter())

        # Add horizontal legend above the plots
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(handles),
            frameon=False,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save
        for ext in [".pdf", ".png"]:
            save_file = save_path.replace(".pdf", ext).replace(".png", ext)
            plt.savefig(save_file, dpi=300, bbox_inches="tight")
            log.info(f"Saved combined sweep plot to {log.yellow(save_file)}")

        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--omega-sweep-path",
        type=str,
        required=True,
        help="Path to the omega sweep directory",
    )
    parser.add_argument(
        "--n-frames-sweep-path",
        type=str,
        required=True,
        help="Path to the n_frames sweep directory",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./output",
        help="Directory to save the plots",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="greedy_entropy",
        choices=["greedy_entropy", "uniform_random", "equispaced"],
        help="Selection strategy to plot",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=None,
        help="Use metrics from only the first N frames",
    )
    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load omega sweep data
    log.info(f"Loading omega sweep from {args.omega_sweep_path}")
    omega_results = extract_sweep_data(
        [args.omega_sweep_path],
        keys_to_extract=["psnr", "lpips"],
        x_axis_key="diffusion_inference.guidance_kwargs.omega",
        config_keys_to_include=["action_selection.n_actions"],
        n_frames=args.n_frames,
    )
    log.info(f"Loaded {len(omega_results)} omega sweep results")

    # Load n_frames sweep data
    log.info(f"Loading n_frames sweep from {args.n_frames_sweep_path}")
    n_frames_results = extract_sweep_data(
        [args.n_frames_sweep_path],
        keys_to_extract=["psnr", "lpips"],
        x_axis_key="diffusion_inference.batch_size",  # Placeholder
        config_keys_to_include=["action_selection.n_actions"],
        n_frames=args.n_frames,
    )
    log.info(f"Loaded {len(n_frames_results)} n_frames sweep results")

    # Add n_frames column by looking up diffusion configs
    log.info("Extracting n_frames from diffusion model configs...")
    n_frames_results = add_n_frames_column(n_frames_results)

    # Filter out rows where n_frames couldn't be extracted
    n_frames_results = n_frames_results[n_frames_results["n_frames"].notna()].copy()
    log.info(f"Using {len(n_frames_results)} results with valid n_frames")

    if len(n_frames_results) == 0:
        log.error("No valid n_frames results found!")
        sys.exit(1)

    # Create combined plot
    save_path = save_dir / f"combined_sweeps_{args.strategy}.pdf"
    plot_combined_sweeps(
        omega_df=omega_results,
        n_frames_df=n_frames_results,
        save_path=str(save_path),
        strategy=args.strategy,
        context="styles/ieee-tmi.mplstyle",
    )

    log.info("Done!")
