"""
Plot the relationship between number of particles and PSNR for different n_actions values.
"""

import argparse
import os

os.environ["KERAS_BACKEND"] = "numpy"
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from rich.console import Console
from rich.table import Table

from zea import Config, init_device, log
from zea.ops import translate

if __name__ == "__main__":
    init_device("cpu")
    sys.path.append("/ulsa")

from plotting.index import extract_sweep_data, index_sweep_data
from plotting.plot_utils import natural_sort

STRATEGY_NAMES = {
    "greedy_entropy": "Active Perception",
    "uniform_random": "Random",
    "equispaced": "Equispaced",
}

METRIC_NAMES = {
    "dice": "DICE (→) [-]",
    "psnr": "PSNR (→) [dB]",
    "ssim": "SSIM (→) [-]",
    "lpips": "LPIPS (←) [-]",
    "mse": "MSE (←) [-]",
    "rmse": "RMSE (←) [-]",
}

# Colors for different n_actions values
N_ACTIONS_COLORS = {
    2: "#1f77b4",  # Blue
    4: "#ff7f0e",  # Orange
    7: "#2ca02c",  # Green
    14: "#d62728",  # Red
    28: "#9467bd",  # Purple
}


def plot_n_particles_vs_metric(
    df: pd.DataFrame,
    metric_name: str,
    save_path: str,
    strategy: str = "greedy_entropy",
    context="styles/ieee-tmi.mplstyle",
):
    """Plot metric vs number of particles for different n_actions values.

    Args:
        df (pd.DataFrame): DataFrame containing the results.
        metric_name (str): Name of the metric to plot.
        save_path (str): Path to save the plot.
        strategy (str): Selection strategy to plot.
        context (str): Matplotlib style context.
    """
    # Filter for the specific strategy
    df_strategy = df[df["selection_strategy"] == strategy].copy()

    if df_strategy.empty:
        log.error(f"No data found for strategy: {strategy}")
        return

    # Get unique values - now n_actions is directly in the DataFrame
    n_particles_values = sorted(df_strategy["x_value"].unique())
    n_actions_values = sorted(df_strategy["n_actions"].dropna().unique())

    log.info(f"Found n_particles values: {n_particles_values}")
    log.info(f"Found n_actions values: {n_actions_values}")

    with plt.style.context(context):
        fig, ax = plt.subplots(figsize=(7.16, 4))

        for n_actions in n_actions_values:
            df_subset = df_strategy[df_strategy["n_actions"] == n_actions]

            means = []
            stds = []
            particles_used = []

            for n_particles in n_particles_values:
                df_n = df_subset[df_subset["x_value"] == n_particles]

                if len(df_n) == 0:
                    continue

                if metric_name.lower() == "rmse":
                    values = np.sqrt(df_n["mse"].values / (255 * 255))
                elif metric_name.lower() == "mse":
                    values = df_n["mse"].values / (255 * 255)
                else:
                    values = df_n[metric_name].values

                # Filter NaN values
                values = values[~np.isnan(values)]

                if len(values) > 0:
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                    particles_used.append(n_particles)

            if len(means) > 0:
                color = N_ACTIONS_COLORS.get(int(n_actions), "#000000")

                # Plot mean line
                ax.plot(
                    particles_used,
                    means,
                    marker="o",
                    linewidth=2,
                    markersize=6,
                    label=f"{int(n_actions)} scan lines",
                    color=color,
                )

                # Add standard deviation as shaded region
                means_array = np.array(means)
                stds_array = np.array(stds)
                ax.fill_between(
                    particles_used,
                    means_array - stds_array,
                    means_array + stds_array,
                    alpha=0.2,
                    color=color,
                )

        # Formatting
        ax.set_xlabel("Number of Particles", fontsize=11)
        formatted_metric_name = METRIC_NAMES.get(metric_name, metric_name.upper())
        ax.set_ylabel(formatted_metric_name, fontsize=11)

        strategy_display = STRATEGY_NAMES.get(strategy, strategy)

        # Move legend outside the plot on the right side
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9,
            framealpha=0.95,
            fancybox=True,
        )

        ax.grid(True, alpha=0.3)
        ax.set_xticks(n_particles_values)

        # Adjust layout to accommodate legend outside
        plt.tight_layout()
        plt.subplots_adjust(right=0.82)  # Leave space on the right for legend

        # Save plot with bbox_inches='tight' to include the legend
        for ext in [".pdf", ".png"]:
            save_file = save_path.replace(".pdf", ext).replace(".png", ext)
            plt.savefig(save_file, dpi=300, bbox_inches="tight")
            log.info(f"Saved plot to {log.yellow(save_file)}")

        plt.close()


def print_results_table(df: pd.DataFrame, metric_name: str, strategy: str):
    """Print results in a table format."""
    df_strategy = df[df["selection_strategy"] == strategy].copy()

    if df_strategy.empty:
        log.error(f"No data found for strategy: {strategy}")
        return

    table = Table(
        title=f"{metric_name.upper()} Results for {STRATEGY_NAMES.get(strategy, strategy)}",
        show_lines=True,
    )
    table.add_column("N Actions", style="cyan", no_wrap=True)
    table.add_column("N Particles", style="magenta")
    table.add_column("Mean", style="green")
    table.add_column("Std", style="yellow")
    table.add_column("Count", style="white")

    # Now n_actions is directly available in the DataFrame
    n_actions_values = sorted(df_strategy["n_actions"].dropna().unique())
    n_particles_values = sorted(df_strategy["x_value"].unique())

    for n_actions in n_actions_values:
        for n_particles in n_particles_values:
            df_subset = df_strategy[
                (df_strategy["n_actions"] == n_actions)
                & (df_strategy["x_value"] == n_particles)
            ]

            if len(df_subset) == 0:
                continue

            if metric_name.lower() == "rmse":
                values = np.sqrt(df_subset["mse"].values / (255 * 255))
            elif metric_name.lower() == "mse":
                values = df_subset["mse"].values / (255 * 255)
            else:
                values = df_subset[metric_name].values

            values = values[~np.isnan(values)]

            if len(values) > 0:
                mean = np.mean(values)
                std = np.std(values)
                count = len(values)

                table.add_row(
                    str(int(n_actions)),
                    str(int(n_particles)),
                    f"{mean:.3f}",
                    f"{std:.3f}",
                    str(count),
                )

    console = Console()
    console.print(table)


def plot_variance_maps_grid(
    sweep_path: str,
    save_dir: Path,
    n_runs_per_config: int = 1,
    n_frames: int = 10,
    strategy: str = "greedy_entropy",
    n_actions: int = 7,
    context="styles/ieee-tmi.mplstyle",
):
    """Plot a grid of variance maps for different numbers of particles.

    Args:
        sweep_path (str): Path to the sweep directory
        save_dir (Path): Directory to save the plots
        n_runs_per_config (int): Number of runs to plot per n_particles configuration
        n_frames (int): Number of frames to show per row
        strategy (str): Selection strategy to filter for
        n_actions (int): Number of actions to filter for
        context (str): Matplotlib style context
    """
    from matplotlib.gridspec import GridSpec

    log.info(f"Loading runs for variance map visualization...")

    # Index all runs
    generator = index_sweep_data([sweep_path])

    # Group runs by n_particles
    runs_by_n_particles = {}

    for run_path, target_file, filename in tqdm.tqdm(generator, desc="Indexing runs"):
        config_path = run_path / "config.yaml"
        metrics_path = run_path / "metrics.npz"

        if not config_path.exists() or not metrics_path.exists():
            continue

        try:
            config = Config.from_yaml(str(config_path))

            # Filter by strategy and n_actions
            if config.get("action_selection", {}).get("selection_strategy") != strategy:
                continue

            run_n_actions = config.get("action_selection", {}).get("n_actions")
            if run_n_actions != n_actions:
                continue

            # Get n_particles from config
            n_particles = config.get("diffusion_inference", {}).get("batch_size")

            if n_particles is None:
                continue

            # Check if metrics has belief_distributions
            metrics = np.load(str(metrics_path), allow_pickle=True)
            if "belief_distributions" not in metrics:
                continue

            if n_particles not in runs_by_n_particles:
                runs_by_n_particles[n_particles] = []

            runs_by_n_particles[n_particles].append(
                {
                    "run_path": run_path,
                    "metrics_path": metrics_path,
                    "filename": filename,
                    "config": config,
                }
            )

        except Exception as e:
            log.warning(f"Error processing {run_path}: {e}")
            continue

    if not runs_by_n_particles:
        log.error("No runs found matching criteria")
        return

    log.info(f"Found runs for n_particles values: {sorted(runs_by_n_particles.keys())}")

    # Plot for each run
    n_particles_values = sorted(runs_by_n_particles.keys())

    for run_idx in range(n_runs_per_config):
        # Check if we have enough runs for this index
        available_n_particles = [
            n_p for n_p in n_particles_values if len(runs_by_n_particles[n_p]) > run_idx
        ]

        if not available_n_particles:
            log.warning(f"Not enough runs for run_idx={run_idx}")
            break

        log.info(f"Creating variance map grid for run {run_idx + 1}...")

        with plt.style.context(context):
            # Create figure with gridspec
            n_rows = len(available_n_particles)
            n_cols = n_frames

            fig = plt.figure(figsize=(n_frames * 1.5, n_rows * 1.5))
            gs = GridSpec(
                n_rows,
                n_cols + 1,
                figure=fig,
                width_ratios=[1] * n_cols + [0.05],
                hspace=0.1,
                wspace=0.05,
            )

            # Track vmin/vmax across all variance maps for consistent colormap
            all_variances = []
            variance_data = []

            # First pass: load all data and compute global vmin/vmax
            for row_idx, n_particles in enumerate(available_n_particles):
                run_info = runs_by_n_particles[n_particles][run_idx]

                try:
                    metrics = np.load(str(run_info["metrics_path"]), allow_pickle=True)
                    belief_dist = metrics["belief_distributions"]

                    # Compute variance: (n_frames, h, w, c)
                    variance_maps = np.var(belief_dist, axis=1)

                    # Take only first n_frames
                    variance_maps = variance_maps[:n_frames]

                    variance_data.append(
                        (n_particles, variance_maps, run_info["filename"])
                    )
                    all_variances.extend(variance_maps.flatten())

                except Exception as e:
                    log.error(f"Error loading variance for {run_info['run_path']}: {e}")
                    continue

            if not variance_data:
                log.error("No variance data could be loaded")
                plt.close(fig)
                continue

            # Compute global vmin/vmax
            vmin = np.percentile(all_variances, 1)
            vmax = np.percentile(all_variances, 99.9)
            # vmin = np.min(all_variances)
            # vmax = np.max(all_variances)

            # Second pass: plot the data
            for row_idx, (n_particles, variance_maps, filename) in enumerate(
                variance_data
            ):
                actual_n_frames = min(len(variance_maps), n_frames)

                for col_idx in range(actual_n_frames):
                    ax = fig.add_subplot(gs[row_idx, col_idx])

                    # Get variance map for this frame
                    var_map = variance_maps[col_idx]

                    # Remove channel dimension if present
                    if var_map.ndim == 3 and var_map.shape[-1] == 1:
                        var_map = var_map[..., 0]

                    # Plot variance map
                    im = ax.imshow(var_map, cmap="viridis", vmin=vmin, vmax=vmax)
                    ax.axis("off")

                    # Add frame number at top of first row
                    if row_idx == 0:
                        ax.set_title(f"Frame {col_idx}", fontsize=8, pad=2)

                    # Add n_particles label on leftmost column
                    if col_idx == 0:
                        ax.text(
                            -0.1,
                            0.5,
                            f"$N_p$={n_particles}",
                            transform=ax.transAxes,
                            fontsize=9,
                            rotation=90,
                            verticalalignment="center",
                            horizontalalignment="right",
                        )

                # Fill remaining columns if we have fewer frames
                for col_idx in range(actual_n_frames, n_frames):
                    ax = fig.add_subplot(gs[row_idx, col_idx])
                    ax.axis("off")

            # Add colorbar
            cbar_ax = fig.add_subplot(gs[:, -1])
            cbar = plt.colorbar(im, cax=cbar_ax)
            cbar.set_label("Variance", fontsize=9)

            # Save plot
            patient_name = variance_data[0][2].replace(".avi", "").replace(".mp4", "")
            for ext in [".pdf", ".png"]:
                save_file = (
                    save_dir
                    / f"variance_maps_grid__n_actions={n_actions}__run{run_idx + 1}_{patient_name}{ext}"
                )
                plt.savefig(save_file, dpi=300, bbox_inches="tight")
                log.info(f"Saved variance map grid to {log.yellow(save_file)}")

            plt.close(fig)


def plot_linewise_entropy_curves(
    sweep_path: str,
    save_dir: Path,
    strategy: str = "greedy_entropy",
    n_actions: int = 7,
    entropy_sigma: float = 1.0,
    context="styles/ieee-tmi.mplstyle",
    seed: int = 0,
):
    """Plot linewise entropy curves for different numbers of particles.

    Args:
        sweep_path (str): Path to the sweep directory
        save_dir (Path): Directory to save the plots
        strategy (str): Selection strategy to filter for
        n_actions (int): Number of actions to filter for
        entropy_sigma (float): Sigma parameter for entropy computation
        context (str): Matplotlib style context
    """
    # Import GreedyEntropy class
    import random
    import sys

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    sys.path.append("/ulsa/zea")
    from zea.agent.selection import GreedyEntropy

    # Set random seed for reproducibility
    random.seed(seed)

    log.info(f"Loading runs for linewise entropy visualization...")

    # Index all runs
    generator = index_sweep_data([sweep_path])

    # Group runs by n_particles
    runs_by_n_particles = {}

    for run_path, target_file, filename in tqdm.tqdm(generator, desc="Indexing runs"):
        config_path = run_path / "config.yaml"
        metrics_path = run_path / "metrics.npz"

        if not config_path.exists() or not metrics_path.exists():
            continue

        try:
            config = Config.from_yaml(str(config_path))

            # Filter by strategy and n_actions
            if config.get("action_selection", {}).get("selection_strategy") != strategy:
                continue

            run_n_actions = config.get("action_selection", {}).get("n_actions")
            if run_n_actions != n_actions:
                continue

            # Get n_particles from config
            n_particles = config.get("diffusion_inference", {}).get("batch_size")

            if n_particles is None:
                continue

            # Check if metrics has belief_distributions
            metrics = np.load(str(metrics_path), allow_pickle=True)
            if "belief_distributions" not in metrics:
                continue

            if n_particles not in runs_by_n_particles:
                runs_by_n_particles[n_particles] = []

            runs_by_n_particles[n_particles].append(
                {
                    "run_path": run_path,
                    "metrics_path": metrics_path,
                    "filename": filename,
                    "config": config,
                }
            )

        except Exception as e:
            log.warning(f"Error processing {run_path}: {e}")
            continue

    if not runs_by_n_particles:
        log.error("No runs found matching criteria")
        return

    log.info(f"Found runs for n_particles values: {sorted(runs_by_n_particles.keys())}")

    # Get all n_particles values
    n_particles_values = sorted(runs_by_n_particles.keys())

    # Find sequences that exist for all n_particles values
    # Group by filename across all n_particles
    sequences_by_filename = {}
    for n_particles in n_particles_values:
        for run_info in runs_by_n_particles[n_particles]:
            filename = run_info["filename"]
            if filename not in sequences_by_filename:
                sequences_by_filename[filename] = {}
            sequences_by_filename[filename][n_particles] = run_info

    # Filter to only sequences that have all n_particles values
    complete_sequences = {
        filename: runs
        for filename, runs in sequences_by_filename.items()
        if len(runs) == len(n_particles_values)
    }

    if not complete_sequences:
        log.error("No sequences found with all n_particles values")
        return

    log.info(f"Found {len(complete_sequences)} complete sequences")

    # Randomly select sequences
    selected_filenames = random.sample(list(complete_sequences.keys()), 2)

    log.info(f"Selected sequences: {selected_filenames}")

    with plt.style.context(context):
        # Create figure with 1 row, 2 columns + space for colorbar
        fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.25), sharey=True)

        # Setup colormap for n_particles
        norm = Normalize(vmin=min(n_particles_values), vmax=max(n_particles_values))
        cmap = plt.cm.viridis
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        for seq_idx, filename in enumerate(selected_filenames):
            ax = axes[seq_idx]

            # Collect entropy data for all n_particles for this sequence
            entropy_curves = {}

            for n_particles in n_particles_values:
                run_info = complete_sequences[filename][n_particles]

                try:
                    metrics = np.load(str(run_info["metrics_path"]), allow_pickle=True)
                    belief_dist = metrics["belief_distributions"]

                    # Get config info
                    config = run_info["config"]
                    n_possible_actions = config.get("action_selection", {}).get(
                        "n_possible_actions", 112
                    )
                    img_height = belief_dist.shape[2]
                    img_width = belief_dist.shape[3]

                    # Initialize GreedyEntropy class with proper parameters
                    entropy_model = GreedyEntropy(
                        n_actions=n_actions,
                        n_possible_actions=n_possible_actions,
                        img_width=img_width,
                        img_height=img_height,
                        entropy_sigma=entropy_sigma,
                    )

                    # Use only the first frame
                    particles = belief_dist[0:1]  # Keep batch dim

                    # Use GreedyEntropy's compute_pixelwise_entropy method
                    pixelwise_entropy = entropy_model.compute_pixelwise_entropy(
                        translate(particles, (0, 255), (-1, 1))[..., 0]
                    )

                    # Sum over height to get linewise entropy
                    linewise_entropy = np.sum(
                        pixelwise_entropy, axis=1
                    )  # Shape: (batch, width)
                    entropy_curves[n_particles] = linewise_entropy[
                        0
                    ]  # Remove batch dim

                except Exception as e:
                    log.error(f"Error loading entropy for {run_info['run_path']}: {e}")
                    import traceback

                    traceback.print_exc()
                    continue

            if not entropy_curves:
                log.error(f"No entropy data could be loaded for sequence {filename}")
                ax.axis("off")
                continue

            # Plot all curves on the same axes with low alpha
            for n_particles in n_particles_values:
                if n_particles not in entropy_curves:
                    continue

                entropy = entropy_curves[n_particles]
                x_vals = np.arange(len(entropy))
                color = cmap(norm(n_particles))

                ax.plot(
                    x_vals,
                    entropy,
                    linewidth=1.5,
                    alpha=0.6,
                    color=color,
                    marker="",
                    linestyle="-",
                )

            # Formatting
            ax.set_title(f"Sequence {seq_idx + 1}", fontsize=8)
            ax.grid(True, alpha=0.3)

        # Set y-label only on the leftmost plot
        axes[0].set_ylabel("Entropy $\hat{H}^\ell$")

        # Add shared x-label at the bottom
        fig.text(0.5, 0.02, "Line Index $\ell$", ha="center")

        # Adjust layout before adding colorbar
        plt.tight_layout(rect=[0, 0, 0.92, 1])  # Leave space on right for colorbar
        # Create discrete colormap
        from matplotlib.colors import BoundaryNorm

        # Create boundaries between n_particles values
        boundaries = []
        for i in range(len(n_particles_values)):
            if i == 0:
                boundaries.append(n_particles_values[i] - 0.5)
            boundaries.append(n_particles_values[i] + 0.5)

        # Create discrete norm and colormap
        norm_discrete = BoundaryNorm(boundaries, cmap.N)
        sm_discrete = ScalarMappable(norm=norm_discrete, cmap=cmap)
        sm_discrete.set_array([])

        # Add shared colorbar with discrete blocks
        cbar = fig.colorbar(
            sm_discrete, ax=axes, orientation="vertical", pad=0.02, aspect=15
        )
        cbar.set_label("$N_p$")
        cbar.set_ticks(n_particles_values)  # Set ticks to actual n_particles values
        cbar.set_ticklabels([str(int(n)) for n in n_particles_values])  # Integer labels
        cbar.ax.minorticks_off()

        # plt.tight_layout()

        # Save plot
        for ext in [".pdf", ".png"]:
            save_file = (
                save_dir / f"linewise_entropy_comparison__n_actions={n_actions}{ext}"
            )
            plt.savefig(save_file, dpi=300, bbox_inches="tight")
            log.info(f"Saved linewise entropy comparison to {log.yellow(save_file)}")

        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-path",
        type=str,
        default="/mnt/z/Ultrasound-BMd/data/oisin/ULSA_hyperparam_sweeps/n_particles_choose_first/sweep_2025_11_04_100001_597713",
        help="Path to the sweep directory",
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
        "--n_frames",
        type=int,
        default=None,
        help="Use metrics on from the first N frames",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["psnr", "lpips", "mse"],
        help="Metrics to plot",
    )
    parser.add_argument(
        "--plot-variance-grids",
        action="store_true",
        help="Generate variance map grids",
    )
    parser.add_argument(
        "--variance-n-runs",
        type=int,
        default=3,
        help="Number of runs to plot variance maps for",
    )
    parser.add_argument(
        "--variance-n-frames",
        type=int,
        default=10,
        help="Number of frames to show in variance grids",
    )
    parser.add_argument(
        "--variance-n-actions",
        type=int,
        default=7,
        help="Number of actions to filter for in variance grids",
    )
    parser.add_argument(
        "--plot-linewise-entropy",
        action="store_true",
        help="Generate linewise entropy curve plots",
    )
    parser.add_argument(
        "--entropy-n-runs",
        type=int,
        default=3,
        help="Number of runs to plot linewise entropy for",
    )
    parser.add_argument(
        "--entropy-n-frames",
        type=int,
        default=10,
        help="Number of frames to show in linewise entropy plots",
    )
    parser.add_argument(
        "--entropy-n-actions",
        type=int,
        default=7,
        help="Number of actions to filter for in linewise entropy plots",
    )
    parser.add_argument(
        "--entropy-sigma",
        type=float,
        default=1.0,
        help="Sigma parameter for entropy computation (default: 1.0)",
    )
    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot linewise entropy curves if requested
    if args.plot_linewise_entropy:
        log.info("Generating linewise entropy curves...")
        plot_linewise_entropy_curves(
            sweep_path=args.sweep_path,
            save_dir=save_dir,
            strategy=args.strategy,
            n_actions=args.variance_n_actions,
            entropy_sigma=args.entropy_sigma,
            context="styles/ieee-tmi.mplstyle",
        )

    # Plot variance grids if requested
    if args.plot_variance_grids:
        log.info("Generating variance map grids...")
        plot_variance_maps_grid(
            sweep_path=args.sweep_path,
            save_dir=save_dir,
            n_runs_per_config=args.variance_n_runs,
            n_frames=args.variance_n_frames,
            strategy=args.strategy,
            n_actions=args.variance_n_actions,
            context="styles/ieee-tmi.mplstyle",
        )

    # Original plotting code
    log.info(f"Loading results from {args.sweep_path}")

    # Extract sweep data
    combined_results = extract_sweep_data(
        [args.sweep_path],
        keys_to_extract=args.metrics,
        x_axis_key="diffusion_inference.batch_size",  # This is n_particles
        config_keys_to_include=["action_selection.n_actions"],
        n_frames=args.n_frames,
    )

    log.info(f"Loaded {len(combined_results)} results")

    # Plot for each metric
    for metric_name in args.metrics:
        log.info(f"Plotting {metric_name} vs n_particles")

        save_path = save_dir / f"n_particles_vs_{metric_name}_{args.strategy}.pdf"

        plot_n_particles_vs_metric(
            combined_results,
            metric_name=metric_name,
            save_path=str(save_path),
            strategy=args.strategy,
            context="styles/ieee-tmi.mplstyle",
        )

        # Print results table
        print_results_table(combined_results, metric_name, args.strategy)

    log.info("Done!")
