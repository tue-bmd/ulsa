"""
Compare 'choose first' vs 'posterior mean' strategies as a function of N_p (number of particles).

Plots PSNR and LPIPS as line graphs with shared x-axis for N_p, with curves for both strategies.
"""

import argparse
import os
import sys

os.environ["MPLBACKEND"] = "Agg"
os.environ["KERAS_BACKEND"] = "numpy"

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from zea import init_device, log

if __name__ == "__main__":
    init_device("cpu")
    sys.path.append("/ulsa")

from ulsa.plotting.index import extract_sweep_data
from ulsa.plotting.plot_utils import METRIC_NAMES

STRATEGY_STYLES = {
    "choose_first": {
        "color": "#1f77b4",
        "marker": "o",
        "label": "Choose First",
    },
    "posterior_mean": {
        "color": "#d62728",
        "marker": "s",
        "label": "Posterior Mean",
    },
}


def extract_metric_vs_n_particles(df, metric_name, n_actions=7):
    """Extract mean and stderr for a metric grouped by N_p, filtered to a specific n_actions.

    Args:
        df: DataFrame from extract_sweep_data (x_value = N_p, n_actions column present)
        metric_name: Name of metric to extract
        n_actions: Number of scan lines to filter for

    Returns:
        (x_values, means, standard_errors) or None if no data
    """
    # Filter to greedy_entropy and the desired n_actions
    mask = df["selection_strategy"] == "greedy_entropy"
    if "n_actions" in df.columns:
        mask &= df["n_actions"] == n_actions
    df_filtered = df[mask].copy()

    if df_filtered.empty:
        return None

    n_particles_values = sorted(df_filtered["x_value"].unique())

    means = []
    stderrs = []
    x_used = []

    for n_p in n_particles_values:
        values = df_filtered[df_filtered["x_value"] == n_p][metric_name].values
        values = values[~np.isnan(values)]
        if len(values) > 0:
            mean = np.mean(values)
            stderr = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            means.append(mean)
            stderrs.append(stderr)
            x_used.append(n_p)

    if not means:
        return None

    return np.array(x_used), np.array(means), np.array(stderrs)


def plot_choose_first_vs_posterior_mean(
    choose_first_df,
    posterior_mean_df,
    save_path,
    n_actions=7,
    context="styles/ieee-tmi.mplstyle",
):
    """Create a 2x1 stacked plot comparing choose_first and posterior_mean strategies.

    Args:
        choose_first_df: DataFrame with choose_first sweep results
        posterior_mean_df: DataFrame with posterior_mean sweep results
        save_path: Path to save the figure
        n_actions: Number of scan lines to filter for
        context: Matplotlib style context
    """
    with plt.style.context(context):
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.5), sharex=True)

        datasets = {
            "choose_first": choose_first_df,
            "posterior_mean": posterior_mean_df,
        }

        # Plot LPIPS (top)
        ax = axes[0]
        for key, df in datasets.items():
            result = extract_metric_vs_n_particles(df, "lpips", n_actions=n_actions)
            if result is None:
                log.warning(f"No LPIPS data for {key}")
                continue
            x_vals, means, stderrs = result
            style = STRATEGY_STYLES[key]
            ax.plot(
                x_vals, means,
                marker=style["marker"],
                color=style["color"],
                linewidth=1,
                markersize=3,
                label=style["label"],
            )
            ax.fill_between(
                x_vals,
                means - stderrs,
                means + stderrs,
                alpha=0.2,
                color=style["color"],
            )
        ax.set_ylabel(METRIC_NAMES.get("lpips", "LPIPS"))
        ax.grid(True, alpha=0.3)

        # Plot PSNR (bottom)
        ax = axes[1]
        for key, df in datasets.items():
            result = extract_metric_vs_n_particles(df, "psnr", n_actions=n_actions)
            if result is None:
                log.warning(f"No PSNR data for {key}")
                continue
            x_vals, means, stderrs = result
            style = STRATEGY_STYLES[key]
            ax.plot(
                x_vals, means,
                marker=style["marker"],
                color=style["color"],
                linewidth=1,
                markersize=3,
                label=style["label"],
            )
            ax.fill_between(
                x_vals,
                means - stderrs,
                means + stderrs,
                alpha=0.2,
                color=style["color"],
            )
        ax.set_xlabel("$N_p$")
        ax.set_ylabel(METRIC_NAMES.get("psnr", "PSNR"))
        ax.grid(True, alpha=0.3)

        # Shared legend above plots
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(handles),
            frameon=False,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        for ext in [".pdf", ".png"]:
            save_file = str(save_path).replace(".pdf", ext).replace(".png", ext)
            plt.savefig(save_file, dpi=300, bbox_inches="tight")
            log.info(f"Saved plot to {log.yellow(save_file)}")

        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare choose_first vs posterior_mean as a function of N_p."
    )
    parser.add_argument(
        "--choose-first-sweep",
        type=str,
        default="/mnt/z/usbmd/ulsa/hyperparam_sweeps/n_particles_choose_first/sweep_2025_11_04_100001_597713",
        help="Path to the choose_first sweep directory",
    )
    parser.add_argument(
        "--posterior-mean-sweep",
        type=str,
        default="/mnt/z/usbmd/ulsa/hyperparam_sweeps/n_particles_posterior_mean/sweep_2025_11_07_141136_956329",
        help="Path to the posterior_mean sweep directory",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./output",
        help="Directory to save the plots",
    )
    parser.add_argument(
        "--n-actions",
        type=int,
        default=7,
        help="Number of scan lines to filter for",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=None,
        help="Use metrics from only the first N frames",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load choose_first sweep
    log.info(f"Loading choose_first sweep from {args.choose_first_sweep}")
    choose_first_df = extract_sweep_data(
        [args.choose_first_sweep],
        keys_to_extract=["psnr", "lpips"],
        x_axis_key="diffusion_inference.batch_size",
        config_keys_to_include=["action_selection.n_actions"],
        n_frames=args.n_frames,
    )
    log.info(f"Loaded {len(choose_first_df)} choose_first results")

    # Load posterior_mean sweep
    log.info(f"Loading posterior_mean sweep from {args.posterior_mean_sweep}")
    posterior_mean_df = extract_sweep_data(
        [args.posterior_mean_sweep],
        keys_to_extract=["psnr", "lpips"],
        x_axis_key="diffusion_inference.batch_size",
        config_keys_to_include=["action_selection.n_actions"],
        n_frames=args.n_frames,
    )
    log.info(f"Loaded {len(posterior_mean_df)} posterior_mean results")

    # Plot
    save_path = save_dir / "choose_first_vs_posterior_mean.pdf"
    plot_choose_first_vs_posterior_mean(
        choose_first_df,
        posterior_mean_df,
        save_path=save_path,
        n_actions=args.n_actions,
        context="styles/ieee-tmi.mplstyle",
    )

    log.info("Done!")
