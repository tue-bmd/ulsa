"""Script to evaluate image quality of in-house cardiac ultrasound data."""

import os

os.environ["KERAS_BACKEND"] = "jax"

# to fix a bug with zea caching (should be fixed in https://github.com/tue-bmd/zea/pull/236)
os.environ["ZEA_DISABLE_CACHE"] = "1"

import zea

zea.init_device("cpu")
import sys
from pathlib import Path

sys.path.append("/ulsa")

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting.index import extract_sweep_data
from plotting.plot_utils import METRIC_NAMES

DATA_ROOT = Path("/ulsa/output/eval_in_house/image_quality")
SUBSAMPLED_PATHS = [
    DATA_ROOT / "sweep_2026_01_15_141144_039279",
    DATA_ROOT / "sweep_2026_01_15_161530_443066",
]

STRATEGY_COLORS = {
    "greedy_entropy": "#1f77b4",
    "equispaced": "#2ca02c",
    "uniform_random": "#ff7f0e",
}

STRATEGY_NAMES = {
    "greedy_entropy": "Cognitive",
    "uniform_random": "Random",
    "equispaced": "Equispaced",
}

STRATEGIES_TO_PLOT = [
    "greedy_entropy",
    "uniform_random",
    "equispaced",
]

combined_results = extract_sweep_data(
    SUBSAMPLED_PATHS,
    keys_to_extract=["mse", "psnr", "dice", "lpips", "ssim"],
    x_axis_key="action_selection.n_actions",
)

rng = np.random.default_rng(42)

with plt.style.context("styles/ieee-tmi.mplstyle"):
    # Create paired dot plot with facets for different x_values
    fig, axes = plt.subplots(1, 3, sharey=True)

    # Get unique x_values and sort them
    x_values = sorted(combined_results["x_value"].unique())

    # Get unique filestems to connect with lines
    filestems = combined_results["filestem"].unique()

    for idx, x_val in enumerate(x_values):
        ax = axes[idx]

        # Filter data for this x_value
        data_subset = combined_results[combined_results["x_value"] == x_val]

        # Plot points for each strategy
        for i, strategy in enumerate(STRATEGIES_TO_PLOT):
            strategy_data = data_subset[data_subset["selection_strategy"] == strategy]

            # Add jitter to x position for visibility
            jitter = rng.normal(0, 0.02, len(strategy_data))
            x_pos = i + jitter

            ax.scatter(
                x_pos,
                strategy_data["psnr"],
                color=STRATEGY_COLORS[strategy],
                alpha=0.6,
                label=STRATEGY_NAMES[strategy] if idx == 0 else None,
            )

        # Draw lines connecting the same filestem across strategies
        for filestem in filestems:
            filestem_data = data_subset[data_subset["filestem"] == filestem]

            # Sort by strategy order
            filestem_data = filestem_data.set_index("selection_strategy")
            filestem_data = filestem_data.reindex(STRATEGIES_TO_PLOT)

            is_harmonic = "harmonic" in str(filestem_data["filepath"])

            if len(filestem_data) == len(STRATEGIES_TO_PLOT):
                x_positions = list(range(len(STRATEGIES_TO_PLOT)))
                y_positions = filestem_data["psnr"].values
                ax.plot(
                    x_positions,
                    y_positions,
                    "k-" if is_harmonic else "k--",
                    alpha=0.2,
                    zorder=0,
                )

        # Formatting
        ax.set_xlabel(f"{int(x_val)}")
        ax.set_xticks([])
        ax.grid(
            True,
            # alpha=0.3,
            axis="y",
        )

        if idx == 0:
            ax.set_ylabel(METRIC_NAMES["psnr"])

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(
        h,
        l,
        loc="outside upper center",
        ncol=3,
        frameon=False,
    )
    fig.supxlabel("# Scan Lines")

    plt.savefig(DATA_ROOT / "paired_dot_plot_psnr.png", dpi=300)
    plt.show()

    print(f"Plot saved to {DATA_ROOT / 'paired_dot_plot_psnr.png'}")
