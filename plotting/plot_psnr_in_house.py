"""Script to evaluate image quality of in-house cardiac ultrasound data."""

import os

os.environ["KERAS_BACKEND"] = "jax"

# to fix a bug with zea caching (should be fixed in https://github.com/tue-bmd/zea/pull/236)
os.environ["ZEA_DISABLE_CACHE"] = "1"

import zea

zea.init_device("cpu")
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from ulsa.plotting.index import extract_sweep_data
from ulsa.plotting.plot_utils import METRIC_NAMES, STRATEGY_COLORS, STRATEGY_NAMES


def compute_paired_wilcoxon(
    combined_results, metrics, cognitive_strategy, baseline_strategies
):
    """Compute paired Wilcoxon signed-rank tests between cognitive and baselines.

    Returns:
        dict: {(metric, baseline, x_val): (stat, p_value, n_pairs, cog_mean, base_mean)}
    """
    results = {}
    cog_df = combined_results[
        combined_results["selection_strategy"] == cognitive_strategy
    ].copy()

    for metric_name in metrics:
        cog_df[metric_name + "_mean"] = cog_df[metric_name].apply(
            lambda v: np.mean(v) if hasattr(v, "__len__") else v
        )
        for baseline in baseline_strategies:
            base_df = combined_results[
                combined_results["selection_strategy"] == baseline
            ].copy()
            base_df[metric_name + "_mean"] = base_df[metric_name].apply(
                lambda v: np.mean(v) if hasattr(v, "__len__") else v
            )
            x_values_available = sorted(
                set(cog_df["x_value"].unique()) & set(base_df["x_value"].unique())
            )
            for x_val in x_values_available:
                cog_subset = cog_df[cog_df["x_value"] == x_val][
                    ["filestem", metric_name + "_mean"]
                ].dropna()
                base_subset = base_df[base_df["x_value"] == x_val][
                    ["filestem", metric_name + "_mean"]
                ].dropna()
                merged = pd.merge(
                    cog_subset,
                    base_subset,
                    on="filestem",
                    suffixes=("_cog", "_base"),
                )
                if len(merged) < 2:
                    continue
                cog_values = merged[f"{metric_name}_mean_cog"].values
                base_values = merged[f"{metric_name}_mean_base"].values
                stat, p_value = wilcoxon(cog_values, base_values)
                results[(metric_name, baseline, x_val)] = (
                    stat,
                    p_value,
                    len(merged),
                    np.mean(cog_values),
                    np.mean(base_values),
                )
    return results


DATA_ROOT = Path("/ulsa/output/eval_in_house/image_quality")
SUBSAMPLED_PATHS = [
    DATA_ROOT / "sweep_2026_01_20_103617_291774",
    DATA_ROOT / "sweep_2026_01_20_113307_398439",
]


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
metrics = ["lpips", "psnr"]


# add RMSE
combined_results["rmse"] = np.sqrt(combined_results["mse"] / (255**2))


wilcoxon_results = compute_paired_wilcoxon(
    combined_results, metrics, "greedy_entropy", ["uniform_random", "equispaced"]
)
print("Wilcoxon results:")
for (metric_name, baseline, x_val), (
    stat,
    p_value,
    n_pairs,
    cog_mean,
    base_mean,
) in wilcoxon_results.items():
    print(
        f"{metric_name} - {baseline} at x={x_val}: stat={stat:.2f}, p={p_value:.4f}, n={n_pairs}, cog_mean={cog_mean:.2f}, base_mean={base_mean:.2f}"
    )


rng = np.random.default_rng(42)
metrics_ordered = ["psnr", "lpips"]  # PSNR top row, LPIPS bottom row

with plt.style.context("styles/ieee-tmi.mplstyle"):
    # Create paired dot plot with 2 rows (metrics) x 3 cols (x_values)
    fig, axes = plt.subplots(2, 3, sharey="row", figsize=(3.5, 3.0))

    # Get unique x_values and sort them
    x_values = sorted(combined_results["x_value"].unique())

    # Get unique filestems to connect with lines
    filestems = combined_results["filestem"].unique()

    for row_idx, metric in enumerate(metrics_ordered):
        for idx, x_val in enumerate(x_values):
            ax = axes[row_idx, idx]

            # Filter data for this x_value
            data_subset = combined_results[combined_results["x_value"] == x_val]

            # Plot points for each strategy
            for i, strategy in enumerate(STRATEGIES_TO_PLOT):
                strategy_data = data_subset[
                    data_subset["selection_strategy"] == strategy
                ]

                # Add jitter to x position for visibility
                jitter = rng.normal(0, 0.02, len(strategy_data))
                x_pos = i + jitter

                ax.scatter(
                    x_pos,
                    strategy_data[metric],
                    color=STRATEGY_COLORS[strategy],
                    alpha=0.6,
                    label=STRATEGY_NAMES[strategy]
                    if (row_idx == 0 and idx == 0)
                    else None,
                )

            # Draw lines connecting the same filestem across strategies
            for filestem in filestems:
                filestem_data: pd.DataFrame = data_subset[
                    data_subset["filestem"] == filestem
                ]

                # Sort by strategy order
                filestem_data = filestem_data.set_index("selection_strategy")
                filestem_data = filestem_data.reindex(STRATEGIES_TO_PLOT)

                is_harmonic = "harmonic" in str(filestem_data["filepath"])

                if len(filestem_data) == len(STRATEGIES_TO_PLOT):
                    x_positions = list(range(len(STRATEGIES_TO_PLOT)))
                    y_positions = filestem_data[metric].values
                    ax.plot(
                        x_positions,
                        y_positions,
                        "k-" if is_harmonic else "k--",
                        alpha=0.2,
                        zorder=0,
                    )

            # Formatting: only label x-axis on bottom row
            if row_idx == len(metrics_ordered) - 1:
                ax.set_xlabel(f"{int(x_val)}")
            ax.set_xticks([])
            ax.grid(
                True,
                # alpha=0.3,
                axis="y",
            )

            if idx == 0:
                ax.set_ylabel(METRIC_NAMES[metric])

    # Add significance annotations (* p<0.05, ** p<0.01)
    cognitive_strategy = "greedy_entropy"
    baseline_strategies = ["uniform_random", "equispaced"]

    # First pass: count max number of significant brackets per row to allocate space
    max_levels_per_row = {row_idx: 0 for row_idx in range(len(metrics_ordered))}
    for row_idx, metric in enumerate(metrics_ordered):
        for idx, x_val in enumerate(x_values):
            level = 0
            for baseline in baseline_strategies:
                key = (metric, baseline, x_val)
                if key in wilcoxon_results:
                    _, p_value, _, _, _ = wilcoxon_results[key]
                    if p_value < 0.05:
                        level += 1
            max_levels_per_row[row_idx] = max(max_levels_per_row[row_idx], level)

    # Second pass: draw brackets at uniform height across each row
    for row_idx, metric in enumerate(metrics_ordered):
        # Get shared ylim across all columns in this row (sharey="row")
        ymin, ymax = axes[row_idx, 0].get_ylim()
        y_range = ymax - ymin
        bracket_height = 0.02 * y_range
        bracket_spacing = 0.07 * y_range
        n_levels = max_levels_per_row[row_idx]

        # Expand ylim once for the whole row to fit all brackets tightly
        if n_levels > 0:
            new_ymax = ymax + n_levels * bracket_spacing + bracket_height * 0.5
            for col_idx in range(len(x_values)):
                axes[row_idx, col_idx].set_ylim(ymin, new_ymax)

        for idx, x_val in enumerate(x_values):
            ax = axes[row_idx, idx]
            cog_pos = STRATEGIES_TO_PLOT.index(cognitive_strategy)
            level = 0
            for baseline in baseline_strategies:
                base_pos = STRATEGIES_TO_PLOT.index(baseline)
                key = (metric, baseline, x_val)
                if key not in wilcoxon_results:
                    continue
                _, p_value, _, _, _ = wilcoxon_results[key]
                if p_value >= 0.05:
                    continue

                stars = "**" if p_value < 0.01 else "*"
                x1 = cog_pos
                x2 = base_pos
                # Place all brackets at the same absolute y, anchored to top
                y = ymax + level * bracket_spacing

                ax.plot(
                    [x1, x1, x2, x2],
                    [y, y + bracket_height, y + bracket_height, y],
                    lw=0.8,
                    c="k",
                    clip_on=False,
                    marker="",
                    linestyle="-" if p_value < 0.01 else "--",
                )
                level += 1

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        h,
        l,
        loc="outside upper center",
        ncol=3,
        frameon=False,
    )
    fig.supxlabel("# Scan Lines")

    savepath = DATA_ROOT / "paired_dot_plot_psnr_lpips.png"
    plt.savefig(savepath)
    plt.savefig(savepath.with_suffix(".pdf"))
    print(f"Plot saved to {savepath}")
