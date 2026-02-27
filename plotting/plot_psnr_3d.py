"""Plot PSNR and LPIPS results for 3D ultrasound active sampling experiments."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.stats import wilcoxon

from zea import init_device, log

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "numpy"
    init_device("cpu")

from ulsa.plotting.index import df_to_dict, extract_sweep_data
from ulsa.plotting.plot_utils import (
    METRIC_NAMES,
    STRATEGY_COLORS,
    STRATEGY_NAMES,
    OverlappingHistogramPlotter,
    ViolinPlotter,
    get_axis_label,
)

AXIS_LABEL_MAP_3D = {
    "n_actions": "# Elevation Planes (out of 48)",
    # Add more mappings as needed
}

SIGNIFICANCE_LEVEL = 0.01


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--x-axis",
        type=str,
        default="action_selection.n_actions",
        help="Config key to use for x-axis (dot notation)",
    )
    args = parser.parse_args()

    # Define your sweep paths here
    SWEEPS = [
        "/mnt/z/usbmd/ulsa/Np_2/eval_3d/sweep_2026_01_09_150907_746810",
        "/mnt/z/usbmd/ulsa/Np_2/eval_3d/sweep_2026_01_09_170631_559863",
    ]

    keys_to_extract = ["psnr", "lpips"]

    combined_results = extract_sweep_data(
        SWEEPS,
        keys_to_extract=keys_to_extract,
        x_axis_key=args.x_axis,
    )

    plotter = ViolinPlotter(
        xlabel="# Elevation Planes (out of 48)",
        group_names=STRATEGY_NAMES,
        group_colors=STRATEGY_COLORS,
        legend_loc="top",
        # scatter_kwargs={"alpha": 0.05, "s": 7},
        context="styles/ieee-tmi.mplstyle",
    )

    # results = {}
    # for metric_name in keys_to_extract:
    #     results[metric_name] = df_to_dict(combined_results, metric_name)
    # combined_results = results

    # PSNR plot
    for metric_name in keys_to_extract:
        x_values = [3, 6, 12]
        formatted_metric_name = METRIC_NAMES.get(metric_name, metric_name.upper())
        plotter.plot(
            # sort_by_names(combined_results[metric_name], STRATEGY_NAMES.keys()),
            df_to_dict(combined_results, metric_name),
            save_path=f"./3d_{metric_name}_violin_plot.pdf",
            x_label_values=x_values,
            metric_name=formatted_metric_name,
            order_by=None,
            legend_kwargs={
                "loc": "outside upper center",
                "ncol": 3,
                "frameon": False,
            },
        )

    # Combined LPIPS and PSNR
    plotter = ViolinPlotter(
        xlabel=get_axis_label(args.x_axis, AXIS_LABEL_MAP_3D),
        group_names=STRATEGY_NAMES,
        legend_loc="top",
        # scatter_kwargs={"alpha": 0.01, "s": 4},
        context="styles/ieee-tmi.mplstyle",
    )
    plt.close("all")
    x_values = [3, 6, 12]
    with plt.style.context("styles/ieee-tmi.mplstyle"):
        fig, axs = plt.subplots(1, 2)
        metric_name = "psnr"
        formatted_metric_name = METRIC_NAMES.get(metric_name, metric_name.upper())
        order_by = plotter._order_groups_by_means(
            df_to_dict(combined_results, metric_name), STRATEGY_NAMES, x_values
        )
        plotter.plot(
            df_to_dict(combined_results, metric_name),
            save_path=None,
            x_label_values=x_values,
            metric_name=formatted_metric_name,
            ax=axs[0],
            legend_kwargs=None,
            order_by=order_by,
        )
        metric_name = "lpips"
        formatted_metric_name = METRIC_NAMES.get(metric_name, metric_name.upper())
        plotter.plot(
            df_to_dict(combined_results, metric_name),
            save_path=None,
            x_label_values=x_values,
            metric_name=formatted_metric_name,
            order_by=order_by,
            ax=axs[1],
            legend_kwargs=None,
        )
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(
            h,
            l,
            loc="outside upper center",
            ncol=3,
            frameon=False,
        )
        for ext in [".pdf", ".png"]:
            save_path = f"./3d_lpips_psnr_combined_violin_plot{ext}"
            plt.savefig(save_path)
            log.info(
                f"Saved combined LPIPS and PSNR violin plot to {log.yellow(save_path)}"
            )

    # Find global min/max for PSNR for consistent binning and ticks
    hist_data = df_to_dict(combined_results, metric_name)
    all_psnr_values = []
    for group in hist_data:
        for x_val in hist_data[group]:
            values = hist_data[group][x_val]
            flat_values = [item for sublist in values for item in sublist]
            all_psnr_values.extend(flat_values)
    global_min = np.min(all_psnr_values)
    global_max = np.max(all_psnr_values)
    bins = 30
    bin_edges = np.linspace(global_min, global_max, bins + 1)

    # Overlapping histogram plot (new)
    hist_plotter = OverlappingHistogramPlotter(
        xlabel="# Elevation Planes (out of 48)",
        group_names=STRATEGY_NAMES,
        group_colors=STRATEGY_COLORS,
        context="styles/ieee-tmi.mplstyle",
        alpha=0.4,
        kde=True,
        kde_lw=2,
        figsize=(6, 5),
        bins=bins,
        density=True,
    )
    hist_plotter.plot_overlapping_histograms_by_xvalue(
        hist_data,
        save_path=f"./3d_{metric_name}_overlapping_histograms.pdf",
        x_label_values=x_values,
        metric_name=formatted_metric_name,
        outer_y_label="# Elevation Planes",  # Large label for the stack
        inner_y_label="Density",  # Small repeated label for each subplot
        bin_edges=bin_edges,
        density=True,
    )

    # Print results in a table format
    for metric_name in ["psnr"]:
        table = Table(title=f"{metric_name.upper()} Results", show_lines=True)
        table.add_column("Strategy", style="cyan", no_wrap=True)
        table.add_column(
            get_axis_label(args.x_axis, AXIS_LABEL_MAP_3D), style="magenta"
        )
        table.add_column("Mean", style="green")
        table.add_column("Std", style="yellow")
        table.add_column("Count", style="white")

        results = df_to_dict(combined_results, metric_name)
        for group in results.keys():
            for x_value in sorted(results[group].keys()):
                values = results[group][x_value]
                # sequences have different lengths in the 3d case
                flat_values = [item for sublist in values for item in sublist]
                mean = np.mean(flat_values)
                std = np.std(flat_values)
                count = len(flat_values)
                table.add_row(
                    str(STRATEGY_NAMES.get(group, group)),
                    str(x_value),
                    f"{mean:.2f}",
                    f"{std:.2f}",
                    str(count),
                )

        console = Console()
        console.print(table)

    # Wilcoxon signed-rank test: Cognitive vs baselines (paired by volume)
    cognitive_strategy = "greedy_entropy"
    baseline_strategies = ["uniform_random", "equispaced"]

    for metric_name in ["psnr", "lpips"]:
        table = Table(
            title=f"Wilcoxon Signed-Rank Test — {metric_name.upper()} "
            f"({STRATEGY_NAMES[cognitive_strategy]} vs baselines)",
            show_lines=True,
        )
        table.add_column("Baseline", style="cyan", no_wrap=True)
        table.add_column(
            get_axis_label(args.x_axis, AXIS_LABEL_MAP_3D), style="magenta"
        )
        table.add_column("Cognitive Mean", style="green")
        table.add_column("Baseline Mean", style="green")
        table.add_column("N (paired)", style="white")
        table.add_column("Statistic", style="yellow")
        table.add_column("p-value", style="bold red")
        table.add_column(f"Significant (p<{SIGNIFICANCE_LEVEL:.2f})", style="bold")

        cog_df = combined_results[
            combined_results["selection_strategy"] == cognitive_strategy
        ].copy()
        # In 3D, metric values can be per-frame arrays; average to one scalar per volume
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

                # Pair by filestem (volume)
                merged = pd.merge(
                    cog_subset,
                    base_subset,
                    on="filestem",
                    suffixes=("_cog", "_base"),
                )

                cog_values = merged[f"{metric_name}_mean_cog"].values
                base_values = merged[f"{metric_name}_mean_base"].values

                stat, p_value = wilcoxon(cog_values, base_values)
                significant = "✓ Yes" if p_value < SIGNIFICANCE_LEVEL else "✗ No"

                table.add_row(
                    STRATEGY_NAMES.get(baseline, baseline),
                    str(x_val),
                    f"{np.mean(cog_values):.4f}",
                    f"{np.mean(base_values):.4f}",
                    str(len(merged)),
                    f"{stat:.1f}",
                    f"{p_value:.2e}",
                    significant,
                )

        console = Console()
        console.print(table)
