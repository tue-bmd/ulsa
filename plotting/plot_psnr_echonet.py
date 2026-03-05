"""
Makes violin plots of PSNR and DICE scores for the various scan line selection strategies.
"""

import argparse
import os
from pathlib import Path

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
    STRATEGY_NAMES,
    ViolinPlotter,
    get_axis_label,
    natural_sort,
)

DATA_ROOT = "/mnt/z/usbmd/ulsa/Np_2"
DATA_FOLDER = Path(DATA_ROOT) / "eval_echonet_dynamic_test_set"
SUBSAMPLED_PATHS = [DATA_FOLDER / "sweep_2026_01_08_225505_654881"]


STRATEGIES_TO_PLOT = [
    # "downstream_propagation_summed",
    "greedy_entropy",
    "uniform_random",
    "equispaced",
    # Add/remove as needed
]

# Add this near the top of the file where other constants are defined
AXIS_LABEL_MAP = {
    "n_actions": "# Scan Lines (out of 112)",
    # Add more mappings as needed
}

HIGHER_IS_BETTER = {
    "psnr": True,
    "ssim": True,
    "dice": True,
    "lpips": False,
    "nrmse": False,
    "rmse": False,
    "mse": False,
}


def compute_win_rate(
    cog_values: np.ndarray,
    base_values: np.ndarray,
    higher_is_better: bool = True,
) -> tuple[int, int, int, int, float]:
    """
    Compute how many times the cognitive strategy beats, ties, or loses to the baseline.

    Args:
        cog_values: Array of metric values for the cognitive strategy (paired by patient).
        base_values: Array of metric values for the baseline strategy (paired by patient).
        higher_is_better: If True, cognitive wins when cog > base. If False, wins when cog < base.

    Returns:
        (wins, ties, losses, total, win_rate) where win_rate = wins / total.
    """
    assert len(cog_values) == len(base_values), "Arrays must be the same length."
    diff = cog_values - base_values
    if higher_is_better:
        wins = int(np.sum(diff > 0))
        losses = int(np.sum(diff < 0))
    else:
        wins = int(np.sum(diff < 0))
        losses = int(np.sum(diff > 0))
    ties = int(np.sum(diff == 0))
    total = len(diff)
    win_rate = wins / total if total > 0 else 0.0
    return wins, ties, losses, total, win_rate


def _log_too_many_blobs_count(results_df: pd.DataFrame):
    unique_filestems = results_df["filestem"].unique()
    unique_files_skipped = 0
    for filestem in unique_filestems:
        filestem_rows = results_df[results_df["filestem"] == filestem]
        if filestem_rows["too_many_blobs"].any():
            assert filestem_rows["too_many_blobs"].all(), (
                "Inconsistent blob filtering results."
            )
            unique_files_skipped += 1
    log.info(
        f"Skipped a total of {unique_files_skipped} files due to poor segmentation masks."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--x-axis",
        type=str,
        default="action_selection.n_actions",
        help="Config key to use for x-axis (dot notation)",
    )
    args = parser.parse_args()

    combined_results = extract_sweep_data(
        SUBSAMPLED_PATHS,
        keys_to_extract=["mse", "psnr", "dice", "lpips", "ssim"],
        x_axis_key=args.x_axis,
    )
    _log_too_many_blobs_count(combined_results)

    plotter = ViolinPlotter(
        xlabel=get_axis_label(args.x_axis, AXIS_LABEL_MAP),
        group_names=STRATEGY_NAMES,
        legend_loc="top",
        # scatter_kwargs={"alpha": 0.01, "s": 4},
        context="styles/ieee-tmi.mplstyle",
    )

    # Combined LPIPS and PSNR
    plt.close("all")
    x_values = [7, 14, 28]
    with plt.style.context("styles/ieee-tmi.mplstyle"):
        fig, axs = plt.subplots(1, 2)
        metric_name = "psnr"
        formatted_metric_name = METRIC_NAMES.get(metric_name, metric_name.upper())
        order_by = plotter._order_groups_by_means(
            df_to_dict(combined_results, metric_name), STRATEGIES_TO_PLOT, x_values
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
            save_path = f"./2d_lpips_psnr_combined_violin_plot{ext}"
            plt.savefig(save_path)
            log.info(
                f"Saved combined LPIPS and PSNR violin plot to {log.yellow(save_path)}"
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
        groups_to_plot=STRATEGIES_TO_PLOT,
        ylim=[0.58, 1.02],
        legend_kwargs={
            "loc": "outside upper center",
            "ncol": 3,
            "frameon": False,
        },
        order_by=order_by,
    )

    # Individual metrics plots
    x_values = [4, 7, 14, 28]
    for metric_name in ["psnr", "lpips", "ssim", "nrmse"]:
        formatted_metric_name = METRIC_NAMES.get(metric_name, metric_name.upper())
        for ext in [".pdf", ".png"]:
            plotter.plot(
                df_to_dict(combined_results, metric_name),
                save_path=f"./{metric_name}_violin_plot{ext}",
                x_label_values=x_values,
                metric_name=formatted_metric_name,
                legend_kwargs={
                    "loc": "outside upper center",
                    "ncol": 3,
                    "frameon": False,
                },
                order_by=order_by,
            )

    # Print results in a table format
    for metric_name in ["dice", "psnr"]:
        table = Table(title=f"{metric_name.upper()} Results", show_lines=True)
        table.add_column("Strategy", style="cyan", no_wrap=True)
        table.add_column(get_axis_label(args.x_axis, AXIS_LABEL_MAP), style="magenta")
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

    # Wilcoxon signed-rank test: Cognitive vs baselines (paired by patient)
    cognitive_strategy = "greedy_entropy"
    baseline_strategies = ["uniform_random", "equispaced"]

    for metric_name in ["dice", "psnr", "lpips", "ssim"]:
        table = Table(
            title=f"Wilcoxon Signed-Rank Test — {metric_name.upper()} "
            f"({STRATEGY_NAMES[cognitive_strategy]} vs baselines)",
            show_lines=True,
        )
        table.add_column("Baseline", style="cyan", no_wrap=True)
        table.add_column(get_axis_label(args.x_axis, AXIS_LABEL_MAP), style="magenta")
        table.add_column("Cognitive Mean", style="green")
        table.add_column("Baseline Mean", style="green")
        table.add_column("N (paired)", style="white")
        table.add_column("Wins/Ties/Losses", style="blue")
        table.add_column("Win Rate", style="bold blue")
        table.add_column("Statistic", style="yellow")
        table.add_column("p-value", style="bold red")
        table.add_column("Significant (p<0.05)", style="bold")

        # Build per-patient metric values from the DataFrame
        metric_col = metric_name
        if metric_name in ["nrmse", "rmse"]:
            # These are derived; skip if not a direct column
            continue

        cog_df = combined_results[
            combined_results["selection_strategy"] == cognitive_strategy
        ]

        for baseline in baseline_strategies:
            base_df = combined_results[
                combined_results["selection_strategy"] == baseline
            ]
            x_values_available = sorted(
                set(cog_df["x_value"].unique()) & set(base_df["x_value"].unique())
            )

            for x_val in x_values_available:
                cog_subset = cog_df[cog_df["x_value"] == x_val][
                    ["filestem", metric_col]
                ].dropna()
                base_subset = base_df[base_df["x_value"] == x_val][
                    ["filestem", metric_col]
                ].dropna()

                # Pair by filestem (patient)
                merged = pd.merge(
                    cog_subset,
                    base_subset,
                    on="filestem",
                    suffixes=("_cog", "_base"),
                )

                if len(merged) < 10:
                    table.add_row(
                        STRATEGY_NAMES.get(baseline, baseline),
                        str(x_val),
                        "-",
                        "-",
                        str(len(merged)),
                        "-",
                        "-",
                        f"Too few pairs ({len(merged)})",
                    )
                    continue

                cog_values = merged[f"{metric_col}_cog"].values
                base_values = merged[f"{metric_col}_base"].values

                stat, p_value = wilcoxon(cog_values, base_values)
                significant = "✓ Yes" if p_value < 0.05 else "✗ No"

                wins, ties, losses, total, win_rate = compute_win_rate(
                    cog_values,
                    base_values,
                    higher_is_better=HIGHER_IS_BETTER.get(metric_name, True),
                )

                table.add_row(
                    STRATEGY_NAMES.get(baseline, baseline),
                    str(x_val),
                    f"{np.mean(cog_values):.4f}",
                    f"{np.mean(base_values):.4f}",
                    str(len(merged)),
                    f"{wins}/{ties}/{losses}",
                    f"{win_rate:.1%}",
                    f"{stat:.1f}",
                    f"{p_value:.2e}",
                    significant,
                )

        console = Console()
        console.print(table)
