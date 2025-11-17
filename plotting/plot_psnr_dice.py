"""
Makes violin plots of PSNR and DICE scores for the various scan line selection strategies.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from zea import init_device, log

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "numpy"
    init_device("cpu")
    sys.path.append("/ulsa")


from plotting.index import extract_sweep_data
from plotting.plot_utils import ViolinPlotter, natural_sort

# DATA_ROOT = "/mnt/z/prjs0966"
# DATA_FOLDER = Path(DATA_ROOT) / "oisin/ULSA_out/eval_echonet_dynamic_test_set"
DATA_ROOT = "/mnt/z/usbmd/Wessel/"
DATA_FOLDER = Path(DATA_ROOT) / "eval_echonet_dynamic_test_set"
SUBSAMPLED_PATHS = [
    DATA_FOLDER / "sharding_sweep_2025-08-05_14-35-11",
    DATA_FOLDER / "sharding_sweep_2025-08-05_14-42-40",
]

STRATEGY_COLORS = {
    "downstream_propagation_summed": "#d62728",  # Red
    "greedy_entropy": "#1f77b4",  # Blue
    "equispaced": "#2ca02c",  # Green
    "uniform_random": "#ff7f0e",  # Orange
}

STRATEGY_NAMES = {
    "downstream_propagation_summed": "Measurement Information Gain",
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
    "dice": "DICE (→) [-]",
    "psnr": "PSNR (→) [dB]",
    "ssim": "SSIM (→) [-]",
    "lpips": "LPIPS (←) [-]",
    "mse": "MSE (←) [-]",  # on [0, 1] scale
    "rmse": "RMSE (←) [-]",  # on [0, 1] scale
    "nrmse": "NRMSE (←) [-]",
}

# Add this near the top of the file where other constants are defined
AXIS_LABEL_MAP = {
    "n_actions": "# Scan Lines (out of 112)",
    # Add more mappings as needed
}


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


def get_axis_label(key):
    """Get friendly label for axis keys."""
    base_key = key.split(".")[-1]
    return AXIS_LABEL_MAP.get(base_key, base_key.replace("_", " ").title())


def sort_by_names(combined_results, names):
    """Sort combined results by strategy names."""
    return {k: combined_results[k] for k in names if k in combined_results}


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
    x_min, x_max = 0, 255

    result = {}
    for _, row in df.iterrows():
        strategy = row["selection_strategy"]
        x_value = row["x_value"]
        if metric_name.lower() == "rmse":
            # scale [0, 255] to [0, 1]
            value = np.sqrt(row["mse"] / (x_max * x_max))
        if metric_name.lower() == "nrmse":
            value = np.sqrt(row["mse"]) / (x_max - x_min)
        elif metric_name.lower() == "mse":
            value = row["mse"] / (x_max * x_max)
        else:
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

    combined_results = extract_sweep_data(
        SUBSAMPLED_PATHS,
        keys_to_extract=["mse", "psnr", "dice", "lpips", "ssim"],
        x_axis_key=args.x_axis,
    )
    _log_too_many_blobs_count(combined_results)

    plotter = ViolinPlotter(
        xlabel=get_axis_label(args.x_axis),
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
            save_path = f"./lpips_psnr_combined_violin_plot{ext}"
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
