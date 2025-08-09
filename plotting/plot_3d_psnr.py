import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from zea import init_device

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "numpy"
    init_device("cpu")
    sys.path.append("/ulsa")

# Import plotting constants/utilities from plot_psnr_dice.py
from plotting.plot_psnr_dice import (
    METRIC_NAMES,
    STRATEGY_COLORS,
    STRATEGY_NAMES,
    df_to_dict,
    extract_and_combine_sweep_data,
    get_axis_label,
    sort_by_names,
)
from plotting.plot_utils import OverlappingHistogramPlotter, ViolinPlotter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--x-axis",
        type=str,
        default="action_selection.n_actions",
        help="Config key to use for x-axis (dot notation)",
    )
    args = parser.parse_args()

    TEMP_FILE = Path("/tmp/plot_3d_psnr.pkl")

    # Define your sweep paths here
    SWEEPS = [
        # "/mnt/z/Ultrasound-BMd/data/oisin/ULSA_benchmarks/3d/sweep_2025_07_30_075551_576866",
        "/mnt/z/Ultrasound-BMd/data/oisin/ULSA_benchmarks/3d/sweep_2025_08_05_135605_502495",
        "/mnt/z/Ultrasound-BMd/data/oisin/ULSA_benchmarks/3d/sweep_2025_08_08_073153_126915",
    ]

    keys_to_extract = ["psnr", "lpips"]

    if TEMP_FILE.exists():
        print(f"Loading existing combined results from {str(TEMP_FILE)}")
        combined_results = pd.read_pickle(TEMP_FILE)
    else:
        combined_results = extract_and_combine_sweep_data(
            SWEEPS,
            keys_to_extract=keys_to_extract,
            x_axis_key=args.x_axis,
        )
        combined_results.to_pickle(TEMP_FILE)

    plotter = ViolinPlotter(
        xlabel="# Elevation Planes (out of 48)",
        group_names=STRATEGY_NAMES,
        group_colors=STRATEGY_COLORS,
        legend_loc="top",
        scatter_kwargs={"alpha": 0.05, "s": 7},
        context="styles/ieee-tmi.mplstyle",
    )

    results = {}
    for metric_name in keys_to_extract:
        results[metric_name] = df_to_dict(combined_results, metric_name)
    combined_results = results

    # PSNR plot
    for metric_name in keys_to_extract:
        x_values = [3, 6, 12]
        formatted_metric_name = METRIC_NAMES.get(metric_name, metric_name.upper())
        plotter.plot(
            sort_by_names(combined_results[metric_name], STRATEGY_NAMES.keys()),
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

    # Find global min/max for PSNR for consistent binning and ticks
    all_psnr_values = []
    for group in combined_results[metric_name]:
        for x_val in combined_results[metric_name][group]:
            values = combined_results[metric_name][group][x_val]
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
        combined_results[metric_name],
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
        table.add_column(get_axis_label(args.x_axis), style="magenta")
        table.add_column("Mean", style="green")
        table.add_column("Std", style="yellow")
        table.add_column("Count", style="white")

        for group in combined_results[metric_name].keys():
            for x_value in sorted(combined_results[metric_name][group].keys()):
                values = combined_results[metric_name][group][x_value]
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
