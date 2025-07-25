import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

from zea import Config, init_device

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "numpy"
    init_device("cpu")
    sys.path.append("/ulsa")

# Import plotting constants/utilities from plot_psnr_dice.py
from plotting.plot_psnr_dice import (
    AXIS_LABEL_MAP,
    METRIC_NAMES,
    STRATEGIES_TO_PLOT,
    STRATEGY_CANONICAL_MAP,
    STRATEGY_COLORS,
    STRATEGY_NAMES,
    extract_sweep_data,
    get_axis_label,
    recursive_map_to_dict,
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

    # Define your sweep paths here
    SWEEPS = [
        "/mnt/z/Ultrasound-BMD/Ultrasound-BMd/data/oisin/ULSA_out/3d_test_3_frame/sweep_2025_06_17_092553_947259",
        "/mnt/z/Ultrasound-BMD/Ultrasound-BMd/data/oisin/ULSA_out/3d_test_3_frame/sweep_2025_06_17_132207_139566",
        "/mnt/z/Ultrasound-BMD/Ultrasound-BMd/data/oisin/ULSA_out/3d_test_3_frame/sweep_2025_06_17_165304_798534",
    ]

    # Hardcoded list of target filepaths to include (set to None to include all)
    # INCLUDE_ONLY_THESE_FILES = [
    #     "/mnt/z/Ultrasound-BMd/data/tristan/elevation_interpolation/data/cleaned/test/002/IM_0033.hdf5",
    #     "/mnt/z/Ultrasound-BMd/data/tristan/elevation_interpolation/data/cleaned/test/0039_MI_PV_strain/IM_0041.hdf5"
    #     # Add more filepaths as needed
    # ]
    # Set to None to include all
    INCLUDE_ONLY_THESE_FILES = None

    # Initialize combined results with the same structure as individual results
    combined_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Process each subsampled path
    for subsampled_path in SWEEPS:
        try:
            results = extract_sweep_data(
                subsampled_path,
                keys_to_extract=["psnr"],  # Only MSE and PSNR
                x_axis_key=args.x_axis,
                include_only_these_files=INCLUDE_ONLY_THESE_FILES,
            )

            # Combine results by extending lists
            for metric in results:
                for strategy in results[metric]:
                    for x_value in results[metric][strategy]:
                        combined_results[metric][strategy][x_value].extend(
                            results[metric][strategy][x_value]
                        )

        except Exception as e:
            print(f"Failed to process {subsampled_path}: {e}")

    combined_results = recursive_map_to_dict(combined_results)
    np.save("./combined_results_3d.npy", combined_results)

    plotter = ViolinPlotter(
        xlabel="# Elevation Planes (out of 48)",
        group_names=STRATEGY_NAMES,
        group_colors=STRATEGY_COLORS,
        legend_loc="top",
        scatter_kwargs={"alpha": 0.05, "s": 7},
        context="styles/ieee-tmi.mplstyle",
    )

    # PSNR plot
    metric_name = "psnr"
    x_values = [3, 6, 12]
    formatted_metric_name = METRIC_NAMES.get(metric_name, metric_name.upper())
    plotter.plot(
        sort_by_names(combined_results[metric_name], STRATEGY_NAMES.keys()),
        save_path=f"./{metric_name}_violin_plot.pdf",
        x_label_values=x_values,
        metric_name=formatted_metric_name,
        order_by_means=False,
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
        save_path=f"./{metric_name}_overlapping_histograms.pdf",
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
