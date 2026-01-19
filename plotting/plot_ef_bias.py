"""
Reconstruction quality (PSNR) plotted against patient ejection fraction. The lack of correlation
indicates that reconstruction performance is consistent across varying ejection fractions,
suggesting no bias against outlier patients.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ulsa.plotting.index import extract_sweep_data


def load_ef_data(csv_path):
    """Load EF data from FileList.csv into a dictionary."""
    df = pd.read_csv(csv_path)
    return dict(zip(df["FileName"], df["EF"]))


def plot_ef_psnr_correlation(df, save_path=None):
    """Create publication-ready joint plot with KDE marginals and background histograms."""
    fig = plt.figure()
    gs = plt.GridSpec(
        2,
        2,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        hspace=0.05,
        wspace=0.05,
        figure=fig,
    )

    # Create the scatter plot and marginal axes
    ax_joint = fig.add_subplot(gs[1, 0])
    ax_marg_x = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)

    # Plot scatter by strategy
    strategies = df["selection_strategy"].unique()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    markers = ["o", "s", "D", "^", "v"]

    for idx, strategy in enumerate(strategies):
        mask = df["selection_strategy"] == strategy
        ax_joint.scatter(
            df[mask]["EF"],
            df[mask]["psnr"],
            alpha=0.7,
            label=strategy,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            s=20,  # Smaller markers for paper
        )

        # Add correlation line
        z = np.polyfit(df[mask]["EF"], df[mask]["psnr"], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df[mask]["EF"].min(), df[mask]["EF"].max(), 100)
        ax_joint.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=1)

    # Calculate correlation coefficient
    corr = df[mask]["EF"].corr(df["psnr"])
    exp = int(np.floor(np.log10(abs(corr))))
    mantissa = corr / (10**exp)
    ax_joint.text(
        0.05,
        0.9,
        f"r = {mantissa:.2f}×10$^{{{exp}}}$",
        transform=ax_joint.transAxes,
    )
    # fig.legend()

    # Plot marginal distributions using KDE
    from scipy.stats import gaussian_kde

    for idx, strategy in enumerate(strategies):
        mask = df["selection_strategy"] == strategy
        color = colors[idx % len(colors)]

        # X-axis histogram and KDE
        ax_marg_x.hist(df[mask]["EF"], bins=15, alpha=0.5, density=True, color=color)
        kde_x = gaussian_kde(df[mask]["EF"])
        x_range = np.linspace(df[mask]["EF"].min(), df[mask]["EF"].max(), 100)
        ax_marg_x.plot(
            x_range,
            kde_x(x_range),
            color=color,
            linewidth=1,
            marker="",
        )
        ax_marg_x.fill_between(x_range, kde_x(x_range), alpha=0.3, color=color)

        # Y-axis histogram and KDE
        ax_marg_y.hist(
            df[mask]["psnr"],
            bins=15,
            alpha=0.5,
            density=True,
            orientation="horizontal",
            color=color,
        )
        kde_y = gaussian_kde(df[mask]["psnr"])
        y_range = np.linspace(df["psnr"].min(), df["psnr"].max(), 100)
        ax_marg_y.plot(
            kde_y(y_range),
            y_range,
            color=color,
            linewidth=1,
            marker="",
        )
        ax_marg_y.fill_betweenx(y_range, kde_y(y_range), alpha=0.3, color=color)

    # Customize plots
    ax_joint.set_xlabel("Ejection Fraction [%]")
    ax_joint.set_ylabel("PSNR [dB]")

    # Turn off tick labels on marginals
    ax_marg_x.tick_params(labelbottom=False)
    ax_marg_y.tick_params(labelleft=False)

    # Turn off marginal spines
    ax_marg_x.spines["top"].set_visible(False)
    ax_marg_x.spines["right"].set_visible(False)
    ax_marg_y.spines["top"].set_visible(False)
    ax_marg_y.spines["right"].set_visible(False)

    # Remove marginal ticks
    ax_marg_x.set_yticks([])
    ax_marg_y.set_xticks([])

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")

    plt.show()


if __name__ == "__main__":
    DATA_ROOT = "/mnt/z/usbmd/Wessel/ulsa"
    DATA_FOLDER = (
        Path(DATA_ROOT)
        / "Np_2/eval_echonet_dynamic_test_set/sweep_2026_01_08_225505_654881"
    )
    EF_CSV_PATH = "/mnt/USBMD_datasets/_RAW/EchoNet-Dynamic/FileList.csv"
    SAVE_ROOT = "./output/"

    ef_lookup = load_ef_data(EF_CSV_PATH)
    results_df = extract_sweep_data(
        [DATA_FOLDER], keys_to_extract=["psnr"], ef_lookup=ef_lookup
    )

    df = results_df[results_df["x_value"] == 14]

    # Create plots
    with plt.style.context("styles/ieee-tmi.mplstyle"):
        strategies = df["selection_strategy"].unique()
        for strategy in strategies:
            mask = df["selection_strategy"] == strategy
            for ext in [".pdf", ".png"]:
                plot_ef_psnr_correlation(
                    df[mask],
                    save_path=os.path.join(
                        SAVE_ROOT, f"ef_psnr_correlation_{strategy}{ext}"
                    ),
                )

    # Print summary statistics
    print("\nSummary Statistics:")
    print(
        results_df.groupby("selection_strategy").agg(
            {"psnr": ["mean", "std"], "EF": ["count", "mean", "std"]}
        )
    )
