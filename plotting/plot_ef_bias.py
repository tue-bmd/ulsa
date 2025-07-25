"""
Reconstruction quality (PSNR) plotted against patient ejection fraction. The lack of correlation
indicates that reconstruction performance is consistent across varying ejection fractions,
suggesting no bias against outlier patients.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # pip install pandas
import yaml


def load_yaml(filepath):
    """Load YAML file."""
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def get_filename_from_path(filepath):
    """Extract filename without extension from full path."""
    return Path(filepath).stem


def load_ef_data(csv_path):
    """Load EF data from FileList.csv into a dictionary."""
    df = pd.read_csv(csv_path)
    return dict(zip(df["FileName"], df["EF"]))


def extract_psnr_ef_data(sweep_dir, ef_lookup):
    """Extract PSNR values and match with EF values."""
    results = []

    for run_dir in sorted(os.listdir(sweep_dir)):
        run_path = os.path.join(sweep_dir, run_dir)
        if not os.path.isdir(run_path):
            continue

        metrics_path = os.path.join(run_path, "metrics.npz")
        filepath_yaml = os.path.join(run_path, "target_filepath.yaml")
        config_path = os.path.join(run_path, "config.yaml")

        if not all(
            os.path.exists(p) for p in [metrics_path, filepath_yaml, config_path]
        ):
            continue

        # Load configuration and get num_lines
        config = load_yaml(config_path)
        num_lines = config.get("action_selection", {}).get("num_lines_to_sample")
        selection_strategy = config.get("action_selection", {}).get(
            "selection_strategy"
        )

        # Get target filename
        target_file = load_yaml(filepath_yaml)["target_filepath"]
        filename = get_filename_from_path(target_file)

        # Get EF value
        if filename not in ef_lookup:
            print(f"Warning: No EF data found for {filename}")
            continue

        ef_value = ef_lookup[filename]

        # Load PSNR values
        metrics = np.load(metrics_path, allow_pickle=True)
        if "psnr" in metrics:
            psnr_values = metrics["psnr"]
            if isinstance(psnr_values, np.ndarray) and psnr_values.size > 0:
                mean_psnr = np.mean(psnr_values)
                results.append(
                    {
                        "EF": ef_value,
                        "PSNR": mean_psnr,
                        "num_lines": num_lines,
                        "strategy": selection_strategy,
                        "filename": filename,
                    }
                )

    return pd.DataFrame(results)


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
    strategies = df["strategy"].unique()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    markers = ["o", "s", "D", "^", "v"]

    for idx, strategy in enumerate(strategies):
        mask = df["strategy"] == strategy
        ax_joint.scatter(
            df[mask]["EF"],
            df[mask]["PSNR"],
            alpha=0.7,
            label=strategy,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            s=20,  # Smaller markers for paper
        )

    # Add correlation line
    z = np.polyfit(df["EF"], df["PSNR"], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df["EF"].min(), df["EF"].max(), 100)
    ax_joint.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=1)

    # Calculate correlation coefficient
    corr = df["EF"].corr(df["PSNR"])
    exp = int(np.floor(np.log10(abs(corr))))
    mantissa = corr / (10**exp)
    ax_joint.text(
        0.05,
        0.9,
        f"r = {mantissa:.2f}×10$^{{{exp}}}$",
        transform=ax_joint.transAxes,
    )

    # Plot marginal distributions using KDE
    from scipy.stats import gaussian_kde

    for idx, strategy in enumerate(strategies):
        mask = df["strategy"] == strategy
        color = colors[idx % len(colors)]

        # X-axis histogram and KDE
        ax_marg_x.hist(df[mask]["EF"], bins=15, alpha=0.5, density=True, color=color)
        kde_x = gaussian_kde(df[mask]["EF"])
        x_range = np.linspace(df["EF"].min(), df["EF"].max(), 100)
        ax_marg_x.plot(x_range, kde_x(x_range), color=color, linewidth=1)
        ax_marg_x.fill_between(x_range, kde_x(x_range), alpha=0.3, color=color)

        # Y-axis histogram and KDE
        ax_marg_y.hist(
            df[mask]["PSNR"],
            bins=15,
            alpha=0.5,
            density=True,
            orientation="horizontal",
            color=color,
        )
        kde_y = gaussian_kde(df[mask]["PSNR"])
        y_range = np.linspace(df["PSNR"].min(), df["PSNR"].max(), 100)
        ax_marg_y.plot(kde_y(y_range), y_range, color=color, linewidth=1)
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
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


if __name__ == "__main__":
    SWEEP_PATH = "/mnt/z/Ultrasound-BMD/Ultrasound-BMd/data/oisin/ULSA_out/ef_bias_2/sweep_2025_03_27_135536"
    EF_CSV_PATH = "/mnt/z/Ultrasound-BMD/Ultrasound-BMd/data/USBMD_datasets/_RAW/EchoNet-Dynamic/FileList.csv"
    SAVE_ROOT = "."

    # Load EF lookup table
    ef_lookup = load_ef_data(EF_CSV_PATH)

    # Extract PSNR and EF data
    results_df = extract_psnr_ef_data(SWEEP_PATH, ef_lookup)

    # Create plots
    with plt.style.context("styles/ieee-tmi.mplstyle"):
        plot_ef_psnr_correlation(
            results_df, save_path=os.path.join(SAVE_ROOT, "ef_psnr_correlation.pdf")
        )

    # Print summary statistics
    print("\nSummary Statistics:")
    print(
        results_df.groupby("strategy").agg(
            {"PSNR": ["mean", "std"], "EF": ["count", "mean", "std"]}
        )
    )
