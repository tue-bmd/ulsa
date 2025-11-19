"""
Plot FWHM traces and reconstructions for CIRS phantom data.

This script loads reconstruction data from different sampling strategies,
scan-converts them to Cartesian coordinates, extracts intensity profiles
along a specified line, computes FWHM in millimeters, and generates a
comparison figure with reconstructions and FWHM traces.

Example usage:
    python plotting/plot_CIRS_fwhm_gcnr.py \
        --data-dir "/mnt/z/usbmd/Wessel/eval_phantom2/20251118_CIRS_0000" \
        --save-dir ./output \
        --frame-idx 19 \
        --point1 54 60 \
        --point2 54 76

"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from zea import init_device, log
from zea.ops import translate

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "numpy"
    init_device("cpu")
    sys.path.append("/ulsa")

from ulsa.io_utils import _scan_convert
from zea.metrics import fwhm

STRATEGY_NAMES = {
    "greedy_entropy": "Active Perception",
    "uniform_random": "Random",
    "equispaced": "Equispaced",
    "focused": "Focused",
    "diverging": "Diverging",
}

STRATEGY_COLORS = {
    "greedy_entropy": "#2ca02c",  # Green
    "uniform_random": "#1f77b4",  # Blue
    "equispaced": "#ff7f0e",  # Orange
    "focused": "#d62728",  # Red
    "diverging": "#9467bd",  # Purple
}

# Strategies that use original acquisitions (not reconstructed from subsampled data)
ORIGINAL_ACQUISITONS = ["focused", "diverging"]


def calculate_fwhm(point1, point2, reconstruction_sc, rho_max):
    # Compute FWHM trace along the measurement line
    # Extract intensity values directly from scan-converted image
    from scipy.ndimage import map_coordinates

    row1, col1 = point1
    row2, col2 = point2
    num_samples = 500

    rows = np.linspace(row1, row2, num_samples)
    cols = np.linspace(col1, col2, num_samples)

    # Extract intensity values along the line using bilinear interpolation
    coordinates = np.vstack([rows, cols])
    trace_db = np.nan_to_num(
        map_coordinates(
            reconstruction_sc,
            coordinates,
            order=1,
            mode="constant",
            cval=np.nan,
        )
    )

    # Calculate physical distance in mm using scan geometry -- we know the total
    # depth in mm (rho_max) and the center of the scan cone, so we can compute
    # the lateral distance between x1 and x2.
    y1, x1 = point1
    y2, x2 = point2
    assert y1 == y2, "Points must be on the same row for horizontal line."
    assert x1 < x2, "point1 must be to the left of point2."
    sc_height, sc_width = reconstruction_sc.shape
    scan_cone_apex = (sc_width // 2) + 1  # Center of scan cone at top
    # We assume that x1 and x2 are to the left of the scan cone apex.
    # If we decide to change point1 and point2 locations we may need to
    # update the trigonometry slightly.
    assert x2 < scan_cone_apex, "point2 must be left of scan cone apex."

    # Calculate lateral distance using trigonometry
    left_distance_from_center = scan_cone_apex - x1
    theta_left = np.arctan(left_distance_from_center / y1)
    left_distance_mm = (y1 / sc_height) * rho_max * np.tan(theta_left)
    x1_x2_distance_mm = ((x2 - x1) / left_distance_from_center) * left_distance_mm

    # Create distance array centered at 0 for symmetric plotting
    # Ranges from -total_distance_mm/2 to +total_distance_mm/2
    total_distance_mm = x1_x2_distance_mm
    distances_mm = np.linspace(
        -total_distance_mm / 2, total_distance_mm / 2, num_samples
    )

    # Calculate FWHM in mm
    # Find the maximum intensity in the trace
    trace_max = np.max(trace_db)
    # Define half maximum as -3 dB from peak (since we're in dB scale)
    half_in_db = 3
    # Find all indices where intensity is at or above half maximum
    indices_above_half_max = np.nonzero(trace_db >= trace_max - half_in_db)[0]
    fwhm_start = indices_above_half_max[0]
    fwhm_end = indices_above_half_max[-1]
    # Calculate FWHM as the distance between start and end points
    fwhm_val_mm = distances_mm[fwhm_end] - distances_mm[fwhm_start]

    return trace_db, distances_mm, fwhm_val_mm


def plot_fwhm_comparison(
    data_dir: Path,
    save_dir: Path,
    frame_idx: int = 3,
    point1: tuple = (56, 50),
    point2: tuple = (56, 150),
    strategies: list = None,
    rho_max: float = 80.0,
    vmin: float = -60,
    vmax: float = 0,
    context="styles/ieee-tmi.mplstyle",
):
    """Plot scan-converted reconstructions with FWHM measurement line and FWHM trace comparison.

    Creates a multi-panel figure with:
    - Left: Colorbar for intensity scale
    - Middle: Scan-converted reconstruction images for each strategy with measurement line overlay
    - Right: Overlaid FWHM traces showing intensity profiles along the measurement line

    Args:
        data_dir: Directory containing .npz files for each strategy
        save_dir: Directory to save output plots
        frame_idx: Frame index to visualize (default: 3)
        point1: Tuple of (row, col) pixel indices for start point of measurement line
        point2: Tuple of (row, col) pixel indices for end point of measurement line
        strategies: List of strategy names to plot (default: all strategies)
        rho_max: Maximum imaging depth in mm (default: 80.0)
        vmin: Minimum dB value for display (default: -60)
        vmax: Maximum dB value for display (default: 0)
        context: Matplotlib style file to use (default: "styles/ieee-tmi.mplstyle")
    """
    if strategies is None:
        strategies = [
            "greedy_entropy",
            "uniform_random",
            "equispaced",
            "focused",
            "diverging",
        ]

    # Load reconstruction data for all strategies
    data = {}
    theta_range = None

    for strategy in strategies:
        npz_path = data_dir / f"{strategy}.npz"
        if npz_path.exists():
            npz_data = np.load(str(npz_path))
            data[strategy] = npz_data

            # Extract theta_range (should be same for all strategies)
            if theta_range is None and "theta_range" in npz_data:
                theta_range = npz_data["theta_range"]
                log.info(f"Found theta_range: {theta_range}")

            log.info(f"Loaded {strategy}: {npz_data['reconstructions'].shape}")
        else:
            log.warning(f"File not found: {npz_path}")

    if not data:
        log.error("No data files found!")
        return

    # Default theta_range if not found in data files
    if theta_range is None:
        theta_range = (-0.78539816, 0.78539816)  # -45 to 45 degrees in radians
        log.warning(f"No theta_range found in data, using default: {theta_range}")

    # Convert theta_range from radians to degrees for scan conversion
    scan_conversion_angles = (np.rad2deg(theta_range[0]), np.rad2deg(theta_range[1]))

    with plt.style.context(context):
        # Create figure layout: colorbar + images + FWHM plot
        n_strategies = len(data)
        fig_width = 0.3 + 2 * n_strategies + 4  # Colorbar + images + plot + margin
        fig = plt.figure(figsize=(fig_width, 2.5))

        import matplotlib.gridspec as gridspec

        # Define grid: [colorbar, image1, image2, ..., imageN, FWHM plot]
        gs = gridspec.GridSpec(
            1,
            n_strategies + 2,
            width_ratios=[0.05] + [1] * n_strategies + [1.5],
            wspace=0.2,
        )

        # Convert reconstructions to dB scale and scan convert once per strategy
        reconstructions_raw = {}
        reconstructions_sc = {}

        for strategy, strategy_data in data.items():
            # Translate from [-1, 1] normalized range to [-60, 0] dB for reconstructed data
            if strategy not in ORIGINAL_ACQUISITONS:
                reconstruction = translate(
                    np.clip(strategy_data["reconstructions"][frame_idx], -1, 1),
                    (-1, 1),
                    (-60, 0),
                )
            else:
                # Original acquisitions are already in dB
                reconstruction = strategy_data["reconstructions"][frame_idx]

            # Clip all reconstructions to expected dynamic range
            reconstructions_raw[strategy] = np.clip(reconstruction, -60, 0)

            # Scan convert once and store
            reconstruction_sc = _scan_convert(
                reconstructions_raw[strategy],
                scan_conversion_angles=scan_conversion_angles,
                fill_value=np.nan,  # NaN for transparent background outside scan cone
                order=1,
            )
            reconstructions_sc[strategy] = reconstruction_sc

        # Create colorbar in first column
        cax = fig.add_subplot(gs[0, 0])
        import matplotlib.cm as cm

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = cm.ScalarMappable(cmap="gray", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label("Intensity [dB]", fontsize=8, rotation=90, labelpad=5)
        cbar.ax.tick_params(labelsize=7)

        # Move colorbar ticks and label to the left side
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")

        # Adjust colorbar height to match images
        pos_cbar = cax.get_position()
        new_cbar_height = 0.45
        new_cbar_bottom = pos_cbar.y0 + (pos_cbar.height - new_cbar_height) / 2
        cax.set_position(
            [pos_cbar.x0, new_cbar_bottom, pos_cbar.width, new_cbar_height]
        )

        # Plot scan-converted reconstructions (starting from column 1 after colorbar)
        for idx, strategy in enumerate(data.keys()):
            reconstruction_sc = reconstructions_sc[strategy]

            ax = fig.add_subplot(gs[0, idx + 1])
            im = ax.imshow(
                reconstruction_sc,
                cmap="gray",
                aspect="equal",  # Maintain correct aspect ratio
                vmin=vmin,
                vmax=vmax,
                origin="upper",
            )

            strategy_display = STRATEGY_NAMES.get(strategy, strategy)
            ax.set_title(strategy_display, fontsize=9, fontweight="bold")

            # Draw FWHM measurement line overlay
            ax.plot(
                [point1[1], point2[1]],
                [point1[0], point2[0]],
                "r-",
                linewidth=1,
                alpha=0.4,
            )
            # Draw endpoint markers
            ax.plot(
                [point1[1], point2[1]],
                [point1[0], point2[0]],
                "ro",
                markersize=2,
                alpha=0.4,
            )

            ax.axis("off")

        # Plot FWHM traces in last column
        ax_fwhm = fig.add_subplot(gs[0, -1])

        all_traces = []
        for strategy in strategies:
            if strategy not in data:
                continue

            # Use pre-computed scan-converted image
            reconstruction_sc = reconstructions_sc[strategy]

            trace_db, distances_mm, fwhm_val_mm = calculate_fwhm(
                point1, point2, reconstruction_sc, rho_max
            )

            # Store trace for half-maximum calculation
            all_traces.append(trace_db)

            color = STRATEGY_COLORS.get(strategy, "#000000")
            strategy_display = STRATEGY_NAMES.get(strategy, strategy)

            # Plot trace with FWHM value in legend
            ax_fwhm.plot(
                distances_mm,
                trace_db,
                linewidth=1.5,
                color=color,
                marker="",
                linestyle="-",
                label=f"{strategy_display} (FWHM={fwhm_val_mm:.2f} mm)",
            )

        # Add vertical line at center (0 mm)
        ax_fwhm.axvline(
            x=0,
            color="gray",
            linestyle=":",
            linewidth=0.8,
            alpha=0.3,
        )

        ax_fwhm.set_xlabel("Lateral distance from center [mm]", fontsize=9)
        ax_fwhm.set_ylabel("Intensity [dB]", fontsize=9)

        # Move y-axis to the right side
        ax_fwhm.yaxis.tick_right()
        ax_fwhm.yaxis.set_label_position("right")

        ax_fwhm.set_ylim(-65, 5)
        ax_fwhm.set_title(
            f"FWHM",
            fontsize=9,
            fontweight="bold",
        )

        # Place legend outside plot area to the right
        ax_fwhm.legend(
            fontsize=6,
            loc="center left",
            bbox_to_anchor=(1.2, 0.5),
            framealpha=0.9,
        )
        ax_fwhm.grid(True, alpha=0.3)
        ax_fwhm.tick_params(labelsize=7)

        # Adjust FWHM subplot height to match images
        pos = ax_fwhm.get_position()
        new_height = 0.45
        new_bottom = pos.y0 + (pos.height - new_height) / 2
        ax_fwhm.set_position([pos.x0, new_bottom, pos.width, new_height])

        # Save figure in multiple formats
        for ext in [".pdf", ".png"]:
            save_file = (
                save_dir
                / f"fwhm_comparison_frame{frame_idx}_p1_{point1[0]}_{point1[1]}__p2_{point2[0]}_{point2[1]}{ext}"
            )
            plt.savefig(save_file, dpi=300, bbox_inches="tight")
            log.info(f"Saved FWHM comparison to {log.yellow(save_file)}")

        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate FWHM comparison plots for CIRS phantom reconstructions"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing .npz files for each strategy",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./output",
        help="Directory to save the plots (default: ./output)",
    )
    parser.add_argument(
        "--frame-idx",
        type=int,
        default=3,
        help="Frame index to visualize (default: 3)",
    )
    parser.add_argument(
        "--point1",
        type=int,
        nargs=2,
        default=[54, 60],
        help="First point as index (row, col) in scan-converted image",
    )
    parser.add_argument(
        "--point2",
        type=int,
        nargs=2,
        default=[54, 76],
        help="Second point as index (row, col) in scan-converted image",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=None,
        help="List of strategies to plot (default: all available strategies)",
    )
    args = parser.parse_args()

    # Create save directory if it doesn't exist
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)

    # Generate comparison plot
    log.info("Generating FWHM comparison plot...")
    plot_fwhm_comparison(
        data_dir=data_dir,
        save_dir=save_dir,
        frame_idx=args.frame_idx,
        point1=tuple(args.point1),
        point2=tuple(args.point2),
        strategies=args.strategies,
        context="styles/ieee-tmi.mplstyle",
    )

    log.info("Done!")
