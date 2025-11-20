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
        --point2 54 76 \
        --gcnr-center 40 82 \
        --gcnr-radius 5 \
        --gcnr-annulus-inner 8 \
        --gcnr-annulus-outer 12

"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from zea import init_device, log
from zea.ops import translate

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "numpy"
    init_device("cpu")
    sys.path.append("/ulsa")

from ulsa.io_utils import _scan_convert
from zea.metrics import gcnr

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
    """Compute FWHM trace along a line between two points in scan-converted image.

    Args:
        point1: Tuple of (row, col) pixel indices for start point
        point2: Tuple of (row, col) pixel indices for end point
        reconstruction_sc: Scan-converted image in dB scale
        rho_max: Maximum imaging depth in mm

    Returns:
        tuple: (trace_db, distances_mm, fwhm_val_mm)
    """
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

    # Calculate physical distance in mm using scan geometry
    # Assumes horizontal line (same row) for simplified geometry
    y1, x1 = point1
    y2, x2 = point2
    assert y1 == y2, "Points must be on the same row for horizontal line."
    assert x1 < x2, "point1 must be to the left of point2."
    sc_height, sc_width = reconstruction_sc.shape
    scan_cone_apex = (sc_width // 2) + 1  # Center of scan cone at top
    assert x2 < scan_cone_apex, "point2 must be left of scan cone apex."

    # Calculate lateral distance using trigonometry
    # Distance from apex to left point (in pixels)
    left_distance_from_center = scan_cone_apex - x1
    # Angle from apex to left point (in radians)
    theta_left = np.arctan(left_distance_from_center / y1)
    # Physical lateral distance from apex to left point (in mm)
    left_distance_mm = (y1 / sc_height) * rho_max * np.tan(theta_left)
    # Physical distance between the two points (in mm)
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
    if len(indices_above_half_max) > 0:
        fwhm_start = indices_above_half_max[0]
        fwhm_end = indices_above_half_max[-1]
        # Calculate FWHM as the distance between start and end points
        fwhm_val_mm = distances_mm[fwhm_end] - distances_mm[fwhm_start]
    else:
        fwhm_val_mm = 0.0

    return trace_db, distances_mm, fwhm_val_mm


def extract_circular_region(image, center, radius):
    """Extract pixels within a circular region.

    Args:
        image: 2D array
        center: Tuple of (row, col) for circle center
        radius: Radius in pixels

    Returns:
        1D array of pixel values within the circle
    """
    row_center, col_center = center
    height, width = image.shape

    # Create coordinate grids
    rows, cols = np.ogrid[:height, :width]

    # Calculate distance from center
    distances = np.sqrt((rows - row_center) ** 2 + (cols - col_center) ** 2)

    # Extract pixels within radius
    mask = distances <= radius
    return image[mask]


def extract_annular_region(image, center, inner_radius, outer_radius):
    """Extract pixels within an annular (ring) region.

    Args:
        image: 2D array
        center: Tuple of (row, col) for annulus center
        inner_radius: Inner radius in pixels
        outer_radius: Outer radius in pixels

    Returns:
        1D array of pixel values within the annulus
    """
    row_center, col_center = center
    height, width = image.shape

    # Create coordinate grids
    rows, cols = np.ogrid[:height, :width]

    # Calculate distance from center
    distances = np.sqrt((rows - row_center) ** 2 + (cols - col_center) ** 2)

    # Extract pixels within annulus
    mask = (distances >= inner_radius) & (distances <= outer_radius)
    return image[mask]


def calculate_gcnr(
    reconstruction_sc, gcnr_center, gcnr_radius, gcnr_annulus_inner, gcnr_annulus_outer
):
    """Calculate GCNR between a circular region and an annular background region.

    Args:
        reconstruction_sc: Scan-converted image in dB scale
        gcnr_center: Tuple of (row, col) for region centers
        gcnr_radius: Radius of inner circle in pixels
        gcnr_annulus_inner: Inner radius of annulus in pixels
        gcnr_annulus_outer: Outer radius of annulus in pixels

    Returns:
        float: GCNR value
    """
    # Extract signal region (inner circle)
    signal_pixels = extract_circular_region(reconstruction_sc, gcnr_center, gcnr_radius)

    # Extract background region (annulus)
    background_pixels = extract_annular_region(
        reconstruction_sc, gcnr_center, gcnr_annulus_inner, gcnr_annulus_outer
    )

    # Calculate GCNR using zea.metrics.gcnr
    gcnr_value = gcnr(signal_pixels, background_pixels)

    return gcnr_value


def plot_fwhm_gcnr_comparison(
    data_dir: Path,
    save_dir: Path,
    frame_idx: int = 3,
    point1: tuple = (56, 50),
    point2: tuple = (56, 150),
    gcnr_center: tuple = (70, 70),
    gcnr_radius: float = 10,
    gcnr_annulus_inner: float = 15,
    gcnr_annulus_outer: float = 20,
    rho_max: float = 80.0,
    vmin: float = -60.0,
    vmax: float = 0.0,
    strategies: list = None,
    context="styles/ieee-tmi.mplstyle",
):
    """Plot scan-converted reconstructions with FWHM measurement line and FWHM trace comparison.

    Creates a multi-panel figure with:
    - Left: Colorbar for intensity scale
    - Middle: Scan-converted reconstruction images for each strategy with measurement line overlay
    - Right: Overlaid FWHM traces showing intensity profiles along the measurement line
    - Far Right: GCNR bar plot

    Args:
        data_dir: Directory containing .npz files for each strategy
        save_dir: Directory to save output plots
        frame_idx: Frame index to visualize (default: 3)
        point1: Tuple of (row, col) pixel indices for start point of measurement line
        point2: Tuple of (row, col) pixel indices for end point of measurement line
        gcnr_center: Tuple of (row, col) for GCNR region centers
        gcnr_radius: Radius of inner circle for GCNR in pixels
        gcnr_annulus_inner: Inner radius of annulus for GCNR in pixels
        gcnr_annulus_outer: Outer radius of annulus for GCNR in pixels
        rho_max: Maximum imaging depth in mm (default: 80.0)
        vmin: Minimum intensity value for display in dB (default: -60.0)
        vmax: Maximum intensity value for display in dB (default: 0.0)
        strategies: List of strategy names to plot (default: all strategies)
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
        # Create figure layout: colorbar + images + FWHM plot + GCNR plot
        n_strategies = len(data)
        fig_width = 0.3 + 2 * n_strategies + 4 + 2  # Added 2 for GCNR plot
        fig = plt.figure(figsize=(fig_width, 2.5))

        import matplotlib.gridspec as gridspec

        # Create main grid: [images section, plots section]
        gs_main = gridspec.GridSpec(
            1,
            2,
            width_ratios=[0.05 + n_strategies, 2.5],  # Images vs plots
            wspace=0.05,  # Small space between image section and plot section
        )

        # Create sub-grid for images: [colorbar, image1, image2, ..., imageN]
        gs_images = gridspec.GridSpecFromSubplotSpec(
            1,
            n_strategies + 1,
            subplot_spec=gs_main[0],
            width_ratios=[0.05] + [1] * n_strategies,
            wspace=0.1,  # Tighter spacing between reconstruction images
        )

        # Create sub-grid for plots: [FWHM plot, GCNR plot]
        gs_plots = gridspec.GridSpecFromSubplotSpec(
            1,
            2,
            subplot_spec=gs_main[1],
            width_ratios=[1.5, 1],
            wspace=0.4,  # Wider spacing between FWHM and GCNR plots
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

        # Create colorbar in first column of images grid
        cax = fig.add_subplot(gs_images[0])
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

            ax = fig.add_subplot(gs_images[idx + 1])
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

            # Draw GCNR regions
            # Inner circle (signal region) - cyan dashed
            circle_signal = Circle(
                (gcnr_center[1], gcnr_center[0]),
                gcnr_radius,
                fill=False,
                edgecolor="cyan",
                linestyle="--",
                linewidth=1,
                alpha=0.6,
            )
            ax.add_patch(circle_signal)

            # Inner annulus boundary - yellow dashed
            circle_annulus_inner = Circle(
                (gcnr_center[1], gcnr_center[0]),
                gcnr_annulus_inner,
                fill=False,
                edgecolor="yellow",
                linestyle="--",
                linewidth=1,
                alpha=0.6,
            )
            ax.add_patch(circle_annulus_inner)

            # Outer annulus boundary - yellow dashed
            circle_annulus_outer = Circle(
                (gcnr_center[1], gcnr_center[0]),
                gcnr_annulus_outer,
                fill=False,
                edgecolor="yellow",
                linestyle="--",
                linewidth=1,
                alpha=0.6,
            )
            ax.add_patch(circle_annulus_outer)

            ax.axis("off")

        # Plot FWHM traces (first plot in plots grid)
        ax_fwhm = fig.add_subplot(gs_plots[0])

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
                label=f"{strategy_display} ({fwhm_val_mm:.2f} mm)",
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
            "FWHM",
            fontsize=9,
            fontweight="bold",
        )

        ax_fwhm.grid(True, alpha=0.3)
        ax_fwhm.tick_params(labelsize=7)

        # Adjust FWHM subplot height to match images
        pos = ax_fwhm.get_position()
        new_height = 0.45
        new_bottom = pos.y0 + (pos.height - new_height) / 2
        ax_fwhm.set_position([pos.x0, new_bottom, pos.width, new_height])

        # Plot GCNR bar chart (second plot in plots grid)
        ax_gcnr = fig.add_subplot(gs_plots[1])

        gcnr_values = []
        for strategy in strategies:
            if strategy not in data:
                continue

            reconstruction_sc = reconstructions_sc[strategy]

            # Calculate GCNR
            gcnr_val = calculate_gcnr(
                reconstruction_sc,
                gcnr_center,
                gcnr_radius,
                gcnr_annulus_inner,
                gcnr_annulus_outer,
            )
            gcnr_values.append(gcnr_val)

        # Create bar chart
        x_pos = np.arange(len(strategies))
        colors = [STRATEGY_COLORS.get(s, "#000000") for s in strategies]

        bars = ax_gcnr.bar(x_pos, gcnr_values, color=colors, alpha=0.7)

        # Add GCNR values as text on top of bars
        for i, (bar, val) in enumerate(zip(bars, gcnr_values)):
            height = bar.get_height()
            ax_gcnr.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )

        ax_gcnr.set_ylabel("GCNR", fontsize=9)
        ax_gcnr.set_title("GCNR", fontsize=9, fontweight="bold")
        ax_gcnr.set_xticks([])  # Remove x-axis ticks and labels
        ax_gcnr.grid(True, alpha=0.3, axis="y")
        ax_gcnr.set_ylim([0, max(gcnr_values) * 1.15])  # Add space for text labels

        # Move y-axis to the right side
        ax_gcnr.yaxis.tick_right()
        ax_gcnr.yaxis.set_label_position("right")

        ax_gcnr.tick_params(labelsize=7)

        # Adjust GCNR subplot height to match images
        pos = ax_gcnr.get_position()
        new_height = 0.45
        new_bottom = pos.y0 + (pos.height - new_height) / 2
        ax_gcnr.set_position([pos.x0, new_bottom, pos.width, new_height])

        # Create shared legend to the right of GCNR plot
        # Collect handles and labels from FWHM plot
        handles, labels = ax_fwhm.get_legend_handles_labels()

        # Add GCNR values to labels
        legend_labels = []
        for i, strategy in enumerate(strategies):
            strategy_display = STRATEGY_NAMES.get(strategy, strategy)
            fwhm_val = float(labels[i].split("(")[1].split(" mm")[0])
            gcnr_val = gcnr_values[i]
            legend_labels.append(
                f"{strategy_display}: FWHM={fwhm_val:.2f} mm, GCNR={gcnr_val:.2f}"
            )

        # Place legend to the right of GCNR plot
        ax_gcnr.legend(
            handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.40, 0.5),
            fontsize=6,
            framealpha=0.9,
        )

        # Save figure in multiple formats
        for ext in [".pdf", ".png"]:
            save_file = (
                save_dir
                / f"fwhm_gcnr_comparison_frame{frame_idx}_p1_{point1[0]}_{point1[1]}__p2_{point2[0]}_{point2[1]}{ext}"
            )
            plt.savefig(save_file, dpi=300, bbox_inches="tight")
            log.info(f"Saved FWHM+GCNR comparison to {log.yellow(save_file)}")

        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate FWHM and GCNR comparison plots for CIRS phantom reconstructions"
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
        "--gcnr-center",
        type=int,
        nargs=2,
        default=[41, 81],
        help="Center of GCNR regions as (row, col) in scan-converted image",
    )
    parser.add_argument(
        "--gcnr-radius",
        type=float,
        default=5,
        help="Radius of inner circle for GCNR in pixels (default: 10)",
    )
    parser.add_argument(
        "--gcnr-annulus-inner",
        type=float,
        default=8,
        help="Inner radius of annulus for GCNR in pixels (default: 15)",
    )
    parser.add_argument(
        "--gcnr-annulus-outer",
        type=float,
        default=12,
        help="Outer radius of annulus for GCNR in pixels (default: 20)",
    )
    parser.add_argument(
        "--rho-max",
        type=float,
        default=80.0,
        help="Maximum imaging depth in mm (default: 80.0)",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=-60.0,
        help="Minimum intensity value for display in dB (default: -60.0)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=0.0,
        help="Maximum intensity value for display in dB (default: 0.0)",
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
    log.info("Generating FWHM and GCNR comparison plot...")
    plot_fwhm_gcnr_comparison(
        data_dir=data_dir,
        save_dir=save_dir,
        frame_idx=args.frame_idx,
        point1=tuple(args.point1),
        point2=tuple(args.point2),
        gcnr_center=tuple(args.gcnr_center),
        gcnr_radius=args.gcnr_radius,
        gcnr_annulus_inner=args.gcnr_annulus_inner,
        gcnr_annulus_outer=args.gcnr_annulus_outer,
        rho_max=args.rho_max,
        vmin=args.vmin,
        vmax=args.vmax,
        strategies=args.strategies,
        context="styles/ieee-tmi.mplstyle",
    )

    log.info("Done!")
