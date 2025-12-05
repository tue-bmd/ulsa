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
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

from zea import init_device, log
from zea.ops import translate

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "numpy"
    init_device("cpu")
    sys.path.append("/ulsa")

from skimage import exposure

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


def get_arrow_patch(row_tip, col_tip, length=30, angle_deg=45):
    """Create a FancyArrowPatch for matplotlib at the specified tip, length, and angle."""
    import math

    angle_rad = math.radians(angle_deg)
    # Calculate tail position
    row_tail = row_tip - length * math.sin(angle_rad)
    col_tail = col_tip - length * math.cos(angle_rad)
    arrow_kwargs = {
        "color": "purple",
        "arrowstyle": "->",
        "mutation_scale": 12,
        "linewidth": 2,
    }
    return FancyArrowPatch(
        (col_tail, row_tail),  # tail (x, y)
        (col_tip, row_tip),  # tip (x, y)
        **arrow_kwargs,
    )


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


def plot_fwhm_gcnr(
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
    context="styles/ieee-tmi.mplstyle",
    arrow_tip=None,
    arrow_length=30,
    arrow_angle=45,
    overlays_on_all=False,  # New parameter
):
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    strategies_row1 = ["equispaced", "uniform_random", "greedy_entropy"]
    strategies_row2 = ["focused", "diverging"]
    all_strategies = strategies_row1 + strategies_row2

    # Load data
    data = {}
    theta_range = None
    for strategy in all_strategies:
        npz_path = data_dir / f"{strategy}.npz"
        if npz_path.exists():
            npz_data = np.load(str(npz_path))
            data[strategy] = npz_data
            if theta_range is None and "theta_range" in npz_data:
                theta_range = npz_data["theta_range"]
        else:
            print(f"File not found: {npz_path}")
    if theta_range is None:
        theta_range = (-0.78539816, 0.78539816)
    scan_conversion_angles = (np.rad2deg(theta_range[0]), np.rad2deg(theta_range[1]))

    # Always use greedy_entropy as reference for histogram matching
    ref_data = data["greedy_entropy"]
    if "greedy_entropy" not in ORIGINAL_ACQUISITONS:
        reference_image = translate(
            np.clip(ref_data["reconstructions"][frame_idx], -1, 1),
            (-1, 1),
            (-60, 0),
        )
    else:
        reference_image = ref_data["reconstructions"][frame_idx]
    reference_image = np.clip(reference_image, vmin, vmax)

    # Prepare reconstructions
    reconstructions_raw = {}
    reconstructions_sc = {}
    for strategy in all_strategies:
        strategy_data = data[strategy]
        if strategy not in ORIGINAL_ACQUISITONS:
            reconstruction = translate(
                np.clip(strategy_data["reconstructions"][frame_idx], -1, 1),
                (-1, 1),
                (-60, 0),
            )
        else:
            reconstruction = strategy_data["reconstructions"][frame_idx]

        # Histogram match to greedy_entropy reference
        reconstruction = exposure.match_histograms(reconstruction, reference_image)

        reconstructions_raw[strategy] = np.clip(reconstruction, vmin, vmax)
        reconstructions_sc[strategy] = _scan_convert(
            reconstructions_raw[strategy],
            scan_conversion_angles=scan_conversion_angles,
            fill_value=np.nan,
            order=0,
        )

    # Compute FWHM and GCNR for each strategy
    fwhm_vals = {}
    gcnr_vals = {}
    for strategy in all_strategies:
        reconstruction_sc = reconstructions_sc[strategy]
        trace_db, distances_mm, fwhm_val_mm = calculate_fwhm(
            point1, point2, reconstruction_sc, rho_max
        )
        fwhm_vals[strategy] = fwhm_val_mm
        gcnr_vals[strategy] = calculate_gcnr(
            reconstruction_sc,
            gcnr_center,
            gcnr_radius,
            gcnr_annulus_inner,
            gcnr_annulus_outer,
        )

    # Plotting
    with plt.style.context(context):
        fig = plt.figure(
            figsize=(7.16 / 2.0, 2.2)
        )  # Single column width, slightly taller
        gs = gridspec.GridSpec(
            2,
            3,
            figure=fig,
            wspace=0.04,
            hspace=0.08,
            left=0.0,
            right=1.0,
            top=0.98,
            bottom=0.02,
        )

        # Titles for each strategy
        titles = {
            "greedy_entropy": "Active Perception (11/90)",
            "uniform_random": "Random (11/90)",
            "equispaced": "Equispaced (11/90)",
            "focused": "Focused (90)",
            "diverging": "Diverging (11)",
        }

        # Plot first row: reconstructions
        text_positions_row1 = []  # Store positions for later
        for col, strategy in enumerate(strategies_row1):
            ax = fig.add_subplot(gs[0, col])
            im = ax.imshow(
                reconstructions_sc[strategy],
                cmap="gray",
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
                origin="upper",
                interpolation="nearest",
            )
            ax.set_title(titles[strategy], fontsize=9, pad=2)
            ax.axis("off")

            # Show overlays on equispaced only, or on all if overlays_on_all is True
            if strategy == "equispaced" or overlays_on_all:
                # Draw FWHM measurement line overlay (thin dashed red, no dots)
                ax.plot(
                    [point1[1], point2[1]],
                    [point1[0], point2[0]],
                    "r-",
                    linewidth=0.8,
                    alpha=0.7,
                )

                # Draw GCNR regions
                circle_signal = Circle(
                    (gcnr_center[1], gcnr_center[0]),
                    gcnr_radius,
                    fill=False,
                    edgecolor="cyan",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                )
                ax.add_patch(circle_signal)

                circle_annulus_inner = Circle(
                    (gcnr_center[1], gcnr_center[0]),
                    gcnr_annulus_inner,
                    fill=False,
                    edgecolor="yellow",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                )
                ax.add_patch(circle_annulus_inner)
                circle_annulus_outer = Circle(
                    (gcnr_center[1], gcnr_center[0]),
                    gcnr_annulus_outer,
                    fill=False,
                    edgecolor="yellow",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                )
                ax.add_patch(circle_annulus_outer)

            # Draw arrow on all first row images
            if arrow_tip is not None:
                arrow_patch = get_arrow_patch(
                    row_tip=arrow_tip[0],
                    col_tip=arrow_tip[1],
                    length=arrow_length,
                    angle_deg=arrow_angle,
                )
                ax.add_patch(arrow_patch)

            # Store text position for later (in figure coordinates)
            text_positions_row1.append((ax, strategy))

        # Plot second row: focused, diverging, FWHM plot
        text_positions_row2 = []  # Store positions for later
        for col, strategy in enumerate(strategies_row2):
            ax = fig.add_subplot(gs[1, col])
            im = ax.imshow(
                reconstructions_sc[strategy],
                cmap="gray",
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
                origin="upper",
                interpolation="nearest",
            )
            ax.set_title(titles[strategy], fontsize=9, pad=2)
            ax.axis("off")

            # Show overlays on all second row images if overlays_on_all is True
            if overlays_on_all:
                # Draw FWHM measurement line overlay (thin dashed red, no dots)
                ax.plot(
                    [point1[1], point2[1]],
                    [point1[0], point2[0]],
                    "r-",
                    linewidth=0.8,
                    alpha=0.7,
                )

                # Draw GCNR regions
                circle_signal = Circle(
                    (gcnr_center[1], gcnr_center[0]),
                    gcnr_radius,
                    fill=False,
                    edgecolor="cyan",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                )
                ax.add_patch(circle_signal)

                circle_annulus_inner = Circle(
                    (gcnr_center[1], gcnr_center[0]),
                    gcnr_annulus_inner,
                    fill=False,
                    edgecolor="yellow",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                )
                ax.add_patch(circle_annulus_inner)
                circle_annulus_outer = Circle(
                    (gcnr_center[1], gcnr_center[0]),
                    gcnr_annulus_outer,
                    fill=False,
                    edgecolor="yellow",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                )
                ax.add_patch(circle_annulus_outer)

            # Store text position for later
            text_positions_row2.append((ax, strategy))

        # FWHM plot in last column, second row
        ax_fwhm = fig.add_subplot(gs[1, 2])
        for strategy in all_strategies:
            trace_db, distances_mm, _ = calculate_fwhm(
                point1, point2, reconstructions_sc[strategy], rho_max
            )
            color = STRATEGY_COLORS.get(strategy, "#000000")
            label = STRATEGY_NAMES.get(strategy, strategy)
            ax_fwhm.plot(
                distances_mm,
                trace_db,
                linewidth=1.2,
                color=color,
                label=label,
                marker="",
            )
        ax_fwhm.axvline(x=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.3)
        ax_fwhm.set_ylabel("Intensity [dB]", fontsize=6, labelpad=1)
        ax_fwhm.set_xlabel("Distance [mm]", fontsize=6, labelpad=0.5)
        ax_fwhm.set_ylim(-50, 5)
        ax_fwhm.yaxis.tick_right()
        ax_fwhm.yaxis.set_label_position("right")
        ax_fwhm.set_title("FWHM", fontsize=9, fontweight="bold", pad=2)
        ax_fwhm.grid(True, alpha=0.3)
        ax_fwhm.tick_params(labelsize=6, pad=1)
        ax_fwhm.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.5),
            fontsize=6,
            framealpha=0.9,
            ncol=2,
            columnspacing=0.8,  # Reduce horizontal spacing between columns
            handletextpad=0.4,
        )

        # Adjust FWHM plot position
        pos = ax_fwhm.get_position()
        ax_fwhm.set_position(
            [
                pos.x0 + 0.055,
                pos.y0 + 0.225,
                pos.width * 0.7,
                pos.height * 0.4,
            ]
        )

        # NOW add text using figure coordinates (after layout is finalized)
        # This makes the text completely independent of axes layout
        for ax, strategy in text_positions_row1:
            # Get axes position in figure coordinates
            bbox = ax.get_position()
            # Position text to the right of the axes
            fig.text(
                bbox.x1 - 0.08,  # Slightly to the right of axes
                bbox.y1 - 0.04,  # Near top of axes
                f"FWHM: {fwhm_vals[strategy]:.2f}\nGCNR: {gcnr_vals[strategy]:.2f}",
                fontsize=6,
                color="black",
                ha="left",
                va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
            )

        for ax, strategy in text_positions_row2:
            bbox = ax.get_position()
            fig.text(
                bbox.x1 - 0.08,
                bbox.y1 - 0.04,
                f"FWHM: {fwhm_vals[strategy]:.2f}\nGCNR: {gcnr_vals[strategy]:.2f}",
                fontsize=6,
                color="black",
                ha="left",
                va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
            )

        # Save with minimal padding
        for ext in [".pdf", ".png"]:
            save_file = (
                save_dir
                / f"fwhm_single_column_frame{frame_idx}_p1_{point1[0]}_{point1[1]}__p2_{point2[0]}_{point2[1]}{ext}"
            )
            plt.savefig(save_file, dpi=300, bbox_inches="tight", pad_inches=0.02)
            log.info(f"Saved single-column FWHM plot to {log.yellow(save_file)}")
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
    parser.add_argument(
        "--arrow-tip",
        type=int,
        nargs=2,
        default=None,
        help="Arrow tip position (row, col) in scan-converted image (default: None, disables arrow)",
    )
    parser.add_argument(
        "--arrow-length",
        type=float,
        default=35,
        help="Arrow length in pixels (default: 30)",
    )
    parser.add_argument(
        "--arrow-angle",
        type=float,
        default=45,
        help="Arrow angle in degrees (default: 45, down and to the right)",
    )
    parser.add_argument(
        "--overlays-on-all",
        action="store_true",
        help="Show FWHM line and GCNR circles on all top row images (default: only on equispaced)",
    )
    args = parser.parse_args()

    # Create save directory if it doesn't exist
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)

    # Generate comparison plot
    log.info("Generating FWHM and GCNR comparison plot...")
    plot_fwhm_gcnr(
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
        context="styles/ieee-tmi.mplstyle",
        arrow_tip=tuple(args.arrow_tip) if args.arrow_tip is not None else None,
        arrow_length=args.arrow_length,
        arrow_angle=args.arrow_angle,
        overlays_on_all=args.overlays_on_all,
    )

    log.info("Done!")
