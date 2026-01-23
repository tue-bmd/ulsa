"""
This script makes a figure with the target, reconstruction, measurements and entropy
for the in-house cardiac dataset.
"""

import os

import matplotlib.pyplot as plt

import zea

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "jax"
    zea.init_device()
    plt.rcdefaults()

import copy
import math
from pathlib import Path

import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyArrowPatch

from ulsa.in_house_cardiac.load_results import load_from_run_dir
from ulsa.transmit_time import max_fps

imshow_kwargs = {
    "vmin": 0,
    "vmax": 255,
    "cmap": "gray",
    "interpolation": "nearest",
}


def _init_grid(
    n_rows,
    grid_columns=3,
    wspace=0.04,
    hspace=0.07,
    left_margin=0.0,
    right_margin=1.0,
    top_margin=0.95,
    fig_height_per_row=1.6,
):
    fig = plt.figure(figsize=(7.16, fig_height_per_row * n_rows))

    grid_shape = (n_rows, grid_columns)

    outer = gridspec.GridSpec(
        *grid_shape,
        figure=fig,
        wspace=wspace,
        hspace=hspace,
        left=left_margin,
        right=right_margin,
        top=top_margin,  # <1.0 to leave space for titles
        bottom=0.0,
        width_ratios=[1.0, 1.0, 1.45],
    )

    return fig, outer


def get_arrow(
    x_tip=880,
    y_tip=650,
    length=310,
    angle_deg=-20 + 90,
):
    """Create an arrow patch with the specified parameters.

    # Arrow tip (x, y)
    # larger y will be lower on the plot
    # bigger x will be further right on the plot
    """

    angle_rad = math.radians(angle_deg)

    # Calculate tail position
    x_tail = x_tip - length * math.cos(angle_rad)
    y_tail = y_tip - length * math.sin(angle_rad)

    arrow_kwargs = {
        "color": "purple",
        "arrowstyle": "->",
        "mutation_scale": 15,
        "linewidth": 3,
    }

    return FancyArrowPatch(
        (x_tail, y_tail),  # tail
        (x_tip, y_tip),  # tip
        **arrow_kwargs,
    )


def tilted_text(ax, text, fontsize=7, rotation=45, color="gray", **kwargs):
    ax.text(
        0.0,  # left=0.0
        0.5,  # upper=1.0
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        rotation=rotation,
        color=color,
        **kwargs,
    )


def stack_plot_from_npz(
    run_dirs: list,
    plot_dir: str | Path,
    exts=(".png", ".pdf"),
    context=None,
    frame_indices: list | None = None,
    arrows: list | None = None,
    ylabels: list | None = None,
    scan_convert_resolution=0.1,
    selection_strategy="greedy_entropy",
    wspace_inner=-0.1,
    hspace_inner=0.02,
    left_margin=0.03,  # space for ylabels
    top_margin=0.95,  # space for titles
    hspace=0.0,
    fig_height_per_row=1.5,
):
    plot_dir = Path(plot_dir)
    plot_path = plot_dir / f"qualitative_in_house_{selection_strategy}"

    if context is None:
        context = "styles/darkmode.mplstyle"

    # constants
    grid_columns = 3
    inner_grid_shape = (2, 2)

    with plt.style.context([context, {"figure.constrained_layout.use": False}]):
        fig, outer = _init_grid(
            len(run_dirs),
            grid_columns=grid_columns,
            left_margin=left_margin,
            top_margin=top_margin,
            hspace=hspace,
            fig_height_per_row=fig_height_per_row,
        )
        for row_idx, run_dir in enumerate(run_dirs):
            is_fundamental = "fundamental" in str(run_dir)
            is_harmonic = "harmonic" in str(run_dir)
            assert is_fundamental or is_harmonic, (
                "Run dir should contain 'fundamental' or 'harmonic' to identify modality."
            )

            (
                focused,
                diverging,
                reconstructions,
                measurements,
                entropy,
                n_actions,
                n_possible_actions,
                _,
                frame_idx,
            ) = load_from_run_dir(
                run_dir,
                frame_idx=frame_indices[row_idx] if frame_indices is not None else None,
                selection_strategy=selection_strategy,
                scan_convert_resolution=scan_convert_resolution,
            )

            if is_harmonic:
                n_possible_actions *= 2
                n_actions *= 2

            arrow = arrows[row_idx] if arrows is not None else None

            ax = fig.add_subplot(outer[row_idx, 0])
            ax.imshow(focused[frame_idx], **imshow_kwargs)
            if row_idx == 0:
                ax.set_title(f"Focused")
            if ylabels is not None and row_idx < len(ylabels):
                ax.set_ylabel(ylabels[row_idx])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
            else:
                ax.axis("off")
            tilted_text(ax, f"{n_possible_actions} transmits")

            ax = fig.add_subplot(outer[row_idx, 1])
            ax.imshow(diverging[frame_idx], **imshow_kwargs)
            if row_idx == 0:
                ax.set_title("Diverging (HFR)")
            ax.axis("off")
            if arrow is not None:
                ax.add_patch(copy.copy(arrow))
            tilted_text(ax, f"{n_actions} transmits")

            inner = gridspec.GridSpecFromSubplotSpec(
                *inner_grid_shape,
                subplot_spec=outer[row_idx, 2],
                width_ratios=[2, 1],
                height_ratios=[1, 1],
                wspace=wspace_inner,
                hspace=hspace_inner * inner_grid_shape[0],
            )

            ax_big = fig.add_subplot(inner[:, 0])
            ax_big.imshow(reconstructions[frame_idx], **imshow_kwargs)
            if row_idx == 0:
                ax_big.set_title("Cognitive (HFR)")
            ax_big.axis("off")
            if arrow is not None:
                ax_big.add_patch(copy.copy(arrow))
            tilted_text(ax_big, f"{n_actions}/{n_possible_actions} transmits")

            ax_bottom = fig.add_subplot(inner[1, 1])
            ax_bottom.imshow(measurements[frame_idx], **imshow_kwargs)
            ax_bottom.axis("off")

            ax_top = fig.add_subplot(inner[0, 1])
            ax_top.imshow(
                entropy,
                cmap="inferno",
                vmin=0,
                vmax=255,
                interpolation="nearest",
            )
            ax_top.axis("off")

        for ext in exts:
            plt.savefig(plot_path.with_suffix(ext))
            zea.log.info(
                f"Saved cardiac reconstruction plot to {zea.log.yellow(plot_path.with_suffix(ext))}"
            )


def animated_plot_from_npz(
    run_dir: str,
    plot_dir: str | Path,
    scan_convert_resolution=0.5,
    selection_strategy="greedy_entropy",
    file_type="gif",
    fill_value="black",
    no_measurement_color="gray",
    drop_first_n_frames=10,
    fps=None,
):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    name = Path(run_dir).name

    is_fundamental = "fundamental" in str(run_dir)
    is_harmonic = "harmonic" in str(run_dir)
    assert is_fundamental or is_harmonic, (
        "Run dir should contain 'fundamental' or 'harmonic' to identify modality."
    )

    (
        focused,
        diverging,
        reconstructions,
        measurements,
        entropy,
        n_actions,
        n_possible_actions,
        _,
        _,
    ) = load_from_run_dir(
        run_dir,
        frame_idx=None,
        selection_strategy=selection_strategy,
        scan_convert_resolution=scan_convert_resolution,
        fill_value=fill_value,
        no_measurement_color=no_measurement_color,
        drop_first_n_frames=drop_first_n_frames,
    )

    if is_harmonic:
        n_possible_actions *= 2
        n_actions *= 2

    if fps is None:
        fps = max_fps(n_tx=n_possible_actions + n_actions, processing_overhead=1.14)
        print(f"Saving animations at {fps:.2f} FPS")

    axis = -1  # concat along width
    comparison = np.concatenate([focused, reconstructions, diverging], axis=axis)
    zea.io_lib.save_video(
        comparison,
        plot_dir / f"target_reconstruction_diverging_{name}.{file_type}",
        fps=fps,
    )

    measurements_reconstruction = np.concatenate(
        [measurements, reconstructions], axis=axis
    )
    zea.io_lib.save_video(
        measurements_reconstruction,
        plot_dir / f"measurements_reconstruction_{name}.{file_type}",
        fps=fps,
    )


if __name__ == "__main__":
    fundamental_file = (
        "/mnt/z/usbmd/ulsa/eval_in_house/cardiac_fundamental/20240701_P1_A4CH_0001"
    )
    harmonic_dir = Path("/mnt/z/usbmd/ulsa/eval_in_house/cardiac_harmonic/")

    stack_plot_from_npz(
        [
            harmonic_dir / "20251222_s3_a4ch_line_dw_0000",
            harmonic_dir / "20251222_s1_a4ch_line_dw_0000",
            fundamental_file,
        ],
        "output/in_house_cardiac",
        context="styles/ieee-tmi.mplstyle",
        frame_indices=[
            68,
            10,
            24,
        ],
        arrows=[None, None, None],
        ylabels=[
            "Harmonic",
            "Harmonic",
            "Fundamental",
        ],
        selection_strategy="greedy_entropy",
        scan_convert_resolution=0.1,
    )

    harmonic_files = [f for f in harmonic_dir.iterdir() if f.is_dir()]
    for harmonic_file in harmonic_files:
        animated_plot_from_npz(
            harmonic_file,
            "output/in_house_cardiac/animations",
            scan_convert_resolution=0.2,
            file_type="gif",
        )
