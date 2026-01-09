"""
This script makes a figure with the target, reconstruction, and measurements
for the in-house cardiac dataset.
"""

import os
import sys

import matplotlib.pyplot as plt

import zea

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "jax"
    zea.init_device()
    sys.path.append("/ulsa")
    plt.rcdefaults()

import copy
import math
from pathlib import Path

import jax.numpy as jnp
import keras
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyArrowPatch
from skimage.exposure import match_histograms

from ulsa.entropy import pixelwise_entropy
from ulsa.io_utils import color_to_value, postprocess_agent_results, side_by_side_gif

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
):
    fig = plt.figure(figsize=(7.16, 1.6 * n_rows))

    grid_shape = (n_rows, grid_columns)

    outer = gridspec.GridSpec(
        *grid_shape,
        figure=fig,
        wspace=wspace,
        hspace=hspace,
        left=left_margin,
        right=right_margin,
        top=0.95,  # <1.0 to leave space for titles
        bottom=0.0,
        width_ratios=[1.0, 1.0, 1.45],
    )

    return fig, outer


def _load_from_run_dir(
    run_dir: Path | str,
    frame_idx=None,
    selection_strategy="greedy_entropy",
    scan_convert_resolution=0.1,
    dynamic_range=None,
):
    run_dir = Path(run_dir)

    focused_results = np.load(run_dir / "focused.npz", allow_pickle=True)
    diverging_results = np.load(run_dir / "diverging.npz", allow_pickle=True)
    results = np.load(run_dir / f"{selection_strategy}.npz", allow_pickle=True)
    n_actions = results["n_actions"].item()
    n_possible_actions = results["n_possible_actions"].item()

    # Load into variables
    focused = focused_results["reconstructions"]
    diverging = diverging_results["reconstructions"]
    reconstructions = results["reconstructions"]
    measurements = results["measurements"]
    masks = results["masks"]
    belief_distributions = results["belief_distributions"]

    # Drop to single frame if selected
    if frame_idx is not None:
        focused = focused[frame_idx, None]
        diverging = diverging[frame_idx, None]
        reconstructions = reconstructions[frame_idx, None]
        measurements = measurements[frame_idx, None]
        masks = masks[frame_idx, None]
        belief_distributions = belief_distributions[frame_idx, None]
        frame_idx = 0

    # histogram match diverging to focused
    match_histograms_vectorized = np.vectorize(
        match_histograms, signature="(n,m),(n,m)->(n,m)"
    )
    diverging = match_histograms_vectorized(diverging, focused)

    # histogram match reconstructions to focused
    reconstructions = match_histograms_vectorized(reconstructions, focused)
    reconstruction_range = results["dynamic_range"]

    if dynamic_range is None:
        dynamic_range = focused_results["dynamic_range"]

    measurements = keras.ops.where(
        masks > 0, measurements, color_to_value(reconstruction_range, "gray")
    )

    io_config = zea.Config(
        scan_convert=True,
        scan_conversion_angles=np.rad2deg(results["theta_range"]),
    )

    print("Postprocessing focused...")
    focused = postprocess_agent_results(
        focused,
        io_config,
        scan_convert_order=0,
        image_range=dynamic_range,
        fill_value="transparent",
        scan_convert_resolution=scan_convert_resolution,
        distance_to_apex=7.0,  # pixels
    )
    print("Postprocessing diverging...")
    diverging = postprocess_agent_results(
        diverging,
        io_config,
        scan_convert_order=0,
        image_range=dynamic_range,
        fill_value="transparent",
        scan_convert_resolution=scan_convert_resolution,
        distance_to_apex=7.0,  # pixels
    )
    print("Postprocessing reconstructions...")
    reconstructions = postprocess_agent_results(
        reconstructions,
        io_config,
        scan_convert_order=0,
        image_range=dynamic_range,
        reconstruction_sharpness_std=0.02,
        fill_value="transparent",
        scan_convert_resolution=scan_convert_resolution,
        distance_to_apex=7.0,  # pixels
    )
    print("Postprocessing measurements...")
    measurements = postprocess_agent_results(
        measurements,
        io_config,
        scan_convert_order=0,
        image_range=reconstruction_range,
        fill_value="transparent",
        scan_convert_resolution=scan_convert_resolution,
        distance_to_apex=7.0,  # pixels
    )

    print("Postprocessing entropy...")
    belief_distributions = belief_distributions[frame_idx]
    entropy = jnp.squeeze(
        pixelwise_entropy(belief_distributions[None], entropy_sigma=255), axis=0
    )
    entropy = postprocess_agent_results(
        entropy,
        io_config=io_config,
        scan_convert_order=1,
        image_range=[0, jnp.nanpercentile(entropy, 98.5)],
        fill_value="transparent",
        scan_convert_resolution=scan_convert_resolution,
        distance_to_apex=7.0,  # pixels
    )

    return (
        focused,
        diverging,
        reconstructions,
        measurements,
        entropy,
        n_actions,
        n_possible_actions,
        frame_idx,
    )


def plot_from_npz(
    run_dir: Path | str,
    plot_dir: Path | str,
    exts=(".png", ".pdf"),
    gif=True,
    gif_fps=8,
    context=None,
    frame_idx=24,
    arrow=None,
    dynamic_range=None,
    scan_convert_resolution=0.1,
    selection_strategy="greedy_entropy",
):
    (
        focused,
        diverging,
        reconstructions,
        measurements,
        entropy,
        n_actions,
        n_possible_actions,
        frame_idx,
    ) = _load_from_run_dir(
        run_dir,
        frame_idx=frame_idx,
        selection_strategy=selection_strategy,
        scan_convert_resolution=scan_convert_resolution,
        dynamic_range=dynamic_range,
    )

    plot_dir = Path(plot_dir)
    plot_path = plot_dir / selection_strategy

    if context is None:
        context = "styles/darkmode.mplstyle"

    if gif:
        print("Creating GIF...")
        side_by_side_gif(
            plot_path.with_suffix(".gif"),
            focused,
            reconstructions,
            diverging,
            labels=[
                f"Focused ({n_possible_actions})",
                f"Reconstruction ({n_actions}/{n_possible_actions})",
                "Diverging (11)",
            ],
            context=context,
            fps=gif_fps,
        )

    with plt.style.context([context, {"figure.constrained_layout.use": False}]):
        fig, outer = _init_grid(1)
        wspace_inner = -0.1
        hspace_inner = 0.02
        inner_grid_shape = (2, 2)

        ax = fig.add_subplot(outer[0])
        ax.imshow(focused[frame_idx], **imshow_kwargs)
        ax.set_title(f"Focused ({n_possible_actions})")
        ax.axis("off")

        ax = fig.add_subplot(outer[1])
        ax.imshow(diverging[frame_idx], **imshow_kwargs)
        ax.set_title("Diverging (11)")
        ax.axis("off")
        if arrow is not None:
            ax.add_patch(copy.copy(arrow))

        inner = gridspec.GridSpecFromSubplotSpec(
            *inner_grid_shape,
            subplot_spec=outer[2],
            width_ratios=[2, 1],
            height_ratios=[1, 1],
            wspace=wspace_inner,
            hspace=hspace_inner * inner_grid_shape[0],
        )

        ax_big = fig.add_subplot(inner[:, 0])
        ax_big.imshow(reconstructions[frame_idx], **imshow_kwargs)
        ax_big.set_title(f"Reconstruction ({n_actions}/{n_possible_actions})")
        ax_big.axis("off")
        if arrow is not None:
            ax_big.add_patch(copy.copy(arrow))

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
):
    plot_dir = Path(plot_dir)
    plot_path = plot_dir / f"qualitative_in_house_{selection_strategy}"

    if context is None:
        context = "styles/darkmode.mplstyle"

    with plt.style.context([context, {"figure.constrained_layout.use": False}]):
        grid_columns = 3
        wspace_inner = -0.1
        hspace_inner = 0.02
        inner_grid_shape = (2, 2)
        # left_margin = 0.08 if ylabels is not None else 0.0
        fig, outer = _init_grid(
            len(run_dirs), grid_columns=grid_columns, left_margin=0.03
        )
        for row_idx, run_dir in enumerate(run_dirs):
            (
                focused,
                diverging,
                reconstructions,
                measurements,
                entropy,
                n_actions,
                n_possible_actions,
                frame_idx,
            ) = _load_from_run_dir(
                run_dir,
                frame_idx=frame_indices[row_idx] if frame_indices is not None else None,
                selection_strategy=selection_strategy,
                scan_convert_resolution=scan_convert_resolution,
            )

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
                ax.set_title("Diverging")
            ax.axis("off")
            if arrow is not None:
                ax.add_patch(copy.copy(arrow))
            tilted_text(ax, "11 transmits")

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
                ax_big.set_title(f"Reconstruction")
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


if __name__ == "__main__":
    # plot_from_npz(
    #     "/mnt/z/usbmd/Wessel/ulsa/eval_in_house/cardiac_fundamental/20240701_P1_A4CH_0001",
    #     "output/in_house_cardiac",
    #     context="styles/ieee-tmi.mplstyle",
    #     arrow=get_arrow(),
    #     gif=False,
    # )

    fundamental_file = "/mnt/z/usbmd/Wessel/ulsa/eval_in_house/cardiac_fundamental/20240701_P1_A4CH_0001"
    harmonic_file = "/mnt/z/usbmd/Wessel/ulsa/eval_in_house/cardiac_harmonic/20251222_s3_a4ch_line_dw_0000"
    frame_indices = [24, 68]
    arrows = [get_arrow(), None]
    ylabels = ["Fundamental", "Harmonic"]
    stack_plot_from_npz(
        [fundamental_file, harmonic_file],
        "output/in_house_cardiac",
        context="styles/ieee-tmi.mplstyle",
        frame_indices=frame_indices,
        arrows=arrows,
        ylabels=ylabels,
        selection_strategy="greedy_entropy",
        scan_convert_resolution=0.1,
    )
