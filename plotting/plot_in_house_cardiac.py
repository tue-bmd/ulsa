"""
This script makes a figure with the target, reconstruction, and measurements
for the in-house cardiac dataset.

# diverging_dynamic_range = [-70, -30]
"""

import os
import sys

import zea

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "jax"
    zea.init_device()
    sys.path.append("/ulsa")

import copy
import math
from pathlib import Path

import jax.numpy as jnp
import keras
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from ulsa.entropy import pixelwise_entropy
from ulsa.io_utils import color_to_value, postprocess_agent_results, side_by_side_gif


def plot_from_npz(
    run_dir: Path | str,
    plot_dir: Path | str,
    exts=(".png", ".pdf"),
    gif=True,
    gif_fps=8,
    context=None,
    frame_idx=24,
    arrow=None,
    diverging_dynamic_range=None,
    focused_dynamic_range=None,
    scan_convert_resolution=0.1,
    selection_strategy="greedy_entropy",
):
    run_dir = Path(run_dir)
    plot_dir = Path(plot_dir)
    plot_path = plot_dir / selection_strategy

    focused_results = np.load(run_dir / "focused.npz", allow_pickle=True)
    diverging_results = np.load(run_dir / "diverging.npz", allow_pickle=True)
    results = np.load(run_dir / f"{selection_strategy}.npz", allow_pickle=True)
    n_actions = results["n_actions"].item()
    n_possible_actions = results["n_possible_actions"].item()

    focused = focused_results["reconstructions"]
    diverging = diverging_results["reconstructions"]

    if focused_dynamic_range is None:
        focused_dynamic_range = focused_results["dynamic_range"]

    if diverging_dynamic_range is None:
        diverging_dynamic_range = diverging_results["dynamic_range"]

    reconstructions = results["reconstructions"]
    reconstruction_range = results["dynamic_range"]

    measurements = results["measurements"]
    masks = results["masks"]
    belief_distributions = results["belief_distributions"]

    measurements = keras.ops.where(
        masks > 0, measurements, color_to_value(reconstruction_range, "gray")
    )

    io_config = zea.Config(
        scan_convert=True,
        scan_conversion_angles=np.rad2deg(results["theta_range"]),
    )

    if not gif:
        focused = focused[frame_idx, None]
        diverging = diverging[frame_idx, None]
        reconstructions = reconstructions[frame_idx, None]
        measurements = measurements[frame_idx, None]
        belief_distributions = belief_distributions[frame_idx, None]
        frame_idx = 0

    print("Postprocessing focused...")
    focused = postprocess_agent_results(
        focused,
        io_config,
        scan_convert_order=0,
        image_range=focused_dynamic_range,
        fill_value="transparent",
        scan_convert_resolution=scan_convert_resolution,
    )
    print("Postprocessing diverging...")
    diverging = postprocess_agent_results(
        diverging,
        io_config,
        scan_convert_order=0,
        image_range=diverging_dynamic_range,
        fill_value="transparent",
        scan_convert_resolution=scan_convert_resolution,
    )
    print("Postprocessing reconstructions...")
    reconstructions = postprocess_agent_results(
        reconstructions,
        io_config,
        scan_convert_order=0,
        image_range=reconstruction_range,
        reconstruction_sharpness_std=0.02,
        fill_value="transparent",
        scan_convert_resolution=scan_convert_resolution,
    )
    print("Postprocessing measurements...")
    measurements = postprocess_agent_results(
        measurements,
        io_config,
        scan_convert_order=0,
        image_range=reconstruction_range,
        fill_value="transparent",
        scan_convert_resolution=scan_convert_resolution,
    )

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
                "Focused (90)",
                f"Reconstruction ({n_actions}/{n_possible_actions})",
                "Diverging (11)",
            ],
            context=context,
            fps=gif_fps,
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
    )

    with plt.style.context([context, {"figure.constrained_layout.use": False}]):
        kwargs = {
            "vmin": 0,
            "vmax": 255,
            "cmap": "gray",
            "interpolation": "nearest",
        }

        grid_shape = (1, 3)
        fig = plt.figure(figsize=(7.16, 1.6))
        wspace = 0.04
        wspace_inner = -0.1
        hspace = 0.07
        hspace_inner = 0.02
        inner_grid_shape = (2, 2)

        outer = gridspec.GridSpec(
            *grid_shape,
            figure=fig,
            wspace=wspace,
            hspace=hspace,
            left=0.0,
            right=1.0,
            top=0.88,  # <1.0 to leave space for titles
            bottom=0.0,
            width_ratios=[1.0, 1.0, 1.42],
        )

        ax = fig.add_subplot(outer[0])
        ax.imshow(focused[frame_idx], **kwargs)
        ax.set_title("Focused (90)")
        ax.axis("off")

        ax = fig.add_subplot(outer[1])
        ax.imshow(diverging[frame_idx], **kwargs)
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
        # ax = fig.add_subplot(outer[2])
        # ax.axis("off")
        # ax.set_title(f"Reconstruction ({n_actions}/{n_possible_actions})")

        ax_big = fig.add_subplot(inner[:, 0])
        ax_big.imshow(reconstructions[frame_idx], **kwargs)
        ax_big.set_title(f"Reconstruction ({n_actions}/{n_possible_actions})")
        ax_big.axis("off")
        if arrow is not None:
            ax_big.add_patch(copy.copy(arrow))

        ax_bottom = fig.add_subplot(inner[1, 1])
        ax_bottom.imshow(measurements[frame_idx], **kwargs)
        # ax_top.set_title(f"Acquisitions ({n_actions}/{n_possible_actions})")
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
        # ax_top.set_title("Entropy")

        for ext in exts:
            plt.savefig(plot_path.with_suffix(ext))
            zea.log.info(
                f"Saved cardiac reconstruction plot to {zea.log.yellow(plot_path.with_suffix(ext))}"
            )


def get_arrow(
    x_tip=880,
    y_tip=650,
    length=310,
    angle_deg=20 + 90,
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


if __name__ == "__main__":
    # PLOT_NPZ_PATH = (
    #     "/mnt/z/usbmd/Wessel/ulsa_paper_plots/20240701_P1_A4CH_0001_results.npz"
    # )
    PLOT_NPZ_PATH = (
        "/mnt/z/usbmd/Wessel/eval_in_house_cardiac_v3/20251222_s1_a4ch_line_dw_0000"
    )

    plot_from_npz(
        PLOT_NPZ_PATH,
        "output/in_house_cardiac",
        context="styles/ieee-tmi.mplstyle",
        frame_idx=24,
        arrow=get_arrow(),
        gif=False,
    )
