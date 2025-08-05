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

import keras
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from ulsa.io_utils import color_to_value, postprocess_agent_results, side_by_side_gif


def plot_from_npz(
    path,
    save_path,
    exts=(".png", ".pdf"),
    gif=True,
    gif_fps=8,
    context=None,
    frame_idx=24,
    arrow=None,
    diverging_dynamic_range=None,
    focused_dynamic_range=None,
):
    save_path = Path(save_path)
    results = np.load(path, allow_pickle=True)

    focused = results["focused"]
    if focused_dynamic_range is None:
        focused_dynamic_range = results["focused_dynamic_range"]

    diverging = results["diverging"]
    if diverging_dynamic_range is None:
        diverging_dynamic_range = results["diverging_dynamic_range"]

    reconstructions = results["reconstructions"]
    reconstruction_range = results["reconstruction_range"]

    measurements = results["measurements"]
    masks = results["masks"]

    measurements = keras.ops.where(
        masks > 0, measurements, color_to_value(reconstruction_range, "gray")
    )

    io_config = zea.Config(
        scan_convert=True,
        scan_conversion_angles=np.rad2deg(results["theta_range"]),
    )

    focused = postprocess_agent_results(
        focused,
        io_config,
        scan_convert_order=0,
        image_range=focused_dynamic_range,
        fill_value="transparent",
    )
    diverging = postprocess_agent_results(
        diverging,
        io_config,
        scan_convert_order=0,
        image_range=diverging_dynamic_range,
        fill_value="transparent",
    )
    reconstructions = postprocess_agent_results(
        reconstructions,
        io_config,
        scan_convert_order=0,
        image_range=reconstruction_range,
        reconstruction_sharpness_std=0.04,
        fill_value="transparent",
    )
    measurements = postprocess_agent_results(
        measurements,
        io_config,
        scan_convert_order=0,
        image_range=reconstruction_range,
        fill_value="transparent",
    )

    if context is None:
        context = "styles/darkmode.mplstyle"

    if gif:
        side_by_side_gif(
            save_path.with_suffix(".gif"),
            focused,
            reconstructions,
            diverging,
            labels=[
                "Focused (90)",
                "Reconstruction (11/90)",
                "Diverging (11)",
            ],
            context=context,
            fps=gif_fps,
        )

    with plt.style.context(context):
        kwargs = {
            "vmin": 0,
            "vmax": 255,
            "cmap": "gray",
            "interpolation": "nearest",
        }
        fig, axs = plt.subplots(2, 2, figsize=(3.5, 2.8))
        axs = axs.flatten()
        axs[0].imshow(focused[frame_idx], **kwargs)
        axs[0].set_title("Focused (90)")

        axs[1].imshow(measurements[frame_idx], **kwargs)
        axs[1].set_title("Acquisitions (11/90)")

        axs[3].imshow(reconstructions[frame_idx], **kwargs)
        axs[3].set_title("Reconstruction (11/90)")

        axs[2].imshow(diverging[frame_idx], **kwargs)
        axs[2].set_title("Diverging (11)")

        if arrow is not None:
            axs[3].add_patch(copy.copy(arrow))
            axs[2].add_patch(copy.copy(arrow))

        for ax in axs:
            ax.axis("off")

        for ext in exts:
            plt.savefig(save_path.with_suffix(ext))
            zea.log.info(
                f"Saved cardiac reconstruction plot to {zea.log.yellow(save_path.with_suffix(ext))}"
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
    PLOT_NPZ_PATH = (
        "/mnt/z/usbmd/Wessel/eval_in_house_cardiac/20240701_P1_A4CH_0001_results.npz"
    )

    plot_from_npz(
        PLOT_NPZ_PATH,
        "output/in_house_cardiac.png",
        context="styles/ieee-tmi.mplstyle",
        frame_idx=24,
        arrow=get_arrow(),
    )
