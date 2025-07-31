"""
This script makes a figure with the target, reconstruction, and measurements
for the in-house cardiac dataset.

# diverging_dynamic_range = [-70, -30]
"""

import sys

import zea

if __name__ == "__main__":
    zea.init_device(allow_preallocate=False)
    sys.path.append("/ulsa")

from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np

from ulsa.io_utils import color_to_value, postprocess_agent_results, side_by_side_gif


def plot_from_npz(
    path,
    save_path,
    exts=(".png", ".pdf"),
    gif=True,
    gif_fps=8,
    context=None,
    frame_idx=24,
):
    save_path = Path(save_path)
    results = np.load(path, allow_pickle=True)

    focused = results["focused"]
    focused_dynamic_range = results["focused_dynamic_range"]

    diverging = results["diverging"]
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

        for ax in axs:
            ax.axis("off")

        for ext in exts:
            plt.savefig(save_path.with_suffix(ext))
            zea.log.info(
                f"Saved cardiac reconstruction plot to {zea.log.yellow(save_path.with_suffix(ext))}"
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
    )
