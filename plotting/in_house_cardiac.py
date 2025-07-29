"""
This script makes a figure with the target, reconstruction, and measurements
for the in-house cardiac dataset.
"""

import sys

import zea

sys.path.append("/ulsa")  # for relative imports

zea.init_device(allow_preallocate=False)

from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np

from active_sampling_temporal import active_sampling_single_file
from diverging import diverging_waves
from ulsa.io_utils import postprocess_agent_results, side_by_side_gif

MAKE_GIF = True
FRAME_IDX = 24
FRAME_CUTOFF = 100
DROP_FIRST_N_FRAMES = 2  # drop first 2 frames to avoid artifacts (from gif only!)

override_config = dict(io_config=dict(frame_cutoff=FRAME_CUTOFF))
target_sequence = (
    "/mnt/USBMD_datasets/2024_USBMD_cardiac_S51/HDF5/20240701_P1_A4CH_0001.hdf5"
)
results, _, _, _, agent, agent_config, _ = active_sampling_single_file(
    "configs/cardiac_112_3_frames.yaml",
    target_sequence=target_sequence,
    override_config=override_config,
)
image_range = agent.input_range

no_measurement_color = "gray"
if no_measurement_color == "white":
    no_measurement_color = image_range[1]
elif no_measurement_color == "black":
    no_measurement_color = image_range[0]
elif no_measurement_color == "gray":
    no_measurement_color = (image_range[0] + image_range[1]) / 2
elif no_measurement_color == "transparent":
    no_measurement_color = np.nan
else:
    raise ValueError(f"Unknown no_measurement_color: {no_measurement_color}")

squeezed_results = results.squeeze(-1)
targets = squeezed_results.target_imgs
reconstructions = squeezed_results.reconstructions
measurements = keras.ops.where(
    squeezed_results.masks > 0,
    squeezed_results.measurements,
    no_measurement_color,
)
io_config = agent_config.io_config
scan_convert_order = io_config.get("plot_frames_for_presentation_kwargs", {}).get(
    "scan_convert_order", 0
)
scan_convert_resolution = 0.1

targets = postprocess_agent_results(
    targets,
    io_config,
    scan_convert_order,
    image_range,
    scan_convert_resolution=scan_convert_resolution,
    fill_value="transparent",
)
reconstructions = postprocess_agent_results(
    reconstructions,
    io_config,
    scan_convert_order,
    image_range,
    scan_convert_resolution=scan_convert_resolution,
    reconstruction_sharpness_std=io_config.get("reconstruction_sharpness_std", 0.0),
    fill_value="transparent",
)
measurements = postprocess_agent_results(
    measurements,
    io_config,
    scan_convert_order=0,  # always 0 for masks!
    image_range=image_range,
    scan_convert_resolution=scan_convert_resolution,
    fill_value="transparent",
)

diverging_dynamic_range = [-70, -30]
diverging_images = diverging_waves(
    target_sequence, FRAME_CUTOFF, diverging_dynamic_range
)

filestem = Path(target_sequence).stem
np.savez(
    f"output/{filestem}.npz",
    targets=targets,
    reconstructions=reconstructions,  # TODO: maybe without reconstruction_sharpness_std?
    measurements=measurements,
    diverging_images=diverging_images,
    diverging_dynamic_range=diverging_dynamic_range,
    image_range=image_range,
    filestem=filestem,
)
images = zea.display.to_8bit(targets, image_range, pillow=False)
zea.utils.save_to_mp4(images, f"output/{filestem}_targets.mp4", fps=5)

exts = ["png", "pdf"]
with plt.style.context("styles/ieee-tmi.mplstyle"):
    kwargs = {
        "vmin": image_range[0],
        "vmax": image_range[1],
        "cmap": "gray",
        "interpolation": "nearest",
    }
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 2.8))
    axs = axs.flatten()
    axs[0].imshow(targets[FRAME_IDX], **kwargs)
    axs[0].set_title("Focused (90)")

    axs[1].imshow(measurements[FRAME_IDX], **kwargs)
    axs[1].set_title("Acquisitions (11/90)")

    axs[3].imshow(reconstructions[FRAME_IDX], **kwargs)
    axs[3].set_title("Reconstruction (11/90)")

    axs[2].imshow(
        diverging_images[FRAME_IDX],
        cmap="gray",
        vmin=diverging_dynamic_range[0],
        vmax=diverging_dynamic_range[1],
        interpolation="nearest",
    )
    axs[2].set_title("Diverging (11)")

    for ax in axs:
        ax.axis("off")

    for ext in exts:
        path = f"output/in_house_cardiac.{ext}"
        plt.savefig(path)
        zea.log.info(f"Saved cardiac reconstruction plot to {zea.log.yellow(path)}")

_diverging_images = zea.utils.translate(
    np.stack(diverging_images), diverging_dynamic_range, image_range
)
for fps in [5, 30]:
    side_by_side_gif(
        f"output/in_house_cardiac_{fps}.gif",
        targets[DROP_FIRST_N_FRAMES:],
        reconstructions[DROP_FIRST_N_FRAMES:],
        _diverging_images[DROP_FIRST_N_FRAMES:],
        vmin=image_range[0],
        vmax=image_range[1],
        labels=[
            "Focused (90)",
            "Reconstruction (11/90)",
            "Diverging (11)",
        ],
        fps=fps,
        context="styles/darkmode.mplstyle",
    )
print("Done.")
