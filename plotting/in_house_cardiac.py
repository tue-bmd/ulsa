"""
This script makes a figure with the target, reconstruction, and measurements
for the in-house cardiac dataset.
"""

import sys

import keras

import zea

sys.path.append("/ulsa")  # for relative imports

zea.init_device(allow_preallocate=False)

import matplotlib.pyplot as plt

from active_sampling_temporal import active_sampling_single_file
from ulsa.io_utils import postprocess_agent_results

frame_idx = 24
results, _, _, _, agent, agent_config, _ = active_sampling_single_file(
    "configs/cardiac_112_3_frames.yaml",
    override_config=dict(io_config=dict(frame_cutoff=frame_idx + 1)),
)
image_range = agent.input_range

no_measurement_color = "gray"
if no_measurement_color == "white":
    no_measurement_color = image_range[1]
elif no_measurement_color == "black":
    no_measurement_color = image_range[0]
elif no_measurement_color == "gray":
    no_measurement_color = (image_range[0] + image_range[1]) / 2
else:
    raise ValueError(f"Unknown no_measurement_color: {no_measurement_color}")

squeezed_results = results.squeeze(-1)
targets = squeezed_results.target_imgs[frame_idx]
reconstructions = squeezed_results.reconstructions[frame_idx]
measurements = keras.ops.where(
    squeezed_results.masks[frame_idx] > 0,
    squeezed_results.measurements[frame_idx],
    no_measurement_color,
)
io_config = agent_config.io_config
scan_convert_order = io_config.get("plot_frames_for_presentation_kwargs", {}).get(
    "scan_convert_order", 0
)
scan_convert_resolution = 0.1

targets = postprocess_agent_results(
    targets[None],
    io_config,
    scan_convert_order,
    image_range,
    scan_convert_resolution=scan_convert_resolution,
    fill_value="white",
)[0]
reconstructions = postprocess_agent_results(
    reconstructions[None],
    io_config,
    scan_convert_order,
    image_range,
    scan_convert_resolution=scan_convert_resolution,
    reconstruction_sharpness_std=io_config.get("reconstruction_sharpness_std", 0.0),
    fill_value="white",
)[0]
measurements = postprocess_agent_results(
    measurements[None],
    io_config,
    scan_convert_order=0,  # always 0 for masks!
    image_range=image_range,
    scan_convert_resolution=scan_convert_resolution,
    fill_value="white",
)[0]

with plt.style.context("styles/ieee-tmi.mplstyle"):
    kwargs = {"vmin": 0, "vmax": 255, "cmap": "gray", "interpolation": "nearest"}
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(measurements, **kwargs)
    axs[0].set_title("Acquisitions")
    axs[1].imshow(targets, **kwargs)
    axs[1].set_title("Target")
    axs[2].imshow(reconstructions, **kwargs)
    axs[2].set_title("Reconstruction")
    for ax in axs:
        ax.axis("off")

    plt.savefig("output/cardiac_reconstruction.png")
zea.log.info(
    f"Saved cardiac reconstruction plot to {zea.log.yellow('output/cardiac_reconstruction.png')}"
)
