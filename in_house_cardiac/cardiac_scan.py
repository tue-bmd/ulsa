"""This script (and function) loads and reconstructs a cardiac ultrasound scan from raw data."""

import os
import sys

import zea

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "jax"
    zea.init_device(allow_preallocate=False)
    sys.path.append("/ulsa")

import matplotlib.pyplot as plt
import numpy as np
from keras import ops

import ulsa.ops
from active_sampling_temporal import preload_data
from ulsa.pipeline import make_pipeline
from zea.display import scan_convert_2d
from zea.func import translate
from zea.io_lib import save_to_gif


def cardiac_scan(
    target_sequence,
    n_frames=None,
    grid_width=None,
    resize_to=None,
    type="focused",  # "focused" or "diverging"
    bandwidth=2e6,  # TODO: wide enough for fundemental?
    polar_limits=None,
    low_pct=18,
    high_pct=95,
):
    pipeline = make_pipeline(
        "data/raw_data",
        output_shape=resize_to,
        action_selection_shape=resize_to,
        rx_apo=(type == "focused"),
        low_pct=low_pct,
        high_pct=high_pct,
    )
    pipeline.append(zea.ops.keras_ops.Squeeze(axis=-1))

    with zea.File(target_sequence) as file:
        raw_data_sequence, scan = preload_data(
            file, n_frames, data_type="data/raw_data", type=type
        )

    if polar_limits is not None:
        scan.polar_limits = list(polar_limits)
    else:
        scan.polar_limits = list(np.deg2rad([-45, 45]))
    if grid_width is not None:
        scan.grid_size_x = grid_width

    scan.dynamic_range = None  # for auto-dynamic range

    params = {}
    if type == "focused":
        params["rx_apo"] = ulsa.ops.lines_rx_apo(
            scan.n_tx, scan.grid_size_z, scan.grid_size_x
        )

    params = pipeline.prepare_parameters(
        scan=scan, bandwidth=bandwidth, minval=0, **params
    )

    images, output = pipeline.run(
        raw_data_sequence, keep_keys=["maxval", "dynamic_range"], **params
    )

    scan.dynamic_range = ops.convert_to_numpy(output["dynamic_range"]).tolist()
    return images, scan


if __name__ == "__main__":
    type = "focused"  # or "focused"
    images, scan = cardiac_scan(
        "/mnt/USBMD_datasets/2024_USBMD_cardiac_S51/HDF5/20240701_P1_A4CH_0001.hdf5",
        n_frames=None,
        type=type,
    )
    dynamic_range = scan.dynamic_range

    images, _ = scan_convert_2d(
        images,
        rho_range=(0, images.shape[-2]),
        theta_range=scan.theta_range,
        resolution=0.1,
        fill_value=np.nan,
        order=0,
    )
    np.savez(f"output/{type}.npz", images=images, dynamic_range=dynamic_range)

    _images = translate(images, dynamic_range, (0, 255))
    _images = np.clip(_images, 0, 255).astype(np.uint8)
    save_to_gif(_images, f"output/{type}.gif", fps=30)
    plt.imshow(images[24], cmap="gray", vmin=dynamic_range[0], vmax=dynamic_range[1])
    plt.axis("off")
    plt.savefig(f"output/{type}.png", dpi=300, bbox_inches="tight")
    zea.log.info(f"Saved frames to output/{type}.png")
