"""This script (and function) loads and reconstructs a cardiac ultrasound scan from raw data."""

import os

import zea

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "jax"
    zea.init_device(allow_preallocate=False)

import matplotlib.pyplot as plt
import numpy as np
from keras import ops

import ulsa.ops
from ulsa.active_sampling_temporal import preload_data
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
    n_transmits=None,
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
            file,
            n_frames,
            data_type="data/raw_data",
            type=type,
            n_transmits=n_transmits,
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
