import os
import sys

import zea

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "jax"
    zea.init_device(allow_preallocate=False)
    sys.path.append("/ulsa")

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from keras import ops
from tqdm import tqdm

import ulsa.ops
from active_sampling_temporal import preload_data
from ulsa.pipeline import make_pipeline
from zea.display import scan_convert_2d
from zea.func import translate
from zea.io_lib import save_to_gif


def cardiac_scan(
    target_sequence,
    n_frames,
    grid_width=90,
    resize_to=(112, 112),
    type="focused",  # "focused" or "diverging"
):
    pipeline = make_pipeline(
        "data/raw_data",
        None,
        resize_to,
        resize_to,
        jit_options="ops",
        rx_apo=(type == "focused"),
    )
    pipeline.append(zea.ops.keras_ops.Squeeze(axis=-1))

    with zea.File(target_sequence) as file:
        raw_data_sequence, scan, _ = preload_data(
            file, n_frames, data_type="data/raw_data", type=type
        )

    width = 2e6  # Hz
    f1 = scan.demodulation_frequency - width / 2
    f2 = scan.demodulation_frequency + width / 2
    # TODO: wide enough for fundemental?
    bandpass_rf = zea.func.get_band_pass_filter(128, scan.sampling_frequency, f1, f2)
    scan.polar_limits = list(np.deg2rad([-45, 45]))
    scan.grid_size_x = grid_width
    scan.dynamic_range = None  # for auto-dynamic range

    if type == "focused":
        kwargs = {
            "rx_apo": ulsa.ops.lines_rx_apo(
                scan.n_tx, scan.grid_size_z, scan.grid_size_x
            )
        }
    else:
        kwargs = {}

    # TODO: fix minval to 0?
    params = pipeline.prepare_parameters(
        scan=scan, bandwidth=2e6, bandpass_rf=bandpass_rf, **kwargs
    )

    images = []
    for raw_data in tqdm(raw_data_sequence):
        output = pipeline(data=raw_data, **params)
        image = output.pop("data")
        params["maxval"] = output.pop("maxval")
        params["dynamic_range"] = output.pop("dynamic_range")
        images.append(ops.convert_to_numpy(image))
    images = np.stack(images, axis=0)

    scan.dynamic_range = params["dynamic_range"]
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
