import os
import sys

import zea

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "jax"
    zea.init_device(allow_preallocate=False)
    sys.path.append("/ulsa")

import jax
import numpy as np
import scipy.signal
from keras import ops
from tqdm import tqdm

from active_sampling_temporal import preload_data
from ulsa.ops import (
    FirFilter,
    GetAutoDynamicRange,
    LowPassFilter,
    WaveletDenoise,
    lines_rx_apo,
)
from zea.display import compute_scan_convert_2d_coordinates, scan_convert_2d
from zea.ops import translate


def focused_waves(target_sequence, n_frames, resize_height=112):
    with zea.File(target_sequence) as file:
        raw_data_sequence, scan, _ = preload_data(
            file,
            n_frames,
            data_type="data/raw_data",
            type="focused",
        )

    pipeline = zea.Pipeline(
        [
            FirFilter(axis=-3, filter_key="bandpass_rf"),
            # WaveletDenoise(),  # optional
            zea.ops.Demodulate(),
            LowPassFilter(complex_channels=True, axis=-2),  # optional
            zea.ops.Downsample(2),  # optional
            zea.ops.PatchedGrid(
                [
                    zea.ops.TOFCorrection(),
                    # zea.ops.PfieldWeighting(),  # optional
                    zea.ops.DelayAndSum(),
                ]
            ),
            zea.ops.EnvelopeDetect(),
            zea.ops.Normalize(),
            GetAutoDynamicRange(),
            zea.ops.LogCompress(),
            zea.ops.Lambda(lambda x: ops.expand_dims(x, axis=-1)),
            zea.ops.Lambda(
                ops.image.resize,
                {
                    "size": (resize_height, scan.grid_size_x),
                    "interpolation": "bilinear",
                    "antialias": True,
                },
            ),
            zea.ops.Lambda(lambda x: ops.squeeze(x, axis=-1)),
            # zea.ops.ScanConvert(),
        ],
        with_batch_dim=False,
        jit_options="ops",
    )

    bandpass_rf = scipy.signal.firwin(
        numtaps=128,
        cutoff=np.array([0.5, 1.5]) * scan.center_frequency,
        pass_zero="bandpass",
        fs=scan.sampling_frequency,
    )
    scan.polar_limits = list(np.deg2rad([-45, 45]))
    scan.dynamic_range = None  # for auto-dynamic range
    rx_apo = lines_rx_apo(scan.n_tx, scan.grid_size_z, scan.grid_size_x)
    params = pipeline.prepare_parameters(
        scan=scan,
        bandwidth=2e6,
        bandpass_rf=bandpass_rf,
        rx_apo=rx_apo,
    )
    scan_convert_2d_jit = jax.jit(
        scan_convert_2d, static_argnames=("fill_value", "order")
    )

    image_shape = (resize_height, scan.grid_size_x)
    coords, _ = compute_scan_convert_2d_coordinates(
        image_shape=image_shape,
        rho_range=(0, image_shape[0]),
        theta_range=scan.theta_range,
        resolution=0.1,
    )

    images = []
    for raw_data in tqdm(raw_data_sequence):
        output = pipeline(data=raw_data, **params)
        image = output.pop("data")
        params["maxval"] = output.pop("maxval")
        image, _ = scan_convert_2d_jit(
            image, coordinates=coords, fill_value=np.nan, order=0
        )
        images.append(ops.convert_to_numpy(image))
    images = np.stack(images, axis=0)

    dynamic_range = output["dynamic_range"]
    return images, dynamic_range


if __name__ == "__main__":
    from pathlib import Path

    save_dir = Path("/mnt/z/usbmd/Wessel/cardiac_mp4s")
    save_dir.mkdir(parents=True, exist_ok=True)

    folder = Path("/mnt/USBMD_datasets/2024_USBMD_cardiac_S51/HDF5/")
    # find all a4ch files
    files = list(folder.glob("*_A4CH_*.hdf5"))
    n_frames = None  # all frames

    for file in files:
        images, dynamic_range = focused_waves(file, n_frames)
        _images = translate(images, dynamic_range, (0, 255))
        _images = np.clip(_images, 0, 255).astype(np.uint8)
        mp4_path = (save_dir / file.stem).with_suffix(".mp4")
        zea.io_lib.save_to_mp4(_images, mp4_path, fps=30)
