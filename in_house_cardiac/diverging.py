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

from active_sampling_temporal import make_pipeline, preload_data
from zea.display import scan_convert_2d
from zea.utils import save_to_gif, translate


def diverging_waves(
    target_sequence,
    n_frames,
    dynamic_range,
    grid_width=90,
    resize_height=112,
):
    shape = (resize_height, grid_width)
    pipeline = make_pipeline(
        "data/raw_data", dynamic_range, dynamic_range, [*shape, 1], shape
    )
    pipeline.append(zea.ops.Lambda(lambda x: ops.squeeze(x, axis=-1)))

    with zea.File(target_sequence) as file:
        raw_data_sequence, scan, probe = preload_data(
            file,
            n_frames,
            data_type="data/raw_data",
            type="diverging",
        )

    bandpass_rf = scipy.signal.firwin(
        numtaps=128,
        cutoff=np.array([0.5, 1.5]) * scan.center_frequency,
        pass_zero="bandpass",
        fs=scan.sampling_frequency,
    )
    scan.polar_limits = list(np.deg2rad([-45, 45]))
    scan.grid_size_x = grid_width
    scan.dynamic_range = dynamic_range
    params = pipeline.prepare_parameters(
        scan=scan,
        bandwidth=2e6,
        bandpass_rf=bandpass_rf,
    )

    images = []
    for raw_data in tqdm(raw_data_sequence):
        output = pipeline(data=raw_data, **params)
        image = output.pop("data")
        params["maxval"] = output.pop("maxval")
        images.append(ops.convert_to_numpy(image))
    images = np.stack(images, axis=0)

    return images, scan


if __name__ == "__main__":
    dynamic_range = [-70, -30]
    images, scan = diverging_waves(
        "/mnt/USBMD_datasets/2024_USBMD_cardiac_S51/HDF5/20240701_P1_A4CH_0001.hdf5",
        n_frames=None,
        dynamic_range=dynamic_range,
    )

    images, _ = scan_convert_2d(
        images,
        rho_range=(0, images.shape[-2]),
        theta_range=scan.theta_range,
        resolution=0.1,
        fill_value=np.nan,
        order=0,
    )
    np.savez("output/diverging.npz", images=images, dynamic_range=dynamic_range)

    _images = translate(images, dynamic_range, (0, 255))
    _images = np.clip(_images, 0, 255).astype(np.uint8)
    save_to_gif(_images, "output/diverging.gif", fps=30)
    plt.imshow(images[24], cmap="gray", vmin=dynamic_range[0], vmax=dynamic_range[1])
    plt.axis("off")
    plt.savefig("output/diverging.png", dpi=300, bbox_inches="tight")
    zea.log.info("Saved diverging frame to output/diverging.png")
