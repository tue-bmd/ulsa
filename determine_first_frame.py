from pathlib import Path

import numpy as np
from tqdm import tqdm

import ulsa.ops
import zea
from ulsa.pipeline import Pipeline, beamforming
from ulsa.utils import update_scan_for_polar_grid

if __name__ == "__main__":
    zea.init_device()

# Pipeline with Lee filter to reduce speckle noise
pipeline = Pipeline(beamforming(rx_apo=True), with_batch_dim=False)
pipeline.append(zea.ops.keras_ops.ExpandDims(axis=-1))
pipeline.append(zea.ops.LeeFilter(sigma=5))
pipeline.append(zea.ops.keras_ops.Squeeze(axis=-1))


def get_first_frame(file: zea.File, scan: zea.Scan, top_cropping=50, **params):
    # TODO: top_cropping should be relative to image size

    data_array, _ = pipeline.run(file, scan, **params)

    data_array = zea.func.translate(data_array, (-60, 0), (0, 1))
    data_array = np.clip(data_array, 0, 1)

    # Add last frame at the beginning to capture changes from last to first frame
    data_array = np.concatenate([data_array[-1:], data_array], axis=0)

    diff = np.abs(np.diff(data_array, axis=0))[:, top_cropping:].mean(axis=(1, 2))
    first_frame_idx = diff.argmax()

    n_frames = file.n_frames
    indices = np.arange(first_frame_idx, first_frame_idx + n_frames) % n_frames

    data_array = zea.display.to_8bit(data_array, (0, 1), pillow=False)

    zea.io_lib.save_video(
        data_array[indices],
        f"first_frame_detection_{file.name}.mp4",
        fps=10,
    )
    print(f"File: {file.path}")
    print("First frame index:", first_frame_idx)

    return first_frame_idx, diff


if __name__ == "__main__":
    folder = "/mnt/z/usbmd/Wessel/Verasonics/2026_USBMD_A4CH_S51/"
    files = Path(folder).glob("*.hdf5")

    for filepath in tqdm(files):
        with zea.File(filepath) as file:
            scan = file.scan()

            scan.set_transmits("focused")
            update_scan_for_polar_grid(scan, pixels_per_wavelength=1, ray_multiplier=3)

            rx_apo = ulsa.ops.lines_rx_apo(
                n_tx=scan.n_tx,
                grid_size_z=scan.grid_size_z,
                grid_size_x=scan.grid_size_x,
            )

            first_frame_idx, _ = get_first_frame(
                file, scan, rx_apo=rx_apo, bandwidth=2e6, minval=0
            )
