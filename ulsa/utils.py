"""Utility functions."""

import matplotlib.pyplot as plt
import numpy as np

import zea
from zea import Scan


def find_best_cine_loop(
    data,  # shape (n_frames, height, width)
    min_sequence_length=30,
    visualize=False,
):
    """Find the best cine loop by comparing frame differences to the first frame.

    Args:
        data: np.ndarray of shape (n_frames, height, width)
        min_sequence_length: Minimum number of frames before considering loop closure.
        visualize: Whether to save a plot of the frame differences.

    Returns:
        Index of the frame that best matches the first frame after min_sequence_length. To create
        the best cine loop, use data[:best_frame_index], or data[:best_frame_index + 1] to
        include the matching frame (this may introduce a very similar frame at the loop point).
    """
    n_frames = np.shape(data)[0]
    assert np.ndim(data) == 3, (
        "Data must be a 3D array of shape (n_frames, height, width)."
    )
    assert n_frames >= 2, "Data must contain at least two frames."
    assert min_sequence_length >= 2, "min_sequence_length must be at least 2"

    # Return full length if min_sequence_length exceeds available frames
    if min_sequence_length > n_frames:
        zea.log.warning(
            f"min_sequence_length {min_sequence_length} is greater than number of frames {n_frames}. "
        )
        return n_frames

    first_frame = data[0]
    other_frames = data[1:]

    # Compute sum of absolute differences from the first frame,
    # this results in an array of shape (n_frames - 1,)
    differences = np.sum(np.abs(other_frames - first_frame[None]), axis=(1, 2))

    min_diff_idx = (
        np.argmin(differences[min_sequence_length - 1 :]) + min_sequence_length
    )

    if visualize:
        _differences = np.concatenate(([0], differences))
        plt.figure()
        plt.plot(_differences)
        plt.axvline(min_sequence_length, color="red", linestyle="--")
        plt.axvline(min_diff_idx, color="green", linestyle="--")
        plt.title("Frame Differences from First Frame")
        plt.xlabel("Frame Index (relative to first frame)")
        plt.ylabel("Sum of Absolute Differences")
        plt.savefig("frame_differences.png")
        plt.close()
        zea.log.info(
            f"Saved frame differences plot to {zea.log.yellow('frame_differences.png')}"
        )
    return min_diff_idx


def round_cm_down(x):
    """Round down to the nearest centimeter."""
    return np.floor(x * 100) / 100


def update_scan_for_polar_grid(
    scan: Scan,
    pfield_kwargs=None,
    f_number=0.3,
    ray_multiplier: int = 6,
    pixels_per_wavelength=4,
    apply_lens_correction: bool = True,
):
    """Update the scan object for line scanning."""
    if pfield_kwargs is None:
        pfield_kwargs = {}
    scan.pfield_kwargs = {
        "downsample": 1,
        "downmix": 1,
        "percentile": 1,
        "alpha": 0.5,
        "norm": False,
    } | pfield_kwargs
    scan.f_number = float(f_number)
    scan.grid_type = "polar"
    scan.grid_size_x = scan.n_tx * ray_multiplier
    scan.polar_limits = scan.polar_angles.min(), scan.polar_angles.max()
    scan.pixels_per_wavelength = pixels_per_wavelength
    scan.zlims = (round_cm_down(scan.zlims[0]), round_cm_down(scan.zlims[1]))
    if hasattr(scan, "n_ch"):
        delattr(scan, "n_ch")

    # For Philips S5-1 probes!
    if apply_lens_correction:
        scan.apply_lens_correction = True
        scan.lens_thickness = 1e-3
        scan.lens_sound_speed = 1000
