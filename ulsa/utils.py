"""Utility functions."""

import matplotlib.pyplot as plt
import numpy as np

import zea
from zea import Scan


def find_best_cine_loop(
    data,  # shape (n_frames, height, width)
    min_sequence_length=30,
    visualize=True,
):
    """Find the best cine loop by comparing frame differences to the first frame.

    Args:
        data: np.ndarray of shape (n_frames, height, width)
        min_sequence_length: Minimum number of frames before considering loop closure.
        visualize: Whether to save a plot of the frame differences.
    """
    first_frame = data[0]
    other_frames = data[1:]
    differences = np.sum(np.abs(other_frames - first_frame[None]), axis=(1, 2))
    min_diff_idx = np.argmin(differences[min_sequence_length:]) + min_sequence_length

    if visualize:
        plt.figure()
        plt.plot(differences)
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
