import numpy as np

from zea import Scan


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
