import jax
import jax.numpy as jnp
import numpy as np
from keras import ops

from zea import Scan
from zea.agent.masks import k_hot_to_indices
from zea.beamform.pixelgrid import radial_pixel_grid

FOCUSED_TRANSMITS = (0, 90)
DIVERGING_TRANSMITS = (90, 101)


def select_transmits_from_pfield(pfield, transmits):
    # pfield: (n_tx, Nz, Nx)
    # transmits: (n_indices, c) -- indices into n_tx
    # Output: (Nz, Nx, c)
    # For each column in transmits, select the corresponding pfield slice and stack along last axis

    # pfield: (n_tx, Nz, Nx)
    # transmits: (n_indices, c)
    # We'll assume n_indices == n_tx or that transmits contains valid indices into n_tx

    # Gather for each column in transmits
    def gather_column(col_indices):
        # col_indices: (n_indices,)
        # Gather pfield[col_indices, :, :] -> (n_indices, Nz, Nx)
        # Then sum or stack as needed; here, stack along new axis
        return pfield[col_indices, :, :]  # (n_indices, Nz, Nx)

    # Apply over columns of transmits
    gathered = jax.vmap(gather_column, out_axes=-1)(transmits.T)
    # gathered: (n_indices, Nz, Nx, c)
    # Let's take the first along n_indices if you want (Nz, Nx, c)
    # Or sum along n_indices if that's the intent
    # Here, let's assume you want to sum along n_indices
    output = ops.sum(gathered, axis=0)  # (Nz, Nx, c)
    return output


def lines_to_pfield(
    selected_lines,  # mask: [c, n_possible_actions]
    pfield,
    n_actions,
    alpha=2.0,
    threshold=0.06,
):
    transmits = k_hot_to_indices(selected_lines, n_actions).T  # (nonzero_w, c)
    summed_pfield = select_transmits_from_pfield(
        pfield**alpha, transmits
    )  # (Nz, Nx, c)

    # Normalize depth wise
    # Each row of pixels must sum to Nx (width of the image)
    summed_pfield = summed_pfield / ops.sum(summed_pfield, axis=1, keepdims=True)

    # Normalize to [0, 1]
    max_vals = ops.max(summed_pfield, axis=(0, 1), keepdims=True)
    # Avoid division by zero: only divide where max_vals > 0, else keep as zero
    summed_pfield = ops.where(
        max_vals > 0,
        summed_pfield / max_vals,
        ops.zeros_like(summed_pfield),
    )

    # Apply threshold
    summed_pfield = ops.where(
        summed_pfield < threshold, ops.zeros_like(summed_pfield), summed_pfield
    )

    return summed_pfield


def set_polar_grid(
    scan: Scan,
    polar_limits=None,
    rlims=None,
    Nr=None,
    angle_upsampling=None,
    n_rays=None,
):
    """Generate a polar grid."""

    # make sure scan.selected_transmits is set to all focused transmits!
    assert scan.n_tx <= 90, (
        "This was put in because the cardiac dataset contains both focused and unfocused transmits"
    )

    def axial_resolution(scan):
        """
        Returns the axial resolution of the scan.

        - Axial resolution is the minimum resolvable distance between two reflectors along
            the depth (axial) direction.
        - The sampling interval in time is 1 / sampling_frequency. To convert this to distance,
            multiply by the speed of sound.
        - Since ultrasound is a pulse-echo technique, the pulse travels to the target and back,
            so the distance is halved.

        Returns:
            float: The axial resolution in meters.
        """
        return scan.sound_speed / (2 * scan.sampling_frequency)

    if rlims is None:
        rlims = scan.zlims

    if polar_limits is None:
        polar_limits = (scan.polar_angles[0], scan.polar_angles[-1])

    if Nr is None:
        dr = axial_resolution(scan)
    else:
        dr = (rlims[1] - rlims[0]) / Nr

    # generate the grid
    print(
        "Half wavelength:{}, dr:{}".format(
            scan.sound_speed / scan.sampling_frequency, dr
        )
    )

    if angle_upsampling is not None and n_rays is None:
        n_rays = int(scan.n_tx * angle_upsampling)
    elif n_rays is None:
        n_rays = scan.n_tx

    oris = np.array([0, 0, 0])
    oris = np.tile(oris, (n_rays, 1))
    dirs_az = np.linspace(*polar_limits, n_rays)

    dirs_el = np.zeros(n_rays)
    dirs = np.vstack((dirs_az, dirs_el)).T
    grid = radial_pixel_grid(rlims, dr, oris, dirs).transpose(1, 0, 2)

    # Set grid
    scan.grid = grid

    # Scan convert parameters:
    scan.rho_range = rlims
    scan.resolution = dr
    return scan


def update_scan_for_polar_grid(
    scan: Scan,
    dynamic_range=(-70, -28),
    pfield_kwargs=None,
    f_number=0,
    n_rays=None,
    extension_factor_polar_limits=4,
):
    """Update the scan object for line scanning."""
    if pfield_kwargs is None:
        pfield_kwargs = {}
    scan.selected_transmits = np.arange(*FOCUSED_TRANSMITS)  # before get_polar_grid!
    scan.pfield_kwargs = {"downsample": 1, "downmix": 1} | pfield_kwargs

    scan.f_number = float(f_number)
    polar_delta = abs(scan.polar_angles[0] - scan.polar_angles[1])
    polar_limits = (
        scan.polar_angles[0] - polar_delta * extension_factor_polar_limits,
        scan.polar_angles[-1] + polar_delta * extension_factor_polar_limits,
    )
    scan = set_polar_grid(scan, polar_limits=polar_limits, n_rays=n_rays)
    scan.dynamic_range = dynamic_range
    scan.fill_value = scan.dynamic_range[0]


def update_scan_for_diverging_waves(
    scan: Scan,
    dynamic_range=(-70, -28),
    pfield_kwargs=None,
    n_rays=300,
    extension_factor_polar_limits=4,
):
    """Update the scan object for diverging waves."""
    if pfield_kwargs is None:
        pfield_kwargs = {}
    scan.selected_transmits = np.arange(*DIVERGING_TRANSMITS)
    scan.pfield_kwargs = {"downsample": 1, "downmix": 1} | pfield_kwargs
    scan.f_number = 0
    polar_delta = abs(scan.polar_angles[0] - scan.polar_angles[1])
    polar_limits = (
        scan.polar_angles[0] - polar_delta * extension_factor_polar_limits,
        scan.polar_angles[-1] + polar_delta * extension_factor_polar_limits,
    )
    scan = set_polar_grid(scan, polar_limits=polar_limits, n_rays=n_rays)
    scan.dynamic_range = dynamic_range
    scan.fill_value = scan.dynamic_range[0]
    scan.rho_range = scan.zlims  # TODO: default for zea?
