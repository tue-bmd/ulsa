import jax
import numpy as np
from keras import ops

from zea import Scan
from zea.agent.masks import k_hot_to_indices

FOCUSED_TRANSMITS = (0, 90)
DIVERGING_TRANSMITS = (90, 101)


def select_transmits_from_pfield(pfield, transmits):
    # pfield: (n_tx, Nz, Nx)
    # transmits: (n_indices, c) -- indices into n_tx
    # Output: (Nz, Nx, c)

    # Gather for each column in transmits
    def gather_column(transmits):
        # transmits: (n_indices,)
        return pfield[transmits, :, :]  # (n_indices, Nz, Nx)

    # Apply over columns of transmits
    gathered = jax.vmap(gather_column, out_axes=-1)(transmits.T)
    # gathered: (n_indices, Nz, Nx, c)

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


def select_transmits(scan, type="focused"):
    if type == "focused":
        scan.selected_transmits = np.arange(*FOCUSED_TRANSMITS)
    elif type == "diverging":
        scan.selected_transmits = np.arange(*DIVERGING_TRANSMITS)
    else:
        raise ValueError(f"Unknown scan type: {type}. Use 'focused' or 'diverging'.")


def update_scan_for_polar_grid(
    scan: Scan,
    dynamic_range=(-70, -28),
    pfield_kwargs=None,
    f_number=0,
):
    """Update the scan object for line scanning."""
    if pfield_kwargs is None:
        pfield_kwargs = {}
    scan.pfield_kwargs = {"downsample": 1, "downmix": 1} | pfield_kwargs
    scan.f_number = float(f_number)
    scan.grid_type = "polar"
    scan.dynamic_range = dynamic_range
    scan.fill_value = float(scan.dynamic_range[0])
