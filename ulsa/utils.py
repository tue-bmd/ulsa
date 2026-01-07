import copy
from typing import List

import jax
import keras
import numpy as np
from keras import ops
from tqdm import tqdm

import zea
from ulsa.pipeline import beamforming
from zea import Scan


def select_transmits_from_pfield(pfield, transmits):
    # pfield: (n_tx, grid_size_z, grid_size_x)
    # transmits: (n_indices, c) -- indices into n_tx
    # Output: (grid_size_z, grid_size_x, c)

    # Gather for each column in transmits
    def gather_column(transmits):
        # transmits: (n_indices,)
        return pfield[transmits, :, :]  # (n_indices, grid_size_z, grid_size_x)

    # Apply over columns of transmits
    gathered = jax.vmap(gather_column, out_axes=-1)(transmits.T)
    # gathered: (n_indices, grid_size_z, grid_size_x, c)

    output = ops.sum(gathered, axis=0)  # (grid_size_z, grid_size_x, c)
    return output


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


def copy_transmits_from_scan(scan: zea.Scan, transmits) -> zea.Scan:
    """This will create a new Scan object with the only selected transmits. The
    other transmits are removed."""
    scan.set_transmits(transmits)
    scan_dict = copy.deepcopy(scan._params)
    for property_name in scan._properties:
        if property_name not in scan.VALID_PARAMS:
            continue
        scan_dict[property_name] = getattr(scan, property_name)
    return zea.Scan(**scan_dict)


def load_subsampled_data(
    file: zea.File, data_type: str, frame_nr: int, selected_lines, n_actions: int
):
    if data_type == "data/raw_data":
        # We can actually subsample the raw data here.
        transmits = selected_lines_to_transmits(selected_lines, n_actions)
        measurement = file.load_data(data_type, (frame_nr, transmits))
    else:
        # Here we assume that every transmit event is a line.
        image_shape = file.shape(data_type)[1:]
        mask = zea.agent.masks.lines_to_im_size(selected_lines[None], image_shape)
        mask = keras.ops.squeeze(mask, 0)
        target = file.load_data(data_type, frame_nr)
        measurement = target * mask

    return measurement


def selected_lines_to_transmits(selected_lines, n_actions: int) -> List[int]:
    transmits = zea.agent.masks.k_hot_to_indices(selected_lines[None], n_actions)
    transmits = keras.ops.squeeze(transmits, 0)
    transmits = list(keras.ops.convert_to_numpy(transmits))
    return transmits


def get_subsampled_parameters(
    data_type: str, scan: zea.Scan, selected_lines, rx_apo, n_actions: int
):
    if data_type == "data/raw_data":
        transmits = selected_lines_to_transmits(selected_lines, n_actions)
        scan.set_transmits(transmits)
        _rx_apo = rx_apo[transmits]
        return {
            "scan": scan,
            "rx_apo": _rx_apo,
        }
    else:
        return {}


def precompute_dynamic_range(file: zea.File, scan: zea.Scan, params: dict):
    """Uses all the transmits to compute the dynamic range and max value"""
    pipeline = zea.Pipeline(beamforming(), with_batch_dim=False)
    scan.set_transmits("all")
    data = file.load_data("data/raw_data", (0, scan.selected_transmits))
    output = pipeline(data=data, **{**pipeline.prepare_parameters(scan=scan), **params})
    return {"maxval": output["maxval"], "dynamic_range": output["dynamic_range"]}


def scan_sequence(data, pipeline, parameters, keep_keys=None, **kwargs):
    """
    Process a sequence of data frames through the pipeline.
    """
    if keep_keys is None:
        keep_keys = []
    images = []
    for raw_data in tqdm(data):
        output = pipeline(data=raw_data, **parameters, **kwargs)
        for key in keep_keys:
            if key in output:
                kwargs[key] = output[key]
        images.append(keras.ops.convert_to_numpy(output["data"]))
    return keras.ops.stack(images, axis=0)
