"""Simple script to calculate transmit time for ultrasound imaging."""

import numpy as np


def n_lines(
    sound_speed=1540,
    aperture=0.02,
    center_frequency=3e6,
    sector_deg=83,
):
    """Calculate the number of lines that have to be acquired for a given aperture and center frequency."""
    wavelength = sound_speed / center_frequency
    angular_resolution = np.rad2deg(np.arcsin(wavelength / aperture))
    n_lines = int(sector_deg * 2 / angular_resolution)
    print(f"Number of lines that have to be acquired: {n_lines}")
    return n_lines


def max_fps(
    depth=0.15,
    sound_speed=1540,
    n_tx=112,
    processing_overhead=1.0,
    verbose=True,
):
    """
    Calculate the maximum frames per second (fps) for given imaging parameters.

    Parameters:
        depth (float): Imaging depth in meters. Default is 0.15 m (15 cm) for cardiac imaging.
        sound_speed (float): Speed of sound in the medium in m/s.
            Default is 1540 m/s for soft tissue.
        n_tx (int): Number of transmits per frame.
            Default is 112 for a typical cardiac ultrasound probe.
        processing_overhead (float): Multiplier to account for processing overhead per transmit.
        verbose (bool): If True, print detailed timing information.
    """
    transmit_time = (2 * depth) / sound_speed
    transmit_time *= processing_overhead

    if verbose:
        transmit_time_us = transmit_time * 1e6  # Convert seconds to microseconds
        print(f"Transmit time for depth {depth} cm: {transmit_time_us:.3f} us")

    total_time = transmit_time * n_tx
    if verbose:
        total_time_ms = total_time * 1e3  # Convert seconds to milliseconds
        print(f"Total transmit time for {n_tx} transmits: {total_time_ms:.3f} ms")

    total_fps = 1 / total_time
    if verbose:
        print(f"Total frames per second (fps): {total_fps:.2f} fps")
    return total_fps


if __name__ == "__main__":
    max_fps(n_tx=n_lines())
