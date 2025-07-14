import jax
import numpy as np
from keras import ops

import zea.ops


def apply_along_axis(func, axis, arr):
    """Apply a function to 1-D slices along the given axis.

    Based on [np.apply_along_axis](https://numpy.org/devdocs/reference/generated/numpy.apply_along_axis.html)
    """
    arr = ops.moveaxis(arr, axis, -1)
    ndim = ops.ndim(arr)
    for _ in range(ndim - 1):
        func = jax.vmap(func)
    result = func(arr)
    return ops.moveaxis(result, -1, axis)


def iq2doppler(
    data,
    center_frequency,
    pulse_repetition_frequency,
    sound_speed,
    hamming_size=None,
    lag=1,
):
    """Compute Doppler from packet of I/Q Data.

    Args:
        data (ndarray): I/Q complex data of shape (n_z, n_x, n_frames).
            n_frames corresponds to the ensemble length used to compute
            the Doppler signal.

    Returns:
        doppler_velocities (ndarray): Doppler velocity map of shape (n_z, n_x).

    """
    assert data.ndim == 3, "Data must be a 3-D array"
    assert isinstance(lag, int) and lag >= 0, "Lag must be a positive integer"

    if hamming_size is None:
        hamming_size = np.array([1, 1])
    elif np.isscalar(hamming_size):
        hamming_size = np.array([hamming_size, hamming_size])
    assert hamming_size.all() > 0 and np.all(hamming_size == np.round(hamming_size)), (
        "hamming_size must contain integers > 0"
    )

    # Auto-correlation method
    iq1 = data[:, :, : data.shape[-1] - lag]
    iq2 = data[:, :, lag:]
    autocorr = ops.sum(iq1 * ops.conj(iq2), axis=2)  # Ensemble auto-correlation
    autocorr = ops.convert_to_numpy(autocorr)

    # Spatial weighted average
    if hamming_size[0] != 1 and hamming_size[1] != 1:
        h_row = np.hamming(hamming_size[0])
        h_col = np.hamming(hamming_size[1])
        autocorr = apply_along_axis(
            lambda x: ops.correlate(x, h_row, mode="same"), 0, autocorr
        )
        autocorr = apply_along_axis(
            lambda x: ops.correlate(x, h_col, mode="same"), 1, autocorr
        )

    # Doppler velocity
    nyquist_velocities = (
        sound_speed * pulse_repetition_frequency / (4 * center_frequency * lag)
    )
    doppler_velocities = -nyquist_velocities * ops.imag(ops.log(autocorr)) / np.pi

    return doppler_velocities


class AntiAliasing(zea.ops.Operation):
    def __init__(self, num_taps=64, axis=-3, complex_channels=False):
        super().__init__(jittable=False)
        self.num_taps = num_taps
        self.axis = axis
        self.complex_channels = complex_channels

    def call(
        self,
        sampling_frequency=None,
        center_frequency=None,
        bandwidth=None,
        factor=2,
        **kwargs,
    ):
        signal = kwargs[self.key]

        if self.complex_channels:
            signal = zea.ops.channels_to_complex(signal)

        if bandwidth is None:
            bandwidth = sampling_frequency / factor

        # Step 1: Design the low-pass filter
        lpf = zea.ops.get_low_pass_iq_filter(
            self.num_taps,
            ops.convert_to_numpy(sampling_frequency).item(),
            center_frequency,
            bandwidth,
        )

        def _correlate(signal):
            """Apply the filter to the signal using correlation."""
            return ops.correlate(signal, lpf[::-1], mode="same")

        filtered_signal = apply_along_axis(_correlate, self.axis, signal)

        if self.complex_channels:
            filtered_signal = zea.ops.complex_to_channels(filtered_signal)

        return {self.output_key: filtered_signal}


if __name__ == "__main__":
    # Example usage
    signal = ops.zeros((100, 20, 10))
    sampling_frequency = 1000  # Hz
    center_frequency = 100  # Hz
    bandwidth = 50  # Hz

    anti_aliasing_op = AntiAliasing()
    filtered_signal = anti_aliasing_op(
        data=signal,
        sampling_frequency=sampling_frequency,
        center_frequency=center_frequency,
        bandwidth=bandwidth,
    )
    print(filtered_signal)  # Output the filtered signal
