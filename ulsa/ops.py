import jax
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
