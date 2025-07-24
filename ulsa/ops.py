from pathlib import Path

import jax
import numpy as np
from keras import ops

import zea.ops


def wavelet_denoise_rf(rf_signal, wavelet="db4", level=4, threshold_factor=0.5):
    """
    Denoise ultrasound RF signal using wavelet thresholding.

    Parameters:
    - rf_signal: 1D numpy array of RF data
    - wavelet: Wavelet type (e.g., 'db4', 'sym8')
    - level: Decomposition level
    - threshold_factor: Scaling for universal threshold

    Returns:
    - Denoised RF signal
    """
    import pywt  # pip install PyWavelets

    # Decompose
    coeffs = pywt.wavedec(rf_signal, wavelet, level=level)

    # Estimate noise from the detail coefficients at the highest level
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = threshold_factor * sigma * np.sqrt(2 * np.log(len(rf_signal)))

    # Threshold detail coefficients
    new_coeffs = [coeffs[0]]  # Keep approximation unaltered
    for c in coeffs[1:]:
        new_c = pywt.threshold(c, threshold, mode="soft")  # or 'hard'
        new_coeffs.append(new_c)

    # Reconstruct signal
    return pywt.waverec(new_coeffs, wavelet)


def wavelet_denoise_full(data, axis, **kwargs):
    """
    Apply wavelet denoising to the data along a specified axis.

    Parameters:
    - data: Input data (e.g., RF signal)
    - axis: Axis along which to apply the denoising
    - kwargs: Additional parameters for wavelet denoising

    Returns:
    - Denoised data
    """
    # Apply wavelet denoising along the specified axis
    return np.apply_along_axis(lambda x: wavelet_denoise_rf(x, **kwargs), axis, data)


class BM3DDenoiser(zea.ops.Operation):
    """Block matching 3D denoiser."""

    def __init__(self, sigma, stage="all_stages", **kwargs):
        super().__init__(**kwargs, jittable=False)
        import bm3d  # pip install bm3d

        self.sigma = sigma
        str_to_stage = {
            "hard_thresholding": bm3d.BM3DStages.HARD_THRESHOLDING,
            "all_stages": bm3d.BM3DStages.ALL_STAGES,
        }

        self.stage = str_to_stage[stage]

    def call(self, **kwargs):
        import bm3d  # pip install bm3d

        image = kwargs[self.key]
        denoised_image = bm3d.bm3d(image, self.sigma, stage_arg=self.stage)

        return {self.output_key: denoised_image}


class Sharpen(zea.ops.Operation):
    """Sharpen an image using unsharp masking."""

    def __init__(self, sigma=1.0, amount=1.0, **kwargs):
        super().__init__(**kwargs, jittable=False)
        self.sigma = sigma
        self.amount = amount

    def call(self, **kwargs):
        from skimage.filters import unsharp_mask  # pip install scikit-image

        image = kwargs[self.key]
        sharpened_image = unsharp_mask(image, sigma=self.sigma, amount=self.amount)

        return {self.output_key: sharpened_image}


class WaveletDenoise(zea.ops.Operation):
    def __init__(self, wavelet="db4", level=4, threshold_factor=0.1, axis=-3):
        super().__init__(jittable=False)
        self.wavelet = wavelet
        self.level = level
        self.threshold_factor = threshold_factor
        self.axis = axis

    def call(self, **kwargs):
        signal = kwargs[self.key]
        denoised_signal = wavelet_denoise_full(
            signal,
            axis=self.axis,
            wavelet=self.wavelet,
            level=self.level,
            threshold_factor=self.threshold_factor,
        )
        return {self.output_key: denoised_signal}


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
        data (ndarray): I/Q complex data of shape (grid_size_z, grid_size_x, n_frames).
            n_frames corresponds to the ensemble length used to compute
            the Doppler signal.
        center_frequency (float): Center frequency of the ultrasound probe in Hz.
        pulse_repetition_frequency (float): Pulse repetition frequency in Hz.
        sound_speed (float): Speed of sound in the medium in m/s.
        hamming_size (int or tuple, optional): Size of the Hamming window to apply
            for spatial averaging. If None, no window is applied.
            If an integer, it is applied to both dimensions. If a tuple, it should
            contain two integers for the row and column dimensions.
        lag (int, optional): Lag for the auto-correlation computation.
            Defaults to 1, meaning Doppler is computed from the current frame
            and the next frame.

    Returns:
        doppler_velocities (ndarray): Doppler velocity map of shape (grid_size_z, grid_size_x).

    """
    assert data.ndim == 3, "Data must be a 3-D array"
    assert isinstance(lag, int) and lag >= 0, "Lag must be a positive integer"
    assert data.shape[-1] > lag, "Data must have more frames than the lag"

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


def tissue_doppler_strain_rate(velocity_map, axis=0, spacing=1.0, method="central"):
    """
    Compute tissue strain rate (velocity gradient) from a tissue Doppler velocity map.

    Args:
        velocity_map (ndarray): Tissue velocity map (e.g., from Doppler), shape (grid_size_z, grid_size_x).
        axis (int): Axis along which to compute the gradient (default: 0, axial/z).
        spacing (float): Physical distance between points along the axis (in mm or m).
        method (str): Gradient method: "central" (default), "forward", or "backward".

    Returns:
        strain_rate (ndarray): Strain rate map (same shape as velocity_map).
    """
    # Central difference (default)
    if method == "central":
        grad = np.gradient(velocity_map, spacing, axis=axis)
    elif method == "forward":
        grad = (
            np.diff(velocity_map, axis=axis, append=velocity_map.take([-1], axis=axis))
            / spacing
        )
    elif method == "backward":
        grad = (
            np.diff(velocity_map, axis=axis, prepend=velocity_map.take([0], axis=axis))
            / spacing
        )
    else:
        raise ValueError("Unknown method: choose 'central', 'forward', or 'backward'")
    return grad


class FirFilter(zea.ops.Operation):
    def __init__(
        self, axis=-3, complex_channels=False, filter_key="fir_filter_taps", **kwargs
    ):
        super().__init__(**kwargs)
        self.axis = axis
        self.complex_channels = complex_channels
        self.filter_key = filter_key

    @property
    def valid_keys(self):
        """Get the valid keys for the `call` method."""
        return self._valid_keys.union({self.filter_key})

    def call(self, **kwargs):
        signal = kwargs[self.key]
        fir_filter_taps = kwargs.get(self.filter_key)

        if self.complex_channels:
            signal = zea.ops.channels_to_complex(signal)

        def _convolve(signal):
            """Apply the filter to the signal using correlation."""
            return ops.correlate(signal, fir_filter_taps[::-1], mode="same")

        filtered_signal = apply_along_axis(_convolve, self.axis, signal)

        if self.complex_channels:
            filtered_signal = zea.ops.complex_to_channels(filtered_signal)

        return {self.output_key: filtered_signal}


class LowPassFilter(FirFilter):
    def __init__(self, num_taps=128, axis=-3, complex_channels=False):
        super().__init__(
            axis=axis,
            complex_channels=complex_channels,
            jittable=False,
        )
        self.num_taps = num_taps

    def call(
        self,
        sampling_frequency=None,
        center_frequency=None,
        bandwidth=None,
        factor=None,
        **kwargs,
    ):
        if bandwidth is None:
            bandwidth = sampling_frequency / factor

        lpf = zea.ops.get_low_pass_iq_filter(
            self.num_taps,
            ops.convert_to_numpy(sampling_frequency).item(),
            center_frequency,
            bandwidth,
        )
        kwargs.pop("fir_filter_taps", None)  # Remove any existing fir_filter_taps
        return super().call(fir_filter_taps=lpf, **kwargs)


class HistogramMatching(zea.ops.Operation):
    """Histogram matching operation."""

    def __init__(self, reference_image, dynamic_range, **kwargs):
        super().__init__(**kwargs, jittable=False)
        self.reference_image = reference_image
        self.dynamic_range = dynamic_range

    def call(self, **kwargs):
        from skimage import exposure  # pip install scikit-image

        image = kwargs[self.key]

        matched_image = exposure.match_histograms(image, self.reference_image)

        return {self.output_key: matched_image, "dynamic_range": self.dynamic_range}


class HistogramMatchingForModel(HistogramMatching):
    def __init__(self, config_path: str, frame_idx: int = 0, **kwargs):
        config = zea.Config.from_yaml(config_path)
        data_paths = zea.set_data_paths("/ulsa/users.yaml")  # TODO hardcoded
        dataset_folder = data_paths.data_root / config.data.train_folder
        files = Path(dataset_folder).glob("*.hdf5")
        reference_path = next(iter(files))
        with zea.File(reference_path) as file:
            reference_image = file.load_data(config.data.hdf5_key, indices=frame_idx)
        super().__init__(reference_image, **kwargs)


class LogCompressNoClip(zea.ops.Operation):
    """Logarithmic compression of data."""

    def call(self, **kwargs):
        data = kwargs[self.key]

        small_number = ops.convert_to_tensor(1e-16, dtype=data.dtype)
        data = ops.where(data == 0, small_number, data)
        compressed_data = 20 * ops.log10(data)

        return {self.output_key: compressed_data}


def lines_rx_apo(n_tx, grid_size_z, grid_size_x):
    """
    Create a receive apodization for line scanning.
    This is a simple apodization that applies a uniform weight to all elements.

    Returns:
        rx_apo: np.ndarray of shape (n_tx, grid_size_z, grid_size_x)
    """
    assert grid_size_x % n_tx == 0, (
        "grid_size_x must be divisible by n_tx for this apodization scheme."
    )
    step = grid_size_x // n_tx
    rx_apo = np.zeros((n_tx, grid_size_z, grid_size_x), dtype=np.float32)
    for tx, line in zip(range(n_tx), range(0, grid_size_x, step)):
        rx_apo[tx, :, line : line + step] = 1.0
    rx_apo = rx_apo.reshape((n_tx, -1))
    return rx_apo[..., None]  # shape (n_tx, n_pix, 1)
