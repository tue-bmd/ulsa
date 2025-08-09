from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from keras import ops

import zea.ops
from zea.utils import translate

NOISE_ESTIMATION_NORMALIZER = (
    0.6745  # Used for robust noise estimation from median absolute deviation
)


def soft_threshold(data, value, substitute=0):
    magnitude = ops.absolute(data)

    # divide by zero okay as np.inf values get clipped, so ignore warning.
    thresholded = 1 - value / magnitude
    ops.clip(thresholded, 0, None)
    thresholded = data * thresholded

    if substitute == 0:
        return thresholded
    else:
        cond = ops.less(magnitude, value)
        return ops.where(cond, substitute, thresholded)


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
    import jaxwt as jwt  # pip install jaxwt
    # import pywt  # pip install PyWavelets

    rf_signal = rf_signal[None]  # add batch dimension

    # Decompose
    coeffs = jwt.wavedec(rf_signal, wavelet, level=level)

    # Estimate noise from the detail coefficients at the highest level
    sigma = ops.median(ops.abs(coeffs[-1])) / NOISE_ESTIMATION_NORMALIZER
    threshold = threshold_factor * sigma * ops.sqrt(2 * ops.log(rf_signal.shape[-1]))

    # Threshold detail coefficients in parallel using jax.vmap
    new_coeffs = [coeffs[0]]  # Keep approximation unaltered

    def threshold_fn(c):
        return soft_threshold(c, threshold)

    # Use jax.vmap to parallelize over the list of detail coefficients
    new_coeffs += list(jax.tree.map(threshold_fn, coeffs[1:]))

    # Reconstruct signal
    out = jwt.waverec(new_coeffs, wavelet)

    return ops.squeeze(out, axis=0)  # remove batch dimension


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
    return apply_along_axis(lambda x: wavelet_denoise_rf(x, **kwargs), axis, data)


class GetAutoDynamicRange(zea.ops.Operation):
    """Compute dynamic range based on percentiles of the input data.
    Only works when dynamic range is not already set in the parameters."""

    def __init__(self, low_pct=18, high_pct=95, exclude_zeros=True, **kwargs):
        super().__init__(input_data_type=zea.ops.DataTypes.ENVELOPE_DATA, **kwargs)
        self.low_pct = low_pct
        self.high_pct = high_pct
        self.exclude_zeros = exclude_zeros

    def call(self, dynamic_range=None, **kwargs):
        data = kwargs[self.key]

        if dynamic_range is not None:
            vmin, vmax = dynamic_range[0], dynamic_range[1]
        else:
            vmin, vmax = None, None

        # Exclude zeros, useful for active scan-line selection :)
        if self.exclude_zeros:
            data = jnp.where(data != 0, data, jnp.nan)

        if vmin is None:
            vmin = jnp.nanquantile(data, self.low_pct / 100)
            vmin = 20 * ops.log10(vmin)
        if vmax is None:
            vmax = jnp.nanquantile(data, self.high_pct / 100)
            vmax = 20 * ops.log10(vmax)

        return {"dynamic_range": [vmin, vmax]}


class TranslateDynamicRange(zea.ops.Operation):
    """Translate data from one range to another.

    Can be disabled by setting `range_to=None`.
    """

    def __init__(self, range_to, **kwargs):
        super().__init__(**kwargs)
        self.range_to = range_to

    def call(self, dynamic_range=None, **kwargs):
        data = kwargs[self.key]
        if self.range_to is not None:
            data = translate(data, dynamic_range, self.range_to)
        return {self.output_key: data}


class ExpandDims(zea.ops.Operation):
    """Expand dimensions of the input data."""

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, **kwargs):
        data = kwargs[self.key]
        expanded_data = ops.expand_dims(data, axis=self.axis)
        return {self.output_key: expanded_data}


class Squeeze(zea.ops.Operation):
    """Squeeze dimensions of the input data."""

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, **kwargs):
        data = kwargs[self.key]
        squeezed_data = ops.squeeze(data, axis=self.axis)
        return {self.output_key: squeezed_data}


class Resize(zea.ops.Operation):
    """Resize the input data to a specified shape."""

    def __init__(self, size, interpolation="bilinear", antialias=True, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.interpolation = interpolation
        self.antialias = antialias

    def call(self, **kwargs):
        data = kwargs[self.key]
        resized_data = ops.image.resize(
            data,
            size=self.size,
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        return {self.output_key: resized_data}


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
    def __init__(self, wavelet="db4", level=4, threshold_factor=0.1, axis=-3, **kwargs):
        super().__init__(**kwargs)
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


def match_histogram_fn(src, target):
    """
    Return a function that matches the histogram of src to target. Can be used to compute a
    matching based on a region of interest (ROI) in the source image, and apply it to the
    entire source image.
    """
    # Get sorted unique values and their counts from the ROI of the source image
    src_values, src_counts = np.unique(src.ravel(), return_counts=True)
    # Get sorted unique values and their counts from the entire target image
    target_values, target_counts = np.unique(target.ravel(), return_counts=True)

    # Compute the cumulative distribution function (CDF) for the ROI of the source
    src_cdf = np.cumsum(src_counts).astype(np.float64)
    src_cdf /= src_cdf[-1]
    # Compute the CDF for the target image
    target_cdf = np.cumsum(target_counts).astype(np.float64)
    target_cdf /= target_cdf[-1]

    # Interpolate to find the target values that correspond to the quantiles of the source ROI
    interp_t_values = np.interp(src_cdf, target_cdf, target_values)

    def _match_histogram(src):
        """Match histogram of src to target using the computed mapping."""
        # Map all pixels in the source image to the new values using linear interpolation
        matched = np.interp(src.ravel(), src_values, interp_t_values)
        # Reshape to the original image shape and cast to the original dtype
        return matched.reshape(src.shape).astype(src.dtype)

    return _match_histogram


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
