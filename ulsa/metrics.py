import dm_pix as pix
import keras
import numpy as np
from keras import ops

import zea
from zea import log
from zea.metrics import gcnr as generalized_contrast_to_noise_ratio
from zea.models.lpips import LPIPS
from zea.utils import translate


def mean_squared_error(y_true, y_pred, **kwargs):
    """Gives the MSE for two input tensors.
    Args:
        y_true (tensor)
        y_pred (tensor)
    Returns:
        (float): mean squared error between y_true and y_pred. L2 loss.

    """
    return reduce_mean(ops.square(y_true - y_pred))


def mean_absolute_error(y_true, y_pred, **kwargs):
    """Gives the MAE for two input tensors.
    Args:
        y_true (tensor)
        y_pred (tensor)
    Returns:
        (float): mean absolute error between y_true and y_pred. L1 loss.

    """
    return reduce_mean(ops.abs(y_true - y_pred))


def peak_signal_to_noise_ratio(y_true, y_pred, max_val=255, **kwargs):
    """Gives the Peak Signal to Noise Ratio (PSNR) for two input tensors.
    Args:
        y_true (tensor): [None, height, width, channels]
        y_pred (tensor): [None, height, width, channels]
        max_val: The dynamic range of the images

    Returns:
        (float): peak signal to noise ratio of y_true and y_pred.

        psnr = 20 * log10(max_val / sqrt(MSE(y_true, y_pred)))
    """
    mse = reduce_mean(ops.square(y_true - y_pred))
    psnr = 20 * ops.log10(max_val) - 10 * ops.log10(mse)
    return psnr


def get_lpips(image_range, batch_size=128, clip=False):
    """
    Get the Learned Perceptual Image Patch Similarity (LPIPS) metric.

    Args:
        image_range (list): The range of the images. Will be translated to [-1, 1] for LPIPS.
        batch_size (int): The batch size for the LPIPS model.
        clip (bool): Whether to clip the images to `image_range`.

    Returns:
        The LPIPS metric function which can be used with [..., h, w, c] tensors in
        the range `image_range`.
    """
    # Get the LPIPS model
    _lpips = LPIPS.from_preset("lpips")
    _lpips.trainable = False
    _lpips.disable_checks = True

    def unstack_lpips(imgs):
        """Unstack the images and calculate the LPIPS metric."""
        img1, img2 = ops.unstack(imgs, num=2, axis=-1)
        return _lpips([img1, img2])

    def lpips(img1, img2, **kwargs):
        """
        The LPIPS metric function.
        Args:
            img1 (tensor) with shape (..., h, w, c)
            img2 (tensor) with shape (..., h, w, c)
        Returns (float): The LPIPS metric between img1 and img2 with shape [...]
        """
        # clip and translate images to [-1, 1]
        if clip:
            img1 = ops.clip(img1, *image_range)
            img2 = ops.clip(img2, *image_range)
        img1 = translate(img1, image_range, [-1, 1])
        img2 = translate(img2, image_range, [-1, 1])

        imgs = ops.stack([img1, img2], axis=-1)
        n_batch_dims = ops.ndim(img1) - 3
        return zea.tensor_ops.func_with_one_batch_dim(
            unstack_lpips, imgs, n_batch_dims, batch_size=batch_size
        )

    return lpips


METRIC_FUNCS = dict(
    mse=mean_squared_error,
    mae=mean_absolute_error,
    psnr=peak_signal_to_noise_ratio,
    ssim=pix.ssim,
    lpips=get_lpips,
    gcnr=generalized_contrast_to_noise_ratio,
)

MINIMIZE = dict(
    mse=True,
    mae=True,
    psnr=False,
    ssim=False,
    lpips=True,
    gcnr=False,
)

SUPERVISED = dict(
    mse=True,
    mae=True,
    psnr=True,
    ssim=True,
    lpips=True,
    gcnr=False,
)


def reduce_mean(array, keep_batch_dim=True):
    """Reduce array by taking the mean.
    Preserves batch dimension if keep_batch_dim=True.
    """
    if keep_batch_dim:
        axis = len(array.shape)
        axis = tuple(range(axis)[-3:])
    else:
        axis = None
    return ops.mean(array, axis=axis)


class Metrics:
    """Class for evaluating metrics."""

    def __init__(
        self,
        metrics: list,
        image_range: list,
        supervised: bool = True,
        **kwargs,
    ):
        """
        Initializes the Metrics class.

        Args:
            metrics (list): A list of metrics to be used.
            image_range (list): A list specifying the range of images.
            supervised (bool, optional): Whether to use supervised metrics or not.
                Defaults to True. Supervised metrics require ground truth images.
            **kwargs: Additional keyword arguments. These are passed to the metric functions
                that start with "get_" in `METRIC_FUNCS` and need to be initialized.
                For instance `get_learned_perceptual_image_patch_similarity` requires the
                `image_shape` argument.

        Raises:
            AssertionError: If a metric in `metrics` is not found in `METRIC_FUNCS`.

        Notes:
            - If `metrics` is a string and equals "all", all metrics in `METRIC_FUNCS` will be used.
            - If `metrics` is a string and not "all", only the specified metric will be used.
            - Each metric function in `METRIC_FUNCS` that starts with "get_" will be initialized.
            - The `minimize` dictionary links each metric to a boolean value indicating
                whether it should be minimized.

        """

        self.metrics = metrics
        self.image_range = image_range
        self.supervised = supervised

        if isinstance(self.metrics, str):
            if self.metrics == "all":
                self.metrics = list(METRIC_FUNCS.keys())
                # filter on supervised metrics
                self.metrics = [
                    metric
                    for metric in self.metrics
                    if SUPERVISED[metric] == self.supervised
                ]
            else:
                self.metrics = [self.metrics]

        for metric in self.metrics:
            assert metric in METRIC_FUNCS, (
                f"cannot find metric: {metric}, should be in \n"
                f"{list(METRIC_FUNCS.keys())}"
            )
            assert SUPERVISED[metric] == self.supervised, (
                f"metric: {metric} is not a supervised metric"
                if self.supervised
                else f"metric: {metric} is a supervised metric"
            )
            # check in METRIC_FUNCS if metric func starts with get_ and if so
            # initialize the metric function
            if METRIC_FUNCS[metric].__name__.startswith("get_"):
                log.info(f"Initializing metric: {log.green(metric)}")
                init_dict = kwargs.get(metric, {})
                METRIC_FUNCS[metric] = METRIC_FUNCS[metric](
                    image_range=self.image_range, **init_dict
                )

        # link each metric to a bool which specifiec whether it is a
        # metric that should be minimized or not
        self.minimize = {metric: MINIMIZE[metric] for metric in self.metrics}

        # multiply metric with this value such that it can become a loss
        # so minimize objectives stay the same and maximize objective are multiplied with -1
        self.loss_multiplier = {
            metric: np.sign(int(self.minimize[metric]) - 0.5) for metric in self.metrics
        }

    def eval_metrics(
        self,
        y_pred,
        y_true=None,
        dtype="float32",
        add_channel_axis=False,
        average_batch=False,
        to_numpy=True,
        nan_to_num=False,
        batch_axes=None,
        batch_size=None,
        verbose=True,
    ):
        """Evaluate metric on y_true and y_pred.

        Args:
            y_true (ndarray): first input array.
            y_pred (ndarray): second input array.
            dtype (str, optional): cast to dtype. Defaults to float32.
            add_channel_axis (bool, optional): add channel axis to data.
                Defaults to False.
            average_batch (bool, optional): return metric averaged over
                batch dimension. Defaults to False.
            to_numpy (bool, optional): return numpy array instead of
                a tensor. Defaults to True.

        Returns:
            dict: dict with metrics. keys are metric names.
        """
        if batch_axes is not None:
            assert isinstance(batch_axes, (int, list, tuple)), (
                "batch_axes should be an integer or list / tuple of integers"
            )
            if isinstance(batch_axes, int):
                batch_axes = [batch_axes]
            # register shape
            batch_shape = [y_pred.shape[i] for i in batch_axes]

        y_pred = self.prepare_tensor(
            y_pred,
            dtype,
            add_channel_axis,
            nan_to_num=nan_to_num,
            batch_axes=batch_axes,
        )
        if y_true is not None:
            y_true = self.prepare_tensor(
                y_true,
                dtype,
                add_channel_axis,
                nan_to_num=nan_to_num,
                batch_axes=batch_axes,
            )
            assert self.supervised, (
                "supervised is set to False, but y_true is provided. "
                "Can only evaluate supervised metrics when y_true is provided."
            )

        m_dict = {}
        for metric in self.metrics:

            def evaluate_batch(pred, true=None):
                if self.supervised:
                    return METRIC_FUNCS[metric](true, pred, max_val=self.image_range[1])
                return METRIC_FUNCS[metric](pred)

            # Get total number of samples from first dimension
            total_samples = y_pred.shape[0]

            if batch_size is None:
                # Process all at once
                all_evaluations = evaluate_batch(y_pred, y_true)
            else:
                # Process in batches
                log.info(f"Computing {metric.upper()}")
                progbar = keras.utils.Progbar(
                    total_samples, verbose=verbose, unit_name="samples"
                )

                batch_evaluations = []
                for start_idx in range(0, total_samples, batch_size):
                    end_idx = min(start_idx + batch_size, total_samples)
                    batch_pred = y_pred[start_idx:end_idx]
                    batch_true = y_true[start_idx:end_idx] if self.supervised else None
                    batch_evaluations.append(evaluate_batch(batch_pred, batch_true))
                    progbar.update(end_idx)

                all_evaluations = ops.concatenate(batch_evaluations, axis=0)

            if average_batch:
                all_evaluations = ops.mean(all_evaluations, axis=0)
            else:
                # reshape into original batch axes shape
                if batch_axes is not None:
                    all_evaluations = ops.reshape(all_evaluations, batch_shape)

            if to_numpy:
                all_evaluations = ops.convert_to_numpy(all_evaluations)

            m_dict[metric] = all_evaluations

        return m_dict

    def prepare_tensor(
        self,
        tensor,
        dtype="float32",
        add_channel_axis=False,
        nan_to_num=False,
        batch_axes=None,
    ):
        """Prepare tensor for evaluation."""
        tensor = ops.array(tensor)
        if nan_to_num is not False:
            tensor = ops.nan_to_num(tensor, nan=nan_to_num)

        assert ops.all(tensor >= self.image_range[0]) and ops.all(
            tensor <= self.image_range[1]
        ), f"tensor should be within image_range: {self.image_range}"

        tensor = ops.cast(tensor, dtype)
        if batch_axes is not None:
            # move batch axes to front
            # if multiple batch axes are provided, flatten the batch axes
            # also preserve the order of the batch axes
            if isinstance(batch_axes, int):
                batch_axes = [batch_axes]

            for i, axis in enumerate(batch_axes):
                tensor = ops.moveaxis(tensor, axis, i)

            if len(batch_axes) > 1:
                # flatten batch axes
                batch_axes = tuple(tensor.shape[len(batch_axes) :])
                tensor = ops.reshape(tensor, (-1,) + batch_axes)

        if add_channel_axis or len(tensor.shape) == 3:
            tensor = ops.expand_dims(tensor, axis=-1)

        return tensor

    @staticmethod
    def print_results(results, to_screen=True, precision=3):
        strings = []
        for metric, value in results.items():
            string = f"{metric:<10}: {np.mean(value):.{precision}f}"
            strings.append(string)
            if to_screen:
                print(string)

        return ", ".join(str(metric) for metric in strings)

    @staticmethod
    def parse_metrics(metrics, reduce_mean=True):
        """
        Parses and optionally reduces a list of metric dictionaries.

        Args:
            metrics (list of dict): A list where each element is a dictionary containing metric names as keys and their values.
            reduce_mean (bool, optional): If True, computes the mean of each metric's values across the list. Defaults to True.

        Returns:
            dict: A dictionary where each key is a metric name and the value is a list of values (or means if reduce_mean is True).
        """
        metrics = {k: [dic[k] for dic in metrics] for k in metrics[0]}
        if reduce_mean:
            metrics = {k: [np.mean(_v) for _v in v] for k, v in metrics.items()}
        return metrics
