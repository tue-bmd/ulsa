from keras import ops

from zea.internal.registry import metrics_registry
from zea.metrics import mse


@metrics_registry(name="rmse", paired=True)
def rmse(y_true, y_pred):
    """Gives the root mean square error (RMSE) for two input tensors.

    Reference:
        https://en.wikipedia.org/wiki/Root_mean_square_deviation

    Args:
        y_true (tensor)
        y_pred (tensor)

    Returns:
        (float): root mean squared error between y_true and y_pred.
    """
    return ops.sqrt(mse(y_true, y_pred))


@metrics_registry(name="nrmse", paired=True)
def nrmse(y_true, y_pred, image_range):
    """Gives the normalized root mean square error (NRMSE) for two input tensors.

    Reference:
        https://en.wikipedia.org/wiki/Root_mean_square_deviation#Normalization

    Args:
        y_true (tensor)
        y_pred (tensor)
        image_range (list): [min, max] pixel values of the images.

    Returns:
        (float): root mean squared error between y_true and y_pred.
    """
    return ops.sqrt(mse(y_true, y_pred)) / (image_range[1] - image_range[0])
