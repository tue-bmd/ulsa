import numpy as np
from keras import ops

from zea.internal.registry import metrics_registry
from zea.metrics import gcnr, mse


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


def gcnr_per_frame(images, mask1, mask2):
    """
    Calculate gCNR for each frame in the images array.

    Parameters:
    - images: numpy array of shape (frames, h, w)
    - mask1: boolean mask for the first region of shape (frames, h, w)
    - mask2: boolean mask for the second region of shape (frames, h, w)

    Returns:
    - List of gCNR values for each frame
    """

    def single_gcnr(img, m1, m2):
        return gcnr(img[m1], img[m2])

    vectorized_gcnr = np.vectorize(single_gcnr, signature="(h,w),(h,w),(h,w)->()")
    return vectorized_gcnr(images, mask1, mask2)
