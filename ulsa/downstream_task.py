from abc import ABC

import cv2
import jax
import numpy as np
from keras import ops

import zea
from ulsa.io_utils import deg2rad
from zea.backend.autograd import AutoGrad
from zea.display import compute_scan_convert_2d_coordinates, scan_convert_2d
from zea.internal.registry import RegisterDecorator
from zea.models.echonet import EchoNetDynamic

downstream_task_registry = RegisterDecorator(items_to_register=["name"])


def compute_dice_score(pred_mask, gt_mask, eps=1e-8):
    """
    Compute DICE score between prediction and ground truth masks.

    Args:
        pred_mask: (N, H, W, 1) array of N hypothesis masks
        gt_mask: (1, H, W, 1) array of single ground truth mask
    """
    assert pred_mask.dtype == np.uint8
    assert gt_mask.dtype == np.uint8
    intersection = np.sum(pred_mask * gt_mask, axis=(-1, -2, -3))
    sum_pred = np.sum(pred_mask, axis=(-1, -2, -3))
    sum_gt = np.sum(gt_mask, axis=(-1, -2, -3))

    union = sum_pred + sum_gt
    dice_scores = np.where(union > 0, 2.0 * intersection / (union + eps), 1.0)

    return dice_scores


def overlay_segmentation_on_image_batch(images, segmentation_masks, alpha=0.3):
    """
    Overlay segmentation masks in red on a batch of grayscale images.
    Only blend in the mask regions, leave the rest of the image unchanged.
    images: (N, H, W) uint8
    segmentation_masks: (N, H, W) uint8, values 0 or 255
    Returns: (N, H, W, 3) uint8
    """
    assert images.shape == segmentation_masks.shape, (
        "Images and masks must have same shape"
    )
    images_rgb = np.stack([images, images, images], axis=-1)  # (N, H, W, 3)
    red_overlay = np.zeros_like(images_rgb)
    red_overlay[..., 0] = segmentation_masks  # Red channel

    mask_bool = segmentation_masks > 127  # (N, H, W)
    mask_bool = np.repeat(mask_bool[..., np.newaxis], 3, axis=-1)  # (N, H, W, 3)

    blended = images_rgb.copy().astype(np.float32)
    blended[mask_bool] = (1 - alpha) * images_rgb[mask_bool] + alpha * red_overlay[
        mask_bool
    ]
    return blended.astype(np.uint8)


def map_range(img, from_range=(-1, 1), to_range=(0, 255)):
    img = ops.convert_to_numpy(img)
    img = zea.func.translate(img, from_range, to_range)
    return np.clip(img, to_range[0], to_range[1])


class DownstreamTask(ABC):
    def __init__(self):
        pass

    def __call__(self, x):
        pass

    def call_generic(self, x):
        pass

    def output_type(self):
        pass


class DifferentiableDownstreamTask(DownstreamTask):
    def call_differentiable(self):
        pass


class NoDownstreamTask(DownstreamTask):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return ops.zeros(1)

    def name(self):
        return "no_downstream_task"


@downstream_task_registry(name="echonet_segmentation")
class EchoNetSegmentation(DifferentiableDownstreamTask):
    def __init__(self, batch_size=4):
        super().__init__()

        self.model = EchoNetDynamic.from_preset("echonet-dynamic")
        self.model.build(input_shape=(batch_size, 112, 112, 3))

        # Scan conversion constants for echonet
        self.rho_range = (0, 112)
        self.theta_range = (deg2rad(-45), deg2rad(45))
        self.fill_value = -1.0
        self.resolution = 1.0  # mm per pixel

        self.n_rho = 112
        self.n_theta = 112

        self.init_scan_conversion()
        # TODO: can we infer output shape from model?
        self.output_shape = (112, 112, 1)

    def init_scan_conversion(self):
        # Precompute interpolation coordinates using the new display.py functions
        image_shape = (self.n_rho, self.n_theta)
        self.coords, _ = compute_scan_convert_2d_coordinates(
            image_shape,
            self.rho_range,
            self.theta_range,
            self.resolution,
        )
        # Output shape is determined by the coordinates shape (2, grid_size_z, grid_size_x)
        self.output_shape = self.coords.shape[1:]  # (grid_size_z, grid_size_x)

        def _scan_convert_2d(image, fill_value=self.fill_value, order=0):
            image = ops.convert_to_tensor(image, dtype="float32")
            image_sc, _ = scan_convert_2d(
                image, coordinates=self.coords, fill_value=fill_value, order=order
            )
            return image_sc

        self.scan_convert_2d = _scan_convert_2d

    def postprocess_for_visualization(self, images, masks):
        images = self.scan_convert_batch(images[..., None])[..., 0]
        images = map_range(images, from_range=(-1, 1), to_range=(0, 255)).astype(
            np.uint8
        )
        masks = map_range(masks, from_range=(0, 1), to_range=(0, 255)).astype(np.uint8)
        images_with_masks = overlay_segmentation_on_image_batch(images, masks)
        images_with_masks = np.pad(
            images_with_masks,
            pad_width=((0, 0), (0, 0), (23, 24), (0, 0)),
            mode="constant",
        )
        return images_with_masks

    def scan_convert_batch(self, x):
        # x: (batch, n_rho, n_theta, 1)
        x_cartesian = ops.vectorized_map(self.scan_convert_2d, x[..., 0])
        x_cartesian_cropped = x_cartesian[:, :, 23 : 159 - 24, None]
        return x_cartesian_cropped

    def call_generic(self, x):
        x_sc = self.scan_convert_batch(x)
        masks = self(x_sc)
        return masks

    def __call__(self, x_sc):
        # echonet expects 3-channel input
        x_sc = ops.tile(x_sc, [1, 1, 1, 3])
        logits = self.model.network(x_sc)
        # return ops.sigmoid(logits)
        return ops.cast(logits > 0.0, dtype="uint8")

    def call_differentiable(self, x):
        x_sc = self.scan_convert_batch(x)
        # echonet expects 3-channel input
        x_sc = ops.tile(x_sc, [1, 1, 1, 3])
        logits = self.model.network(x_sc)
        return ops.sigmoid(logits)

    def compute_output_and_saliency_propagation_hutchinson(self, x):
        def call_echonet(posterior_samples):
            x_scan_converted = self.scan_convert_batch(posterior_samples)
            mask_logits = self.call_differentiable(x_scan_converted)
            return mask_logits

        key = jax.random.PRNGKey(0)  # TODO: pass in global key?
        num_hutchinson_samples = 5

        mask_logits = call_echonet(x)

        z_squared = ops.zeros_like(mask_logits)
        for i in range(num_hutchinson_samples):
            subkey = jax.random.fold_in(key, i)
            v = jax.random.normal(subkey, mask_logits.shape)
            # vjp returns a function that computes J^T v
            _, vjp_fun = jax.vjp(call_echonet, x)
            jt_v = vjp_fun(v)[0]  # [0] to get the input tangent
            z_squared += jt_v**2

        posterior_variance = ops.expand_dims(ops.var(x, axis=0), axis=0)

        return mask_logits, posterior_variance * z_squared

    def compute_output_and_saliency_propagation_summed(self, x):
        autograd = AutoGrad()

        def call_echonet(posterior_samples):
            x_scan_converted = self.scan_convert_batch(posterior_samples)
            mask_logits = self.call_differentiable(x_scan_converted)
            return ops.sum(mask_logits), mask_logits

        autograd.set_function(call_echonet)
        echonet_grad_and_value_fn = autograd.get_gradient_and_value_jit_fn(has_aux=True)
        grads, (_, mask_logits) = echonet_grad_and_value_fn(x)

        posterior_variance = ops.expand_dims(ops.var(x, axis=0), axis=0)
        mean_absolute_jacobian = ops.expand_dims(
            ops.mean(ops.abs(grads), axis=0), axis=0
        )
        return mask_logits, posterior_variance * mean_absolute_jacobian

    def compute_output_and_saliency_gradient_descent(self, x):
        autograd = AutoGrad()

        def call_echonet(posterior_samples):
            x_scan_converted = self.scan_convert_batch(posterior_samples)
            mask_logits = self.call_differentiable(x_scan_converted)
            return ops.sum(ops.var(mask_logits, axis=0)), mask_logits

        autograd.set_function(call_echonet)
        echonet_grad_and_value_fn = autograd.get_gradient_and_value_jit_fn(has_aux=True)

        grads, (_, mask_logits) = echonet_grad_and_value_fn(x)

        posterior_mean = ops.expand_dims(ops.mean(x, axis=0), axis=0)
        expected_change = -(x - posterior_mean)
        expected_effects_of_change = expected_change * grads
        saliency = (
            -expected_effects_of_change
        )  # negative because we want to minimize the variance
        return mask_logits, ops.expand_dims(ops.mean(saliency, axis=0), axis=0)

    def get_compute_output_and_saliency_fn(self, selection_strategy):
        if selection_strategy == "downstream_propagation_summed":
            return self.compute_output_and_saliency_propagation_summed
        elif selection_strategy == "downstream_propagation_hutchinson":
            return self.compute_output_and_saliency_propagation_hutchinson
        elif selection_strategy == "downstream_gradient_descent":
            return self.compute_output_and_saliency_gradient_descent
        else:
            raise UserWarning(
                "Unknown downstream selection strategy for echonet segmentation"
            )

    def name(self):
        return "echonet_segmentation"

    def output_type(self):
        return "segmentation_mask"

    def beliefs_to_reconstruction(self, belief_distributions):
        """
        This function maps a set of beliefs about the DST output to
        a single reconstruction.

        Params:
        - belief_distributions (Tensor of shape [batch, n_particles, ...])
        """
        # NOTE: we choose 0.5 here as a hard-coded cutoff for the proportion of
        # beliefs that need to agree in order for a pixel to be included in the
        # reconstruction mask.
        # reconstructions = ops.cast(((ops.mean(belief_distributions, axis=1)) > 0.5), "uint8")

        # NOTE: here we use 'choose first' reconstruction in order to get a segmentation
        # that is consistent with the first particle video
        reconstructions = ops.cast((belief_distributions[:, 0, ...] > 0.5), "uint8")
        return reconstructions
