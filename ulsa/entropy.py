"""Implements entropy calculation taken from `zea.agent.selection.GreedyEntropy` but simplified
and working for N-D data."""

from keras import ops


def pairwise_pixel_gaussian_error(particles, entropy_sigma=1.0):
    """Compute the pairwise pixelwise Gaussian error.

    This function computes the Gaussian error between each pair of pixels in the
    set of particles provided. This can be used to approximate the entropy of
    a Gaussian mixture model, where the particles are the means of the Gaussians.
    For more details see Section 4 here: https://arxiv.org/abs/2406.14388

    Args:
        particles (Tensor): Particles of shape (batch_size, n_particles, *pixels)

    Returns:
        Tensor: batch of pixelwise pairwise Gaussian errors,
            of shape (batch_size, n_particles, n_particles, *pixels)
    """
    assert particles.shape[1] > 1, (
        "The entropy cannot be approximated using a single particle."
    )
    particles = ops.cast(particles, "float32")

    # TODO: I think we only need to compute the lower triangular
    # of this matrix, since it's symmetric
    squared_l2_error_matrices = (
        particles[:, :, None, ...] - particles[:, None, :, ...]
    ) ** 2
    gaussian_error_per_pixel_i_j = ops.exp(
        -(squared_l2_error_matrices) / (2 * entropy_sigma**2)
    )
    # [batch_size, n_particles, n_particles, *pixels]
    return gaussian_error_per_pixel_i_j


def pixelwise_entropy(particles, entropy_sigma=1.0):
    """
    This function computes the entropy for each line using a Gaussian Mixture Model
    approximation of the posterior distribution.
    For more details see Section VI. B here: https://arxiv.org/pdf/2410.13310

    Args:
        particles (Tensor): Particles of shape (batch_size, n_particles, *pixels)

    Returns:
        Tensor: batch of entropies per pixel of shape (batch_size, *pixels)
    """
    n_particles = ops.shape(particles)[1]
    gaussian_error_per_pixel_stacked = pairwise_pixel_gaussian_error(
        particles, entropy_sigma
    )
    # sum out first dimension of (n_particles x n_particles) error matrix
    # [n_particles, batch, height, width]
    pixelwise_entropy_sum_j = ops.sum(
        (1 / n_particles) * gaussian_error_per_pixel_stacked, axis=1
    )
    log_pixelwise_entropy_sum_j = ops.log(pixelwise_entropy_sum_j)
    # sum out second dimension of (n_particles x n_particles) error matrix
    # [batch, height, width]
    pixelwise_entropy = -ops.sum(
        (1 / n_particles) * log_pixelwise_entropy_sum_j, axis=1
    )
    return pixelwise_entropy
