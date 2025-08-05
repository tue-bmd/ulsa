"""

Prototypes for additional `zea` selection functions

"""

import jax
import numpy as np
from keras import ops

from ulsa.downstream_task import DifferentiableDownstreamTask, downstream_task_registry
from zea.agent import masks
from zea.agent.selection import GreedyEntropy, LinesActionModel
from zea.backend.autograd import AutoGrad
from zea.internal.registry import action_selection_registry


@action_selection_registry(name="downstream_task_selection")
class DownstreamTaskSelection(GreedyEntropy):
    """
    Select lines to maximize information gain with respect to a
    downstream task outcome.
    """

    def __init__(
        self,
        n_actions: int,
        n_possible_actions: int,
        img_width: int,
        img_height: int,
        downstream_task_key: str,
        # downstream_task_output_shape: tuple,
        mean: float = 0,
        std_dev: float = 1,
        num_lines_to_update: int = 5,
    ):
        """
        downstream_task_model should be differentiable
        """
        super().__init__(
            n_actions,
            n_possible_actions,
            img_width,
            img_height,
            mean,
            std_dev,
            num_lines_to_update,
        )
        # NOTE: for echonet don't forget to wrap scan convert in here
        # TODO: batch_size=4 shouldn't be hard coded
        self.downstream_task: DifferentiableDownstreamTask = (
            downstream_task_registry[downstream_task_key]
        )(batch_size=4)

    def compute_output_and_saliency_propagation_hutchinson(self, particles, key):
        num_hutchinson_samples = 5

        z_squared = ops.zeros(self.downstream_task.output_shape)
        for i in range(num_hutchinson_samples):  # TODO: vmap
            subkey = jax.random.fold_in(key, i)
            v = jax.random.normal(
                subkey, self.downstream_task
            )  # TODO: downstream task output shape?
            # vjp returns a function that computes J^T v
            _, vjp_fun = jax.vjp(self.downstream_task.call_differentiable, particles)
            jt_v = vjp_fun(v)[0]  # [0] to get the input tangent
            z_squared += jt_v**2

        posterior_variance = ops.expand_dims(ops.var(particles, axis=0), axis=0)

        return (
            posterior_variance * z_squared
        )  # TODO: where does z_squared get rid of the particle dim?

    def compute_output_and_saliency_propagation_summed(self, particles):
        autograd = AutoGrad()

        def call_model(model_input):
            model_output = self.downstream_task.call_differentiable(model_input)
            return ops.sum(model_output)

        autograd.set_function(call_model)
        echonet_grad_and_value_fn = autograd.get_gradient_and_value_jit_fn()
        grads, _ = echonet_grad_and_value_fn(particles)

        posterior_variance = ops.expand_dims(ops.var(particles, axis=0), axis=0)
        mean_absolute_jacobian = ops.expand_dims(
            ops.mean(ops.abs(grads), axis=0), axis=0
        )
        return posterior_variance * mean_absolute_jacobian

    def sum_neighbouring_columns_into_n_possible_actions(self, full_linewise_salience):
        full_image_width = ops.shape(full_linewise_salience)[1]
        assert full_image_width % self.n_possible_actions == 0, (
            "n_possible_actions must divide evenly into image width"
        )
        stacked_linewise_salience = ops.reshape(
            full_linewise_salience,
            (self.n_possible_actions, full_image_width // self.n_possible_actions),
        )
        return ops.sum(stacked_linewise_salience, axis=1)[None, ...]

    def sample(self, particles):
        """Sample the action using the greedy entropy method.

        Args:
            particles (Tensor): Particles of shape (batch_size, n_particles, height, width)

        Returns:
            Tuple[Tensor, Tensor]:
                - Newly selected lines as k-hot vectors, shaped (batch_size, n_possible_actions)
                - Masks of shape (batch_size, img_height, img_width)
        """
        particles = ops.expand_dims(particles[0], axis=-1)  # remove batch, add channel
        pixelwise_contribution_to_var_dst = (
            self.compute_output_and_saliency_propagation_summed(particles)
        )
        linewise_contribution_to_var_dst = ops.sum(
            pixelwise_contribution_to_var_dst[..., 0], axis=1
        )
        linewise_contribution_to_var_dst = (
            self.sum_neighbouring_columns_into_n_possible_actions(
                linewise_contribution_to_var_dst
            )
        )

        # Greedily select best line, reweight entropies, and repeat
        all_selected_lines = []
        for _ in range(self.n_actions):
            max_contribution_line, linewise_contribution_to_var_dst = (
                ops.vectorized_map(
                    self.select_line_and_reweight_entropy,
                    linewise_contribution_to_var_dst,
                )
            )
            all_selected_lines.append(max_contribution_line)

        selected_lines_k_hot = ops.any(
            ops.one_hot(
                all_selected_lines, self.n_possible_actions, dtype=masks._DEFAULT_DTYPE
            ),
            axis=0,
        )
        return (
            selected_lines_k_hot,
            self.lines_to_im_size(selected_lines_k_hot),
            pixelwise_contribution_to_var_dst,
        )


@action_selection_registry(name="greedy_variance")
class GreedyVariance(GreedyEntropy):
    """
    Greedily select the lines with maximum variance.
    """

    def __init__(
        self,
        n_actions: int,
        n_possible_actions: int,
        img_width: int,
        img_height: int,
        mean: float = 0,
        std_dev: float = 1,
        num_lines_to_update: int = 5,
        average_across_batch: bool = False,
    ):
        super().__init__(
            n_actions,
            n_possible_actions,
            img_width,
            img_height,
            mean,
            std_dev,
            num_lines_to_update,
        )
        self.average_across_batch = average_across_batch

    def sample(self, particles):
        """Sample the action using the greedy variance method.

        Args:
            particles (Tensor): Particles of shape (batch_size, n_particles, height, width)

        Returns:
            Tuple[Tensor, Tensor]:
                - Newly selected lines as k-hot vectors, shaped (batch_size, n_possible_actions)
                - Masks of shape (batch_size, img_height, img_width)
        """
        pixelwise_variance = ops.var(particles, axis=1)  # [batch_size, height, width]
        # NOTE: we compute the linewise variance as the sum of pixelwise variances.
        # Since we're using variance as our entropy, we have that H(X_1, X_2, ...) = H(X_1) + H(X_2) + ...
        linewise_variance = ops.sum(pixelwise_variance, axis=1)  # [batch_size, width]
        if self.average_across_batch:
            linewise_variance = ops.mean(
                linewise_variance, axis=1
            )  # for 3d case [1, width]

        # Greedily select best line, reweight entropies, and repeat
        all_selected_lines = []
        for _ in range(self.n_actions):
            max_contribution_line, linewise_variance = ops.vectorized_map(
                self.select_line_and_reweight_entropy, linewise_variance
            )
            all_selected_lines.append(max_contribution_line)

        selected_lines_k_hot = ops.any(
            ops.one_hot(
                all_selected_lines, self.n_possible_actions, dtype=masks._DEFAULT_DTYPE
            ),
            axis=0,
        )
        return selected_lines_k_hot, self.lines_to_im_size(selected_lines_k_hot)


@action_selection_registry(name="greedy_entropy_univariate_gaussian")
class GreedyEntropyUnivariateGaussian(GreedyEntropy):
    """
    Greedily select the lines with maximum variance.
    """

    def __init__(
        self,
        n_actions: int,
        n_possible_actions: int,
        img_width: int,
        img_height: int,
        mean: float = 0,
        std_dev: float = 1,
        num_lines_to_update: int = 5,
        average_across_batch: bool = False,
    ):
        super().__init__(
            n_actions,
            n_possible_actions,
            img_width,
            img_height,
            mean,
            std_dev,
            num_lines_to_update,
        )
        self.average_across_batch = average_across_batch

    def sample(self, particles):
        """Sample the action using the greedy entropy method with univariate Gaussian assumption.
        See 'Entropy' here: https://en.wikipedia.org/wiki/Normal_distribution
        NOTE: this is designed for line-based subsampling, i.e. where self.img_height = |A^l|
            - See `derivations/Pixelwise_Gaussian_Entropy.pdf` for more details.

        Args:
            particles (Tensor): Particles of shape (batch_size, n_particles, height, width)

        Returns:
            Tuple[Tensor, Tensor]:
                - Newly selected lines as k-hot vectors, shaped (batch_size, n_possible_actions)
                - Masks of shape (batch_size, img_height, img_width)
        """
        # NOTE: we don't really need this constant offset since we take the argmax, and
        # |A^l| doesn't change across l, but we keep it here for completeness.
        constant_offset = self.img_height * 0.5 * ops.log(2 * np.pi * np.e)
        pixelwise_variance = ops.var(particles, axis=1)
        pixelwise_entropy = constant_offset + 0.5 * ops.log(
            pixelwise_variance
        )  # [batch_size, height, width]
        # NOTE: we compute the linewise variance as the sum of pixelwise variances.
        # Since we're using variance as our entropy, we have that H(X_1, X_2, ...) = H(X_1) + H(X_2) + ...
        linewise_entropy = ops.sum(pixelwise_entropy, axis=1)  # [batch_size, width]
        if self.average_across_batch:
            linewise_entropy = ops.mean(
                linewise_entropy, axis=1
            )  # for 3d case [1, width]

        # Greedily select best line, reweight entropies, and repeat
        all_selected_lines = []
        for _ in range(self.n_actions):
            max_contribution_line, linewise_entropy = ops.vectorized_map(
                self.select_line_and_reweight_entropy, linewise_entropy
            )
            all_selected_lines.append(max_contribution_line)

        selected_lines_k_hot = ops.any(
            ops.one_hot(
                all_selected_lines, self.n_possible_actions, dtype=masks._DEFAULT_DTYPE
            ),
            axis=0,
        )
        return selected_lines_k_hot, self.lines_to_im_size(selected_lines_k_hot)


def selector_from_name(name: str, **kwargs) -> LinesActionModel:
    """Get the action selection model from its name."""
    assert name in action_selection_registry, f"Unknown action selection model: {name}"

    # Override zea defaults
    if name == "covariance":
        assert "covariance" in action_selection_registry, (
            "Covariance sampling is not registered."
        )
        kwargs["n_masks"] = int(1e5)

    return action_selection_registry[name](**kwargs)
