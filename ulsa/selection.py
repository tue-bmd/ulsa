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
        pixelwise_variance = 0.5 * ops.log(
            ((2 * np.pi * np.e) ** self.img_height) * ops.var(particles, axis=1)
        )  # [batch_size, height, width]
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


@action_selection_registry(name="greedy_entropy_fixed")
class GreedyEntropyFixed(LinesActionModel):
    """Greedy entropy action selection.

    Selects the max entropy line and reweights the entropy values around it,
    approximating the decrease in entropy that would occur from observing that line.

    The neighbouring values are decreased by a Gaussian function centered at the selected line.
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
        entropy_sigma: float = 1.0,
    ):
        """Initialize the GreedyEntropy action selection model.

        Args:
            n_actions (int): The number of actions the agent can take.
            n_possible_actions (int): The number of possible actions.
            img_width (int): The width of the input image.
            img_height (int): The height of the input image.
            mean (float, optional): The mean of the RBF. Defaults to 0.
            std_dev (float, optional): The standard deviation of the RBF. Defaults to 1.
            num_lines_to_update (int, optional): The number of lines around the selected line
                to update. Must be odd.
            entropy_sigma (float, optional): The standard deviation of the Gaussian
                Mixture components used to approximate the posterior.
        """
        super().__init__(n_actions, n_possible_actions, img_width, img_height)

        # Number of samples must be odd so that the entropy
        # of the selected line is set to 0 once it's been selected.
        assert num_lines_to_update % 2 == 1, "num_samples must be odd."
        self.num_lines_to_update = num_lines_to_update

        # see here what I mean by upside_down_gaussian:
        # https://colab.research.google.com/drive/1CQp_Z6nADzOFsybdiH5Cag0vtVZjjioU?usp=sharing
        upside_down_gaussian = lambda x: 1 - ops.exp(-0.5 * ((x - mean) / std_dev) ** 2)
        # Sample `num_lines_to_update` points symmetrically around the mean.
        # This can be tuned to determine how the entropy for neighbouring lines is updated
        # TODO: learn this function from training data
        points_to_evaluate = ops.linspace(
            mean - 2 * std_dev,
            mean + 2 * std_dev,
            self.num_lines_to_update,
        )
        self.upside_down_gaussian = upside_down_gaussian(points_to_evaluate)
        self.entropy_sigma = entropy_sigma

    @staticmethod
    def compute_pairwise_pixel_gaussian_error(
        particles, stack_n_cols=1, n_possible_actions=None, entropy_sigma=1
    ):
        """Compute the pairwise pixelwise Gaussian error.

        This function computes the Gaussian error between each pair of pixels in the
        set of particles provided. This can be used to approximate the entropy of
        a Gaussian mixture model, where the particles are the means of the Gaussians.
        For more details see Section 4 here: https://arxiv.org/abs/2406.14388

        Args:
            particles (Tensor): Particles of shape (batch_size, n_particles, height, width)

        Returns:
            Tensor: batch of pixelwise pairwise Gaussian errors,
            of shape (n_particles, n_particles, batch, height, width)
        """
        assert particles.shape[1] > 1, (
            "The entropy cannot be approximated using a single particle."
        )

        if n_possible_actions is None:
            n_possible_actions = particles.shape[-1]

        # TODO: I think we only need to compute the lower triangular
        # of this matrix, since it's symmetric
        squared_l2_error_matrices = (
            particles[:, :, None, ...] - particles[:, None, :, ...]
        ) ** 2
        gaussian_error_per_pixel_i_j = ops.exp(
            -(squared_l2_error_matrices) / (2 * entropy_sigma**2)
        )
        # Vertically stack all columns corresponding with the same line
        # This way we can just sum across the height axis and get the entropy
        # for each pixel in a given line
        batch_size, n_particles, _, height, _ = gaussian_error_per_pixel_i_j.shape
        gaussian_error_per_pixel_stacked = ops.transpose(
            ops.reshape(
                ops.transpose(gaussian_error_per_pixel_i_j, (0, 1, 2, 4, 3)),
                [
                    batch_size,
                    n_particles,
                    n_particles,
                    n_possible_actions,
                    height * stack_n_cols,
                ],
            ),
            (0, 1, 2, 4, 3),
        )
        # [n_particles, n_particles, batch, height, width]
        return gaussian_error_per_pixel_stacked

    def compute_pixelwise_entropy(self, particles):
        """Compute the entropy for each line using a Gaussian Mixture Model.

        This function computes the entropy for each line using a Gaussian Mixture Model
        approximation of the posterior distribution.
        For more details see Section 4 here: https://arxiv.org/abs/2406.14388

        Args:
            particles (Tensor): Particles of shape (batch_size, n_particles, height, width)

        Returns:
            Tensor: batch of entropies per line, of shape (batch, n_possible_actions)
        """
        gaussian_error_per_pixel_stacked = (
            GreedyEntropy.compute_pairwise_pixel_gaussian_error(
                particles,
                self.stack_n_cols,
                self.n_possible_actions,
                self.entropy_sigma,
            )
        )
        # sum out first dimension of (n_particles x n_particles) error matrix
        # [n_particles, batch, height, width]
        pixelwise_entropy_sum_j = ops.sum(gaussian_error_per_pixel_stacked, axis=1)
        log_pixelwise_entropy_sum_j = ops.log(pixelwise_entropy_sum_j)
        # sum out second dimension of (n_particles x n_particles) error matrix
        # [batch, height, width]
        pixelwise_entropy = ops.sum(log_pixelwise_entropy_sum_j, axis=1)
        return pixelwise_entropy

    def select_line_and_reweight_entropy(self, entropy_per_line):
        """Select the line with maximum entropy and reweight the entropies.

        Selected the max entropy line and reweights the entropy values around it,
        approximating the decrease in entropy that would occur from observing that line.

        Args:
            entropy_per_line (Tensor): Entropy per line of shape
                (batch_size, n_possible_actions)

        Returns:
            Tuple: The selected line index and the updated entropies per line
        """

        # Find the line with maximum entropy
        max_entropy_line = ops.argmax(entropy_per_line)

        ## The rest of this function updates the entropy values around max_entropy_line
        ## by multiplying them with an upside-down Gaussian function centered at
        ## max_entropy_line, setting the entropy of the selected line to 0, and decreasing
        ## the entropies of neighbouring lines.

        # Pad the entropy per line to allow for re-weighting with fixed
        # size RBF, which is necessary for tracing.
        padded_entropy_per_line = ops.pad(
            entropy_per_line,
            (self.num_lines_to_update // 2, self.num_lines_to_update // 2),
        )
        # because the entropy per line has now been padded, the start index
        # of the set of lines to update is simply the index of the max_entropy_line
        start_index = max_entropy_line

        # Create the re-weighting vector
        reweighting = ops.ones_like(padded_entropy_per_line)
        reweighting = ops.slice_update(
            reweighting,
            (start_index,),
            ops.cast(self.upside_down_gaussian, dtype=reweighting.dtype),
        )

        # Apply re-weighting to entropy values
        updated_entropy_per_line_padded = padded_entropy_per_line * reweighting
        updated_entropy_per_line = ops.slice(
            updated_entropy_per_line_padded,
            (self.num_lines_to_update // 2,),
            (self.n_possible_actions,),
        )
        return max_entropy_line, updated_entropy_per_line

    def sample(self, particles):
        """Sample the action using the greedy entropy method.

        Args:
            particles (Tensor): Particles of shape (batch_size, n_particles, height, width)

        Returns:
           Tuple[Tensor, Tensor]:
                - Newly selected lines as k-hot vectors, shaped (batch_size, n_possible_actions)
                - Masks of shape (batch_size, img_height, img_width)
        """
        pixelwise_entropy = self.compute_pixelwise_entropy(particles)
        linewise_entropy = ops.sum(pixelwise_entropy, axis=1)

        # Greedily select best line, reweight entropies, and repeat
        all_selected_lines = []
        for _ in range(self.n_actions):
            max_entropy_line, linewise_entropy = ops.vectorized_map(
                self.select_line_and_reweight_entropy, linewise_entropy
            )
            all_selected_lines.append(max_entropy_line)

        selected_lines_k_hot = ops.any(
            ops.one_hot(
                all_selected_lines, self.n_possible_actions, dtype=masks._DEFAULT_DTYPE
            ),
            axis=0,
        )
        return selected_lines_k_hot, self.lines_to_im_size(selected_lines_k_hot)
