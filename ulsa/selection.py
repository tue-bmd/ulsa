"""

Prototypes for additional `zea` selection functions

"""

from keras import ops
import jax

from zea.agent.selection import GreedyEntropy
from zea.backend.autograd import AutoGrad
from zea.agent import masks
from zea.internal.registry import action_selection_registry

from ulsa.downstream_task import downstream_task_registry, DifferentiableDownstreamTask

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
        super().__init__(n_actions, n_possible_actions, img_width, img_height, mean, std_dev, num_lines_to_update)
         # NOTE: for echonet don't forget to wrap scan convert in here
         # TODO: batch_size=4 shouldn't be hard coded
        self.downstream_task: DifferentiableDownstreamTask = (downstream_task_registry[downstream_task_key])(batch_size=4)

    def compute_output_and_saliency_propagation_hutchinson(self, particles, key):
        num_hutchinson_samples = 5

        z_squared = ops.zeros(self.downstream_task.output_shape)
        for i in range(num_hutchinson_samples): # TODO: vmap
            subkey = jax.random.fold_in(key, i)
            v = jax.random.normal(subkey, self.downstream_task) # TODO: downstream task output shape?
            # vjp returns a function that computes J^T v
            _, vjp_fun = jax.vjp(self.downstream_task.call_differentiable, particles)
            jt_v = vjp_fun(v)[0]  # [0] to get the input tangent
            z_squared += jt_v**2

        posterior_variance = ops.expand_dims(ops.var(particles, axis=0), axis=0)

        return posterior_variance * z_squared # TODO: where does z_squared get rid of the particle dim?

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
        assert full_image_width % self.n_possible_actions == 0, "n_possible_actions must divide evenly into image width"
        stacked_linewise_salience = ops.reshape(full_linewise_salience, (self.n_possible_actions, full_image_width // self.n_possible_actions))
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
        particles = ops.expand_dims(particles[0], axis=-1) # remove batch, add channel
        pixelwise_contribution_to_var_dst = self.compute_output_and_saliency_propagation_summed(particles)
        linewise_contribution_to_var_dst = ops.sum(pixelwise_contribution_to_var_dst[...,0], axis=1)
        linewise_contribution_to_var_dst = self.sum_neighbouring_columns_into_n_possible_actions(linewise_contribution_to_var_dst)

        # Greedily select best line, reweight entropies, and repeat
        all_selected_lines = []
        for _ in range(self.n_actions):
            max_contribution_line, linewise_contribution_to_var_dst = ops.vectorized_map(
                self.select_line_and_reweight_entropy, linewise_contribution_to_var_dst
            )
            all_selected_lines.append(max_contribution_line)

        selected_lines_k_hot = ops.any(
            ops.one_hot(all_selected_lines, self.n_possible_actions, dtype=masks._DEFAULT_DTYPE),
            axis=0,
        )
        return selected_lines_k_hot, self.lines_to_im_size(selected_lines_k_hot), pixelwise_contribution_to_var_dst