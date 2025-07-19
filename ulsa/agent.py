from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Callable, Tuple, Optional

import jax
import keras
import numpy as np
from keras import ops
from rich.console import Console
from rich.table import Table

import zea
from ulsa.buffer import FrameBuffer, lifo_shift
from ulsa.pfield import lines_to_pfield
from ulsa.selection import DownstreamTaskSelection
from zea.agent.selection import (
    EquispacedLines,
    GreedyEntropy,
    LinesActionModel,
    MaskActionModel,
    UniformRandomLines,
)
from zea.backend import jit
from zea.config import Config
from zea.internal.operators import Operator, operator_registry
from zea.internal.registry import action_selection_registry
from zea.models.diffusion import DiffusionModel
from zea.tensor_ops import split_seed
from zea.utils import translate


@operator_registry(name="soft_inpainting")
class SoftInpaintingOperator(Operator):
    """Soft inpainting operator class.

    Soft inpainting uses a soft grayscale mask for a smooth transition between
    the inpainted and generated regions of the image.
    """

    def __init__(self, image_range, mask_range=None):
        self.image_range = tuple(image_range)
        assert len(self.image_range) == 2

        if mask_range is None:
            mask_range = (0.0, 1.0)
        self.mask_range = tuple(mask_range)
        assert len(self.mask_range) == 2
        assert self.mask_range[0] == 0.0, "mask_range[0] must be 0.0"

    def forward(self, data, mask):
        data1 = translate(data, self.image_range, self.mask_range)
        data2 = mask * data1
        data3 = translate(data2, self.mask_range, self.image_range)
        return data3

    def __str__(self):
        return "SoftInpaintingOperator"


DEBUGGING = False


# Static agent properties go here
@dataclass(frozen=True)
class Agent:
    initial_action_selection: Callable
    recover: Callable
    input_shape: Tuple
    input_range: Tuple
    n_particles: int
    selection_strategy: str
    pre_action: Callable
    post_action: Callable
    operator: Any
    pfield: Any

    def print_summary(self):
        print("\n")
        console = Console()
        table = Table(
            title="🔊 Agent Configuration ♻️",
            show_header=True,
            header_style="bold magenta",
        )

        # Add columns
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        # Add rows
        table.add_row("Input Shape", str(self.input_shape))
        table.add_row("Input Range", str(self.input_range))
        table.add_row("Number of Particles", str(self.n_particles))
        table.add_row("Selection Strategy", self.selection_strategy)

        console.print(table)
        print("\n")


# dynamic agent properties go here
@jax.tree_util.register_pytree_node_class
@dataclass
class AgentState:
    measurement_buffer: FrameBuffer
    mask: Any  # tensor of shape [height, width, n_frames]
    seed: Any
    selected_lines: Any
    posterior_samples: Any  # tensor of shape [n_particles, height, width, n_frames]: these are used as initial samples for SeqDiff
    belief_distribution: Any  # tensor of shape [n_particles, height, width, 1]: posterior samples at time t
    pipeline_state: Any
    target_pipeline_state: Any
    saliency_map: Any # heatmap used to make action selection decisions

    def tree_flatten(self):
        # All fields are dynamic
        children = (
            self.measurement_buffer,
            self.mask,
            self.seed,
            self.selected_lines,
            self.posterior_samples,
            self.belief_distribution,
            self.pipeline_state,
            self.target_pipeline_state,
            self.saliency_map,
        )
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def get_initial_action_selection_fn(
    action_selector: LinesActionModel, initial_selection_strategy="uniform_random"
):
    if isinstance(action_selector, GreedyEntropy):
        selector_class: MaskActionModel = action_selection_registry[
            initial_selection_strategy
        ]
        initial_selector = selector_class(
            n_actions=action_selector.n_actions,
            n_possible_actions=action_selector.n_possible_actions,
            img_width=action_selector.img_width,
            img_height=action_selector.img_height,
        )
        return initial_selector
    elif isinstance(action_selector, (EquispacedLines, UniformRandomLines)):
        return action_selector
    else:
        raise UserWarning("Invalid action selection strategy")


def action_selection_wrapper(action_selector: LinesActionModel):
    """
    Different selection strategy sample functions require different input parameters.
    This function defines a generic wrapper over those sample functions so that the
    agent code can be agnostic to the choice of sample_fn, giving a standardized signature:
        particles, current_lines, seed -> selected_lines, mask
    """

    # Makes sure signature is the same for all action selection strategies
    if isinstance(action_selector, DownstreamTaskSelection):
        def action_selection(particles, current_lines, seed):
            selected_lines, mask, salience = action_selector.sample(
                particles=particles
            )
            return (selected_lines, mask), salience
    elif isinstance(action_selector, GreedyEntropy):
        def action_selection(particles, current_lines, seed):
            return action_selector.sample(particles=particles), None
    elif isinstance(action_selector, EquispacedLines):

        def selection(particles, current_lines, seed):
            if current_lines is None:
                return action_selector.initial_sample_stateless(), None
            else:
                return action_selector.sample_stateless(current_lines=current_lines), None

        action_selection = selection
    elif isinstance(action_selector, UniformRandomLines):
        def action_selection(particles, current_lines, seed):
            return action_selector.sample(seed=seed), None
    else:
        raise UserWarning("Invalid action selection strategy")

    def wrapper(particles, current_lines, seed):
        """
        Ensures batch dims are removed and the mask has a channel dim.
        """
        # Add empty batch dim if not present
        if current_lines is not None:
            current_lines = ops.expand_dims(current_lines, axis=0)
        if particles is not None:
            particles = ops.expand_dims(particles, axis=0)

        # Run the action selection function
        (selected_lines, mask_t), saliency_map = action_selection(particles, current_lines, seed)

        # Ensure the output has the correct shape
        selected_lines = ops.squeeze(selected_lines, axis=0)  # remove batch dim
        mask_t = ops.squeeze(mask_t, axis=0)  # remove batch dim
        mask_t = mask_t[..., None]  # add channel dim
        return selected_lines, mask_t, saliency_map

    return wrapper


def identity(x):
    """
    Return the input without changing it. Useful default for callables/
    """
    return x


def action_selection_pre_post(
    action_selector, pre: callable = identity, post: callable = identity
):
    """
    This function is used to wrap the action selection function with a pre and post function
    and applies them to the input and output of the action selection function.

    Note that the selected_lines are not modified by the post function.
    """

    def wrapper(particles, current_lines, seed):
        """
        Args:
            particles: tensor of shape [n_particles, batch_size, height, width]
        """
        # Apply pre function
        if particles is not None:
            particles = pre(particles)

        # Call action selection function
        selected_lines, mask_t, saliency_map = action_selector(particles, current_lines, seed)

        # Apply post function
        mask_t = post(mask_t)

        return selected_lines, mask_t, saliency_map

    return wrapper


def action_selection_pfield(action_selector, **kwargs):
    """
    This function is used to wrap the action selection function to return a
    pfield instead of a mask.
    """

    def wrapper(particles, current_lines, seed):
        selected_lines, _ = action_selector(particles, current_lines, seed)
        pfield_mask = lines_to_pfield(selected_lines[None], **kwargs)
        return selected_lines, pfield_mask

    return wrapper


def reset_agent_state(agent: Agent, seed, batch_size=None):
    _, _, n_frames = agent.input_shape
    selected_lines, mask_t, saliency_map = agent.initial_action_selection(
        particles=None, current_lines=None, seed=seed
    )
    mask = lifo_shift(ops.zeros(agent.input_shape), mask_t)

    return AgentState(
        measurement_buffer=FrameBuffer(
            image_shape=agent.input_shape, batch_size=batch_size, buffer_size=n_frames
        ),
        mask=mask,
        seed=seed,
        selected_lines=selected_lines,
        posterior_samples=None,
        belief_distribution=None,
        pipeline_state={},
        target_pipeline_state={},
        saliency_map=saliency_map
    )


def get_operator_dict(agent_config):
    if (
        "pfield" in agent_config.action_selection
        and agent_config.diffusion_inference.get("guidance_method", "dps") == "dps"
    ):
        return {"name": "soft_inpainting"}
    elif agent_config.diffusion_inference.get("operator", "inpainting") == "blur_noise":
        psf = np.load("psf.npy")
        psf = ops.expand_dims(
            ops.repeat(
                ops.expand_dims(ops.convert_to_tensor(psf), axis=-1), 3, axis=-1
            ),
            axis=-1,
        )
        return {"name": "blur_noise", "params": {"psf": psf, "noise_std": 0.1}}
    else:
        return {"name": "inpainting", "params": {"min_val": 0}}


def setup_agent(
    agent_config: Config,
    seed,
    batch_size=None,
    pfield=None,
    jit_mode="recover",
    model=None,
) -> Tuple[Agent, AgentState]:
    """
    Uses the parsed YAML config details stored in agent_config to initialise
    the functions in AgentConfig
    """

    if model is None:
        # 1: Set up perception model (posterior sampler)
        model_path = agent_config.diffusion_inference.run_dir
        guidance_method = agent_config.diffusion_inference.get("guidance_method", "dps")

        # NOTE: we now assume our posterior sampler to be a diffusion model, but other than this
        # setup function the code should be agnostic to this.
        model = DiffusionModel.from_preset(
            model_path,
            guidance={
                "name": guidance_method,
                "params": {"disable_jit": True},
            },  # we jit later
            operator=get_operator_dict(agent_config),
        )

    n_particles = agent_config.diffusion_inference.batch_size

    def posterior_sample(measurements, mask, initial_samples, seed, batched=False):
        # duplicate measurements for each particle
        measurements = ops.repeat(measurements[None], n_particles, axis=0)

        # pack particles into additional batch dim
        if batched:
            n_part, n_batch, height, width, frame = measurements.shape
            measurements = ops.reshape(
                measurements, (n_part * n_batch, height, width, frame)
            )
            # assert ops.shape(mask) == (n_part, n_batch, height, width, frame)
            # mask = ops.reshape(mask, (n_part * n_batch, height, width, frame))
            if initial_samples is not None:
                assert ops.shape(initial_samples) == (
                    n_part,
                    n_batch,
                    height,
                    width,
                    frame,
                )
                initial_samples = ops.reshape(
                    initial_samples, (n_part * n_batch, height, width, frame)
                )

        # we vmap the posterior sampling to make the guidance weight
        # omega independent of batch size.
        def posterior_sample_individual(args):
            measurement, seed_i, initial_sample, initial_step, omega = args
            posterior_samples = model.posterior_sample(
                measurements=measurement,
                mask=mask[None],  # add dummy batch dim
                n_steps=agent_config.diffusion_inference.num_steps,
                initial_step=initial_step,
                initial_samples=initial_sample,
                seed=seed_i,
                track_progress_type=None,
                omega=omega,
            )
            return ops.squeeze(posterior_samples, axis=0)

        measurements = ops.expand_dims(measurements, axis=1)  # add dummy batch dim
        seeds = split_seed(seed, len(measurements))
        omega = agent_config.diffusion_inference.guidance_kwargs.omega
        if initial_samples is None:
            initial_omega = agent_config.diffusion_inference.guidance_kwargs.get(
                "initial_omega", omega
            )
            if DEBUGGING:
                psi = lambda meas, seed: posterior_sample_individual(
                    (meas, seed, None, 0, initial_omega)
                )
                posterior_samples = ops.stack(
                    [psi(m, s) for m, s in zip(measurements, seeds)]
                )
            else:
                posterior_samples = ops.vectorized_map(
                    lambda meas_seed: posterior_sample_individual(
                        (*meas_seed, None, 0, initial_omega)
                    ),
                    (measurements, seeds),
                )
        else:
            initial_samples = initial_samples[
                :, None, None
            ]  # add dummy batch and particle dim
            if DEBUGGING:
                psi = lambda meas, seed, inits: posterior_sample_individual(
                    (meas, seed, inits, 0, initial_omega)
                )
                posterior_samples = ops.stack(
                    [
                        psi(m, s, i)
                        for m, s, i in zip(measurements, seeds, initial_samples)
                    ]
                )
            else:
                posterior_samples = ops.vectorized_map(
                    lambda meas_seed_inits: posterior_sample_individual(
                        (
                            *meas_seed_inits,
                            agent_config.diffusion_inference.initial_step,
                            omega,
                        )
                    ),
                    (measurements, seeds, initial_samples),
                )

        # remove dummy batch dim
        posterior_samples = ops.squeeze(posterior_samples, axis=1)

        # unpack particles from batch dim
        if batched:
            n_batch, height, width, frame = posterior_samples.shape
            n_batch_unpacked = n_batch / n_particles
            assert n_batch_unpacked.is_integer()
            n_batch_unpacked = int(n_batch_unpacked)
            posterior_samples = ops.reshape(
                posterior_samples, (n_particles, n_batch_unpacked, height, width, frame)
            )

        return posterior_samples

    # 2: set up action model
    if agent_config.action_selection.get("shape"):
        img_height, img_width = agent_config.action_selection.shape
    else:
        img_height, img_width, _ = model.input_shape
        agent_config.action_selection.shape = (img_height, img_width)

    action_selection_class: MaskActionModel = action_selection_registry[
        agent_config.action_selection.selection_strategy
    ]
    action_selector = action_selection_class(
        n_actions=agent_config.action_selection.n_actions,
        n_possible_actions=agent_config.action_selection.n_possible_actions,
        img_height=img_height,
        img_width=img_width,
        **agent_config.action_selection.get("kwargs", {}),
    )
    initial_action_selection = action_selection_wrapper(
        get_initial_action_selection_fn(action_selector)
    )
    action_selection = action_selection_wrapper(action_selector)

    if agent_config.action_selection.get("pfield"):
        assert pfield is not None, "pfield must be provided"
        # TODO: is resizing the pfield fine?
        pfield = ops.image.resize(
            pfield,  # (n_tx, n_z, n_x)
            agent_config.action_selection.shape,
            interpolation="bilinear",
            antialias=True,
            data_format="channels_first",  # n_tx is the channel dim
        )

        initial_action_selection = action_selection_pfield(
            initial_action_selection,
            pfield=pfield,
            n_actions=agent_config.action_selection.n_actions,
        )
        action_selection = action_selection_pfield(
            action_selection,
            pfield=pfield,
            n_actions=agent_config.action_selection.n_actions,
        )
    else:
        pfield = None

    pre_action = keras.layers.CenterCrop(
        *agent_config.action_selection.shape, data_format="channels_first"
    )
    post_action = lambda data: zea.ops.Pad(
        model.input_shape[:-1],
        axis=(-3, -2),
        jit_compile=False,  # we jit later
        # fail_on_bigger_shape=False,
    )(data=data)["data"]

    initial_action_selection = action_selection_pre_post(
        initial_action_selection, pre_action, post_action
    )
    action_selection = action_selection_pre_post(
        action_selection, pre_action, post_action
    )

    if jit_mode == "posterior_sample":
        posterior_sample = jit(posterior_sample)

    recover_p = partial(
        recover,
        reconstruction_method=agent_config.diffusion_inference.reconstruction_method,
        posterior_sample=partial(
            posterior_sample, batched=agent_config.get("is_3d", False)
        ),
        action_selection=action_selection,
    )
    if jit_mode == "recover":
        recover_p = jit(recover_p)

    agent = Agent(
        initial_action_selection=initial_action_selection,
        recover=recover_p,
        input_shape=model.input_shape,
        input_range=model.input_range,
        n_particles=n_particles,
        selection_strategy=agent_config.action_selection.selection_strategy,
        pre_action=pre_action,
        post_action=post_action,
        operator=model.operator,
        pfield=pfield,
    )

    return agent, reset_agent_state(agent, seed, batch_size=batch_size)


def beliefs_to_recovered_frame(belief_distribution, reconstruction_method):
    if reconstruction_method == "choose_first":
        return belief_distribution[0]
    elif reconstruction_method == "mean":
        return ops.mean(belief_distribution, axis=0)
    else:
        raise UserWarning("Invalid reconstruction_method")


def recover(
    measurements,
    agent_state: AgentState,
    reconstruction_method: str,
    posterior_sample: Callable,
    action_selection: Callable,
):
    # 1. parse agent state
    measurement_buffer = agent_state.measurement_buffer
    mask = agent_state.mask
    base_seed = agent_state.seed
    selected_lines = agent_state.selected_lines
    posterior_samples = agent_state.posterior_samples

    measurement_buffer.shift(measurements)

    seed_1, seed_2, base_seed = split_seed(base_seed, 3)

    # 2. recover current beliefs from measurements
    new_posterior_samples = posterior_sample(
        measurement_buffer.buffer,
        mask,
        posterior_samples,
        seed=seed_1,
    )

    # 3. pass those into action selection
    belief_distribution = new_posterior_samples[..., -1, None]  # x_t ~ p(x_t | y_<t)
    new_selected_lines, new_mask_t, saliency_map = action_selection(
        belief_distribution[..., 0], selected_lines, seed_2
    )
    new_mask = lifo_shift(mask, new_mask_t)

    # 4. create a reconstruction from the beliefs
    recovered_frame = beliefs_to_recovered_frame(
        belief_distribution, reconstruction_method=reconstruction_method
    )

    # 5. update the state
    new_agent_state = replace(
        agent_state,
        measurement_buffer=measurement_buffer,
        mask=new_mask,
        selected_lines=new_selected_lines,
        posterior_samples=new_posterior_samples,
        belief_distribution=belief_distribution,
        seed=base_seed,
        saliency_map=saliency_map
    )

    return recovered_frame, new_agent_state
