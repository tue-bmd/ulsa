"""
Training DDIM model
Author(s): Tristan Stevens, Oisín Nolan, Wessel van Nierop
Date: 01/02/2024
"""

import argparse
import gc
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from zea import init_device

    init_device()

import json

import keras
import tensorflow as tf
import wandb
from agent_callback import AgentCallback
from keras import ops
from keras.callbacks import Callback
from wandb.integration.keras import WandbMetricsLogger

from zea import Config, Dataset, set_data_paths
from zea.backend.tensorflow.dataloader import make_dataloader
from zea.func import translate
from zea.io_lib import load_image, save_to_gif
from zea.models.base import BaseModel
from zea.models.diffusion import DiffusionModel
from zea.utils import get_date_string, log
from zea.visualize import plot_image_grid


# TODO: make this a registry & add combined loss
def get_loss(loss_name, *args, **kwargs):
    """Get a loss function by name."""

    assert isinstance(loss_name, str), (
        "loss_name must be a string or a list of strings / dicts"
    )

    if loss_name == "mse":
        return keras.losses.MeanSquaredError(*args, **kwargs)
    elif loss_name == "mae":
        return keras.losses.MeanAbsoluteError(*args, **kwargs)
    elif loss_name == "binary_crossentropy":
        return keras.losses.BinaryCrossentropy(*args, **kwargs)
    else:
        raise ValueError(f"Invalid loss name: {loss_name}")


# TODO: put this in zea wandb script
def setup_wandb(config, project, entity, job_type=None, wandb_dir="./wandb"):
    """Setup wandb logging
    Args:
        config (Config): The configuration object.
        project (str): The wandb project name.
        entity (str): The wandb entity name.
        job_type (str, optional): The job type. Defaults to None.
        wandb_dir (str, optional): The wandb directory. Defaults to "./wandb".

    Returns:
        Config: The configuration object with wandb setup.
    """

    if sys.gettrace():
        log.debug("Debugging, disabling wandb.")
        config.wandb = False
        return config

    wandb.login()

    run = wandb.init(
        project=project,
        entity=entity,
        config=config,
        allow_val_change=True,
        dir=wandb_dir,
        job_type=job_type,
    )

    log.info(f"wandb: {run.job_type} run {run.name}\n")

    config.wandb_log_dir = run.dir
    config.wandb = True

    return config


def get_postprocess_fn(training_config):
    def postprocess(img):
        img = ops.convert_to_numpy(img)
        img = translate(img, training_config.dataloader.normalization_range)
        return ops.cast(ops.clip(img, 0, 255), "uint8")

    return postprocess


# TODO: put this on zea base model
def load_model_from_checkpoint(json_path, checkpoint_path, skip_mistmatch):
    with open(json_path, "r", encoding="utf-8") as file:
        json_model = file.read()
    model: BaseModel = keras.models.model_from_json(json_model)
    model = model.load_weights(
        checkpoint_path,
        skip_mismatch=skip_mistmatch,
    )
    return model


# TODO: put this on zea base model
def save_model_json(model: BaseModel, json_path):
    """Save model as JSON file."""
    with open(json_path, "w", encoding="utf-8") as file:
        file.write(model.to_json(indent=4))
    log.success(f"Succesfully saved model architecture to {log.yellow(json_path)}")


# TODO: zea models / training utils?
class StatefulProgbar(keras.utils.Progbar):
    """Progbar that treats all metrics as stateful (prevents averaging)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override stateful_metrics to always return True
        self.stateful_metrics = type("", (), {"__contains__": lambda self, _: True})()


# TODO: zea models / training utils?
class StatefulProgbarLogger(keras.callbacks.ProgbarLogger):
    """
    ProgbarLogger wrapper that avoids metric averaging and allows customizing the unit name.
    """

    def __init__(self, unit_name="step", **kwargs):
        super().__init__()
        self.unit_name = unit_name
        self.progbar_kwargs = kwargs

    def on_train_begin(self, logs=None):
        # Ensure validation metrics are shown even in fit()
        self._called_in_fit = False

    def _maybe_init_progbar(self):
        if self.progbar is None:
            self.progbar = StatefulProgbar(
                target=self.target, verbose=self.verbose, **self.progbar_kwargs
            )


# TODO: zea models / training utils?
class ClearMemory(keras.callbacks.Callback):
    """Keras callback to clear memory after each epoch."""

    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        keras.backend.clear_session()


class DDIMSamplingCallback(Callback):
    """Sample and plot random generated images for visual evaluation of generation quality"""

    def __init__(
        self,
        diffusion_model,
        training_config,
        image_shape,
        diffusion_steps,
        batch_size,
        save_dir,
        postprocess_func,
        n_frames=1,
        start_with_eval=True,
        plot_aspect=None,
        wandb_log=False,
        seed=None,
        **kwargs,
    ):
        super().__init__()
        self.diffusion_model: DiffusionModel = diffusion_model
        self.training_config = training_config
        self.image_shape = image_shape
        self.diffusion_steps = diffusion_steps
        self.batch_size = batch_size
        self.save_dir = Path(save_dir)
        self.n_frames = n_frames
        self.postprocess_func = postprocess_func
        self.start_with_eval = start_with_eval
        self.plot_aspect = plot_aspect
        self.wandb_log = wandb_log
        self.kwargs = kwargs

        self.save_dir.mkdir(exist_ok=True)

        self.seed = seed

    def on_epoch_end(self, epoch, logs=None):
        samples = self.diffusion_model.sample(
            n_samples=self.batch_size,
            n_steps=self.diffusion_steps,
            seed=self.seed,
            **self.kwargs,
        )
        plot_batch(
            samples,
            f"generated_epoch_{epoch}",
            self.save_dir,
            self.training_config,
            wandb_tag="DM_generated_samples",
        )

    def on_train_begin(self, logs=None):
        if not self.start_with_eval:
            return
        return self.on_epoch_end(0, logs)


def plot_batch(batch, title, save_dir, training_config, wandb_tag):
    aspect = training_config.data.get("plot_aspect_ratio", "auto")
    if ops.shape(batch)[-1] == 1:
        fig, _ = plot_image_grid(
            batch,
            suptitle=title,
            aspect=aspect,
        )
        batch_plot_path = save_dir / f"{title}_batch.png"
        fig.savefig(batch_plot_path, bbox_inches="tight")
        if training_config.wandb:
            wandb.log({wandb_tag: [wandb.Image(str(batch_plot_path))]})
    else:
        n_frames = ops.shape(batch)[-1]
        for i in range(n_frames):
            fig, _ = plot_image_grid(
                batch[..., i],
                suptitle=f"{title} - Frame {i}",
                aspect=aspect,
            )
            fig.savefig(save_dir / f"{title}_batch_frame_{i}.png", bbox_inches="tight")
        animation_path = save_dir / f"{title}_batch_animation.gif"
        save_to_gif(
            [
                load_image(save_dir / f"{title}_batch_frame_{i}.png")
                for i in range(n_frames)
            ],
            animation_path,
            fps=10,
        )
        if training_config.wandb:
            wandb.log({wandb_tag: [wandb.Video(str(animation_path))]})


def print_train_summary(config):
    """Print training summary from config."""
    print("=" * 57)
    print("Training Summary:")
    print("=" * 57)
    print(f"| {'Parameter':<20} | {'Value':<30} |")
    print("|" + "-" * 55 + "|")
    print(f"| {'Epochs':<20} | {config.optimization.num_epochs:<30} |")
    print(f"| {'Learning rate':<20} | {config.optimization.learning_rate:<30} |")
    print(
        f"| {'Image shape':<20} | {', '.join(str(dim) for dim in config.dataloader.image_size):<30} |"
    )
    print(
        f"| {'Normalized range':<20} | "
        f"{', '.join(str(val) for val in config.dataloader.normalization_range):<30} |"
    )
    print(f"| {'#Frames':<20} | {config.dataloader.n_frames:<30} |")
    print("=" * 57)


def train_ddim(
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    config: Config,
    run_dir: str,
    postprocess_func: callable,
    train: bool = True,
    seed_gen: keras.random.SeedGenerator = None,
) -> DiffusionModel:
    """
    Trains the DDIM (Diffusion Model) using the provided datasets and configuration.

    Args:
        train_dataset (tf.data.Dataset): The training dataset.
        val_dataset (tf.data.Dataset): The validation dataset.
        config (Config): The configuration object.
        run_dir (str): The directory to save the training results.
        postprocess_func (callable): The postprocessing function to apply to the generated images
            such that they can be visualized.
        train (bool, optional): Whether to perform training or not. Defaults to True.
        seed_gen (keras.random.SeedGenerator, optional): The seed generator to use for random number generation.
    Returns:
        DiffusionModel: The trained DDIM model.
    """
    input_shape = train_dataset.element_spec.shape[1:].as_list()
    run_eagerly = config.run_eagerly if not sys.gettrace() else True

    # For debugging run, make sure the entire script is tested:
    num_epochs = config.optimization.num_epochs
    steps_per_epoch = config.optimization.get("steps_per_epoch")
    validation_steps = config.optimization.get("validation_steps")

    # create and compile the model
    if config.model.get("checkpoint") is None:
        model = DiffusionModel(
            input_shape,
            input_range=config.dataloader.normalization_range,
            min_signal_rate=config.sampling.min_signal_rate,
            max_signal_rate=config.sampling.max_signal_rate,
            network_kwargs=config.model.get("network_kwargs", {}),
            ema_val=config.optimization.get("ema_val", 0.999),
            guidance="dps",
            operator="inpainting",
        )
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=config.optimization.learning_rate,
                weight_decay=config.optimization.weight_decay,
                **config.optimization.get("kwargs", {}),
            ),
            loss=get_loss(config.optimization.loss),
            run_eagerly=run_eagerly,
        )
        model.sample(n_steps=1)  # will build the model
    else:
        model = load_model_from_checkpoint(
            run_dir / "config.json",
            config.model.checkpoint,
            config.optimization.get("skip_mismatch", False),
        )

    checkpoint_dir = Path(run_dir) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # save model architecture to run_dir/checkpoints/model.json
    # also includes the compile_config for optimizer and loss details
    save_model_json(model, checkpoint_dir / "config.json")

    # save the best model
    checkpoint_path = str(
        checkpoint_dir / (str(model.get_config()["name"]) + "_{epoch}.weights.h5")
    )
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=config.model.get("save_weights_only", True),
        monitor="i_loss",
        mode="min",
        save_best_only=False,
    )

    validation_sample_callback = DDIMSamplingCallback(
        model,
        config,
        input_shape,
        config.evaluation.diffusion_steps,
        config.evaluation.batch_size,
        save_dir=run_dir / "samples",
        n_frames=config.dataloader.n_frames,
        postprocess_func=postprocess_func,
        start_with_eval=config.evaluation.get("start_with_eval", False),
        plot_aspect=config.data.get("plot_aspect_ratio", "auto"),
        wandb_log=config.wandb,
        seed=seed_gen,
    )

    if "scheduler" in config.optimization:
        if not "factor" in config.optimization.scheduler:
            config.optimization.scheduler["factor"] = 0.1
        if not "patience" in config.optimization.scheduler:
            config.optimization.scheduler["patience"] = 10
        scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor="i_loss", verbose=1, mode="min", **config.optimization.scheduler
        )

    # save config to yaml file without reordering in run_dir
    config.save_to_yaml(Path(run_dir) / "config.yaml")

    print_train_summary(config)

    callbacks = [
        validation_sample_callback,
        checkpoint_callback,
        ClearMemory(),
        StatefulProgbarLogger(),
    ]

    if args.agent_callback:
        agent_callback = AgentCallback(
            args.agent_config,
            dataset=Dataset.from_config(
                config.data.val_folder.replace("{data_root}/", ""),
                config.dataloader.key,
                user=config.user,
            ),
            dynamic_range=config.dataloader.image_range,
            save_dir=run_dir / "agent",
            wandb_log=config.wandb,
        )
        callbacks += [agent_callback]

    if config.wandb:
        callbacks += [WandbMetricsLogger()]
    if "scheduler" in config.optimization:
        callbacks += [scheduler]

    # run training and plot generated images periodically
    if train:
        start_training_str = (
            f"Starting training for {num_epochs} epochs on {get_date_string()}..."
        )
        print("-" * len(start_training_str))
        print(start_training_str)
        model.fit(
            train_dataset,
            epochs=num_epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
        )

    # Check if trained model can be loaded (nice for debug runs)
    try:
        log.info("Trying to load model from checkpoint...")
        _ = DiffusionModel.from_checkpoint(
            checkpoint_path.format(epoch=num_epochs),
            skip_mismatch=config.optimization.get("skip_mismatch", False),
        )
    except Exception as e:
        log.warning(
            f"Could not load model from checkpoint: {checkpoint_path.format(epoch=num_epochs)}. "
            f"Error: {e}. Please check the checkpoint file."
        )
        pass
    return model


def parse_args():
    """Parse arguments for training DDIM."""
    parser = argparse.ArgumentParser(description="DDIM training")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/ddim/ddim_train_echonet_112_3_frames.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "-r",
        "--run_dir",
        type=str,
        default=None,
        help="Base path of the running directory.",
    )
    parser.add_argument(
        "-w",
        "--wandb",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable or disable wandb logging. 1 for enable, 0 for disable.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="latent-ultrasound-diffusion",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="aiteam-tue",
    )
    parser.add_argument(
        "--override_config",
        type=json.loads,
        default=None,
    )
    parser.add_argument(
        "--agent_callback",
        action="store_true",
        help="Whether or not to run the agent callback at the end of each epoch",
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        default=None,
        help="The agent_config to user for agent_callback. Must be specified if agent_callback is true.",
    )
    return parser.parse_args()


def validate_args(args):
    assert not (args.agent_callback and args.agent_config is None), (
        "An agent_config must be specified in order to user agent_callback."
    )


if __name__ == "__main__":
    # Load training config, args and data paths
    args = parse_args()
    validate_args(args)
    data_paths = set_data_paths("users.yaml", local=False)
    training_config = Config.from_yaml(args.config)
    training_config.user = {"data_root": data_paths["data_root"]}

    keras.utils.set_random_seed(training_config.seed)
    seed_gen = keras.random.SeedGenerator(training_config.seed)

    # Create dataloaders
    train_dataset = make_dataloader(
        file_paths=training_config.data.train_folder.format(
            data_root=data_paths["data_root"]
        ),
        **training_config.dataloader,
    )
    val_dataset = make_dataloader(
        file_paths=training_config.data.val_folder.format(
            data_root=data_paths["data_root"]
        ),
        **training_config.dataloader,
    )

    # Set up run dir and wandb
    date = get_date_string("%Y_%m_%d_%H%M%S_%f")
    debug_str = "_debug" if sys.gettrace() else ""

    if args.run_dir is None:
        args.run_dir = data_paths["output"] / "diffusion"

    run_dir = Path(args.run_dir) / (date + "_" + Path(args.config).stem + debug_str)
    run_dir.mkdir(exist_ok=True, parents=True)
    training_config.run_dir = str(run_dir)

    if args.wandb:
        training_config = setup_wandb(
            training_config,
            project=args.wandb_project,
            entity=args.wandb_entity,
            wandb_dir="./wandb",
            job_type="train_diffusion",
        )
    else:
        training_config.wandb = False

    # Plot training and validation data to sanity check
    log.info("Getting a batch for visualization...")
    train_batch = next(iter(train_dataset))
    val_batch = next(iter(val_dataset))

    postprocess_func = get_postprocess_fn(training_config)
    for batch, title in zip([val_batch, train_batch], ["validation", "train"]):
        plot_batch(
            batch, title, run_dir, training_config, wandb_tag=f"DM_{title}_batch"
        )

    model = train_ddim(
        train_dataset,
        val_dataset,
        training_config,
        run_dir,
        postprocess_func,
        train=True,
        seed_gen=seed_gen,
    )
    print(f"Training complete. Check results and cpkt in {run_dir}")
    del train_dataset
    del val_dataset
