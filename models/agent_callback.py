import os
from pathlib import Path

import keras
import numpy as np
import wandb

from ulsa.benchmark_active_sampling_ultrasound import benchmark
from ulsa.io_utils import postprocess_agent_results, side_by_side_gif
from zea import Config, Dataset
from zea.metrics import Metrics


class AgentCallback(keras.callbacks.Callback):
    """Run the ULSA agent as a callback during training."""

    def __init__(
        self,
        agent_config,
        dataset: Dataset,
        dynamic_range: tuple,
        n_files=5,
        n_frames=30,
        seed=None,
        save_dir=None,
        wandb_log=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(agent_config, (str, Path)):
            agent_config = Config.from_yaml(agent_config)
        self.agent_config = agent_config

        self.seed = seed
        # TODO: Setting the seed does not work with the agent code in tensorflow and we usually train using that.
        assert self.seed is None, "Setting seed not supported in this callback."

        self.n_files = n_files
        self.n_frames = n_frames
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir()

        self.dynamic_range = dynamic_range  # TODO: I don't want to fix this.
        self.dataset = dataset
        self.wandb_log = wandb_log

        self.metrics = Metrics(["psnr"], image_range=[0, 255])

    @property
    def n_actions(self) -> int:
        """The number of actions the agent can take."""
        return self.agent_config.action_selection.n_actions

    @property
    def action_selection_shape(self) -> tuple:
        """The shape of the action selection space."""
        return self.agent_config.action_selection.shape

    @property
    def subsampling_rate(self) -> str:
        """The subsampling rate of the agent."""
        return (
            f"{self.agent_config.action_selection.n_actions}"
            + f"/{self.agent_config.action_selection.n_possible_actions}"
        )

    def on_epoch_end(self, epoch, logs=None):
        (
            all_metrics_results,
            _,
            _,
            _,
            agent_results,  # last result of loop
            agent,
        ) = benchmark(
            self.agent_config,
            self.dataset,
            self.dynamic_range,
            list(range(self.n_files)),
            self.n_frames,
            self.seed,
            self.model,
            jit_mode="off",
            metrics=self.metrics,
            save_dir=None,  # we dont want to save all the data in the callback
        )

        avg_per_file = self.metrics.parse_metrics(
            all_metrics_results, reduce_mean=True
        )  # avg per file per metric
        avg_results = {k: np.mean(v) for k, v in avg_per_file.items()}  # avg per metric
        if self.wandb_log:
            wandb.log(avg_results)

        # Save the last sequence to gif
        if self.save_dir is not None:
            kwargs = {
                "drop_first_n_frames": agent.input_shape[-1],
                "io_config": self.agent_config.io_config,
                "scan_convert_order": 0,
                "image_range": agent.input_range,
                "scan_convert_resolution": 0.5,
            }
            agent_results = agent_results.squeeze(-1)
            target_imgs = postprocess_agent_results(agent_results.target_imgs, **kwargs)
            reconstructions = postprocess_agent_results(
                agent_results.reconstructions, **kwargs
            )
            measurements = postprocess_agent_results(
                agent_results.measurements, **kwargs
            )
            gif_path = self.save_dir / f"agent_epoch_{epoch}.gif"
            side_by_side_gif(
                gif_path,
                target_imgs,
                reconstructions,
                measurements,
                dpi=150,
                labels=[
                    "Target",
                    "Reconstruction",
                    f"Measurements ({self.subsampling_rate})",
                ],
            )
            if self.wandb_log:
                wandb.log({"agent": [wandb.Video(str(gif_path))]})
