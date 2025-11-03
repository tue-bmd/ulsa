import os
import sys

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from zea import init_device

    init_device()

from pathlib import Path

from benchmark_active_sampling_ultrasound import run_benchmark
from zea import Config

if __name__ == "__main__":
    BASE_ULSA_CONFIG = Path("/ulsa/configs/echonet_3_frames_downstream_task.yaml")
    TARGET_DIR = Path("/mnt/z/Ultrasound-BMd/data/USBMD_datasets/echonet/val")
    SAVE_DIR = Path(
        "/mnt/z/Ultrasound-BMd/data/oisin/ULSA_out_dst/echonet_downstream_task/run1_22_07_25"
    )

    ulsa_agent_config = Config.from_yaml(BASE_ULSA_CONFIG)

    run_benchmark(
        agent_config=ulsa_agent_config,
        target_dir=TARGET_DIR,
        save_dir=SAVE_DIR,
        sweep_params={
            "action_selection.n_actions": [1, 3, 7, 14],
        },
        limit_n_samples=25,
    )

    run_benchmark(
        agent_config=ulsa_agent_config,
        target_dir=TARGET_DIR,
        save_dir=SAVE_DIR,
        sweep_params={
            "action_selection.n_actions": [1, 3, 7, 14],
            "action_selection.selection_strategy": ["greedy_entropy"],
            "action_selection.kwargs": [{}],
        },
        limit_n_samples=25,
    )
