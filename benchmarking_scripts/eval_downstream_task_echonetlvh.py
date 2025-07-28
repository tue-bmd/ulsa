import os
import sys

if __name__ == "__main__":
    sys.path.append("/latent-ultrasound-diffusion")
    sys.path.append("/latent-ultrasound-diffusion/active_sampling")
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from zea import init_device

    init_device()

from pathlib import Path

from zea import Config

from benchmark_active_sampling_ultrasound import run_benchmark

if __name__ == "__main__":

    TARGET_DIR = Path("/mnt/z/Ultrasound-BMd/data/USBMD_datasets/echonetlvh/val")
    SAVE_DIR = Path(
        "/mnt/z/Ultrasound-BMd/data/oisin/ULSA_out_dst/echonetlvh_downstream_task/23_07_25_run1"
    )

    ulsa_agent_dst_config = Config.from_yaml(
        Path("/ulsa/configs/echonetlvh_3_frames_downstream_task.yaml")
    )
    ulsa_agent_tig_config = Config.from_yaml(
        Path("/ulsa/configs/echonetlvh_3_frames.yaml")
    )

    # run_benchmark(
    #     agent_config=ulsa_agent_dst_config,
    #     target_dir=TARGET_DIR,
    #     save_dir=SAVE_DIR,
    #     sweep_params={
    #         "action_selection.n_actions": [1, 3, 5, 7],
    #     },
    #     limit_n_samples=50,
    #     image_range=(0, 255),
    #     validate_dataset=False
    # )

    run_benchmark(
        agent_config=ulsa_agent_tig_config,
        target_dir=TARGET_DIR,
        save_dir=SAVE_DIR,
        sweep_params={
            "action_selection.n_actions": [1, 3, 5, 7],
        },
        limit_n_samples=50,
        image_range=(0, 255),
        validate_dataset=False
    )

    