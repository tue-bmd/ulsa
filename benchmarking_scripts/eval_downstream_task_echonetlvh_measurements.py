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

from benchmark_active_sampling_ultrasound import run_benchmark
from zea import Config

if __name__ == "__main__":
    TARGET_DIR = Path("/mnt/z/Ultrasound-BMd/data/USBMD_datasets/echonetlvh/val")
    SAVE_DIR = Path(
        "/mnt/z/Ultrasound-BMd/data/oisin/ULSA_out_dst/echonetlvh_downstream_task/26_08_25_run3"
    )

    ulsa_agent_dst_config = Config.from_yaml(
        Path("/ulsa/configs/echonetlvh_3_frames_downstream_task.yaml")
    )

    run_benchmark(
        agent_config=ulsa_agent_dst_config,
        target_dir=TARGET_DIR,
        save_dir=SAVE_DIR,
        sweep_params={
            "action_selection.n_actions": [1, 3, 5],
            "action_selection.kwargs.measurement_type": ["LVID", "IVS", "LVPW"],
        },
        limit_n_samples=2,
        image_range=(0, 255),
        validate_dataset=False,
    )
