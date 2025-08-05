import argparse
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D benchmark evaluation.")
    parser.add_argument(
        "--target_dir",
        type=str,
        default=None,
        help="Path to the target directory. If not set, uses default from data_paths.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Path to the save directory. If not set, uses default from data_paths.",
    )
    args = parser.parse_args()

    sys.path.append("/ulsa")
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from zea import init_device, set_data_paths

    init_device()
    data_paths = set_data_paths("/ulsa/users.yaml")

from pathlib import Path

from benchmark_active_sampling_ultrasound import run_benchmark
from zea import Config

if __name__ == "__main__":
    if args.target_dir is not None:
        TARGET_DIR = Path(args.target_dir)
    if args.save_dir is not None:
        SAVE_DIR = args.save_dir
    else:
        SAVE_DIR = data_paths.output / "ULSA_benchmarks" / "3d"

    ulsa_agent_config = Config.from_yaml(Path("/ulsa/configs/elevation_3d.yaml"))

    sweep_save_dir, all_metrics_results = run_benchmark(
        agent_config=ulsa_agent_config,
        target_dir=TARGET_DIR,
        save_dir=SAVE_DIR,
        sweep_params={
            "action_selection.n_actions": [3, 6, 12],
            "action_selection.selection_strategy": ["greedy_entropy"],
            "action_selection.kwargs": [{"average_across_batch": True}],
            "diffusion_inference.batch_size": [4],
        },
        image_range=(0, 255),
        data_type="data/image_3D",
        validate_dataset=False,  # 3D data not in official USBMD format
    )

    # sweep_save_dir, all_metrics_results = run_benchmark(
    #     agent_config=ulsa_agent_config,
    #     target_dir=TARGET_DIR,
    #     save_dir=SAVE_DIR,
    #     sweep_params={
    #         "action_selection.n_actions": [3, 6, 12],
    #         "action_selection.selection_strategy": [
    #             "uniform_random",
    #             "equispaced",
    #         ],
    #         "diffusion_inference.batch_size": [4],
    #     },
    #     image_range=(0, 255),
    #     data_type="data/image_3D",
    #     validate_dataset=False,  # 3D data not in official USBMD format
    # )
