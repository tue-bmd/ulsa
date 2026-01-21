"""
This script evaluates the image reconstruction quality of in-house cardiac ultrasound images
using the same benchmarking framework as use as for echonet. This script does not compare to
diverging waves, use `benchmarking_scripts/eval_in_house_cardiac.py` for that.

```bash
./launch/start_container.sh python benchmarking_scripts/eval_in_house_cardiac_image_quality.py
```
"""

import argparse
import os

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "jax"

    from zea import init_device, set_data_paths

    init_device()
    data_paths = set_data_paths("/ulsa/users.yaml")

from pathlib import Path

import keras

from ulsa.benchmark_active_sampling_ultrasound import AgentConfig, run_benchmark


def parse_args():
    parser = argparse.ArgumentParser(description="Run ULSA benchmark for echonet.")
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--n_actions",
        type=int,
        nargs="+",
        help="List of n_actions values to sweep over, e.g. --n_actions 4 7 14",
        default=[7, 11, 22],
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./output/eval_in_house/image_quality/",
        help="Directory to save the benchmark results.",
    )
    parser.add_argument(
        "--limit_n_frames",
        type=int,
        default=100,
        help="Number of frames to use per patient for the benchmark.",
    )
    parser.add_argument(
        "--limit_n_samples",
        type=int,
        default=None,
        help="Number of samples to use for the benchmark. If None, all samples are used.",
    )
    parser.add_argument(
        "--selection_strategy",
        type=str,
        nargs="+",
        help="List of selection strategies to sweep over, e.g. --selection_strategy equispaced greedy_entropy",
        default=[
            "equispaced",
            "greedy_entropy",
            "uniform_random",
        ],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    keras.mixed_precision.set_global_policy("float32")  # echonet-dynamic uses float32

    TARGET_DIRS = [
        (data_paths.data_root / "datasets/2026_ULSA_A4CH_S51/fundamental/"),
        (data_paths.data_root / "datasets/2026_ULSA_A4CH_S51/harmonic/"),
    ]
    ulsa_agent_configs = [
        AgentConfig.from_yaml("/ulsa/configs/cardiac_112_3_frames.yaml"),
        AgentConfig.from_yaml("/ulsa/configs/cardiac_112_3_frames_harmonic.yaml"),
    ]

    for target_dir, ulsa_agent_config in zip(TARGET_DIRS, ulsa_agent_configs):
        sweep_save_dir, all_metrics_results = run_benchmark(
            agent_config=ulsa_agent_config,
            target_dir=target_dir,
            save_dir=Path(args.save_dir),
            sweep_params={
                "action_selection.n_actions": args.n_actions,
                "action_selection.selection_strategy": args.selection_strategy,
                "diffusion_inference.batch_size": [2],
                # "downstream_task": ["echonet_segmentation"],  # just runs additionally
            },
            limit_n_samples=args.limit_n_samples,  # set to None to use all samples
            limit_n_frames=args.limit_n_frames,  # makes sure every patient is equally represented
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            data_type="data/raw_data",
            reinitialize_every_file=True,
            jit_options="ops",
            metrics_batch_size=50,
        )
        print("Benchmark completed successfully: ", sweep_save_dir)
