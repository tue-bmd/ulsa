"""

python benchmarking_scripts/sweep_num_particles.py \
    --save_dir /mnt/z/Ultrasound-BMd/data/oisin/ULSA_hyperparam_sweeps/n_particles_choose_first \
    --n_actions 7 14 28 \
    --limit_n_samples 20 \
    --limit_n_frames 100 \
    --selection_strategy greedy_entropy \
    --n_particles 2 3 4 5 6

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
        default=[2, 4, 7, 14, 28],
    )
    parser.add_argument(
        "--n_particles",
        type=int,
        nargs="+",
        help="List of n_particles values to sweep over, e.g. --n_actions 4 7 14",
        default=[2, 3, 4, 5, 6],
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/mnt/z/Ultrasound-BMd/data/oisin/ULSA_hyperparam_sweeps/n_particles_choose_first",
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
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split to use for the benchmark, e.g. 'train', 'val', 'test'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    keras.mixed_precision.set_global_policy("float32")  # echonet-dynamic uses float32

    TARGET_DIR = (
        data_paths.data_root
        / "USBMD_datasets"
        / "_LEGACY"
        / "echonet_legacy"
        / args.split
    )

    ulsa_agent_config = AgentConfig.from_yaml("/ulsa/configs/echonet_3_frames.yaml")

    sweep_save_dir, all_metrics_results = run_benchmark(
        agent_config=ulsa_agent_config,
        target_dir=TARGET_DIR,
        save_dir=Path(args.save_dir),
        sweep_params={
            "action_selection.n_actions": args.n_actions,
            "action_selection.selection_strategy": args.selection_strategy,
            "diffusion_inference.batch_size": args.n_particles,
            # "diffusion_inference.reconstruction_method": ["mean"],
        },
        limit_n_samples=args.limit_n_samples,  # set to None to use all samples
        limit_n_frames=args.limit_n_frames,  # makes sure every patient is equally represented
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    print("Benchmark completed successfully: ", sweep_save_dir)
