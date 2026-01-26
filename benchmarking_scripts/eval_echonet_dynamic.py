"""
Linux-server:

```bash
./launch/start_container.sh python benchmarking_scripts/eval_echonet_dynamic.py
```

Snellius

(check comment in snellius_entry.sh to make sure the code is used):
```bash
sbatch --time=05:00:00 launch/snellius_entry.sh python benchmarking_scripts/eval_echonet_dynamic.py
```

Snellius (sharding):

(make sure that the number of shards is actually needed for the sweep)
(assume ≈2 minutes of startup time per shard)

e.g.

500 files, 15 sweep entries, 100 frames = 750,000 frames
assume 2 fps
375,000 s ≈ 6,250 min
assume 750 shards -> 9 min per shard


```bash
sbatch --time=00:15:00 --array=0-749 \
    --output=slurm/slurm-%A_%a.out launch/snellius_sharded.sh \
    python benchmarking_scripts/eval_echonet_dynamic.py \
    --save_dir "/path/to/sharding_sweep_$(date +"%Y-%m-%d_%H-%M-%S")" --num_shards 750 \
    --split val
```

> [!TIP]
> Test one or two shards before starting all (to see if the code works and the time is sufficient).

```bash
sbatch --time=00:14:00 --array=0,1 \
    --output=slurm/slurm-%A_%a.out launch/snellius_sharded.sh \
    python benchmarking_scripts/eval_echonet_dynamic.py ...
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
        default=[2, 4, 7, 14, 28],
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/mnt/z/usbmd/ulsa/eval_echonet_dynamic",
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
            "diffusion_inference.batch_size": [2],
            "downstream_task": ["echonet_segmentation"],  # just runs additionally
        },
        limit_n_samples=args.limit_n_samples,  # set to None to use all samples
        limit_n_frames=args.limit_n_frames,  # makes sure every patient is equally represented
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    print("Benchmark completed successfully: ", sweep_save_dir)
