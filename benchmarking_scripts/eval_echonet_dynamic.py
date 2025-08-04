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

e.g.

500 files, 9 sweep entries, 100 frames = 450000 frames
assume 5 fps
90000s = 1500min
assume 200 shards -> 7.5min per shard


```bash
sbatch --time=00:12:00 --array=0-199 \
    --output=slurm/slurm-%A_%a.out launch/snellius_sharded.sh \
    python benchmarking_scripts/eval_echonet_dynamic.py \
    --save_dir "/path/to/sharding_sweep_$(date +"%Y-%m-%d_%H-%M-%S")" --num_shards 200
```
"""

import argparse
import os
import sys

if __name__ == "__main__":
    sys.path.append("/ulsa")
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from zea import init_device, set_data_paths

    init_device()
    data_paths = set_data_paths("/ulsa/users.yaml")

from pathlib import Path

import keras

from benchmark_active_sampling_ultrasound import run_benchmark
from zea import Config


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
        default="/mnt/z/usbmd/Wessel/eval_echonet_dynamic",
        help="Directory to save the benchmark results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    keras.mixed_precision.set_global_policy("float32")  # echonet-dynamic uses float32

    TARGET_DIR = data_paths.data_root / "USBMD_datasets" / "echonet_legacy" / "val"
    SAVE_DIR = Path(args.save_dir)

    ulsa_agent_config = Config.from_yaml("/ulsa/configs/echonet_3_frames.yaml")

    # 41 hours
    sweep_save_dir, all_metrics_results = run_benchmark(
        agent_config=ulsa_agent_config,
        target_dir=TARGET_DIR,
        save_dir=SAVE_DIR,
        sweep_params={
            "action_selection.n_actions": args.n_actions,
            "action_selection.selection_strategy": [
                "equispaced",
                "greedy_variance",
                "greedy_entropy_univariate_gaussian",
                "uniform_random",
            ],
            "diffusion_inference.batch_size": [4],
            "downstream_task": ["echonet_segmentation"],  # just runs additionally
        },
        limit_n_samples=None,  # set to None to use all samples
        limit_n_frames=100,  # makes sure every patient is equally represented
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    # 21 hours
    sweep_save_dir, all_metrics_results = run_benchmark(
        agent_config=ulsa_agent_config,
        target_dir=TARGET_DIR,
        save_dir=SAVE_DIR,
        sweep_params={
            "action_selection.n_actions": args.n_actions,
            "action_selection.selection_strategy": ["covariance"],
            "action_selection.kwargs": [{"n_masks": int(1e5)}],
            "diffusion_inference.batch_size": [4],
            "downstream_task": ["echonet_segmentation"],  # just runs additionally
        },
        limit_n_samples=None,  # set to None to use all samples
        limit_n_frames=100,  # makes sure every patient is equally represented
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    print("Benchmark completed successfully: ", sweep_save_dir)
