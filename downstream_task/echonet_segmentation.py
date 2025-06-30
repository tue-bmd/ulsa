import argparse
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "jax"

    from zea import init_device

    init_device()

import numpy as np
import yaml
from keras import ops
from zea import Dataset
from zea.config import Config
from zea.tensor_ops import batched_map
from zea.utils import translate

from ulsa.downstream_task import EchoNetSegmentation
from ulsa.io_utils import map_range


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run EchoNet segmentation on target videos and save results as metrics.npz in run dirs."
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="/mnt/z/Ultrasound-BMd/data/USBMD_datasets/echonet/val",
        help="Directory containing target .hdf5 files.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/mnt/z/Ultrasound-BMd/data/Wessel/output/lud/ULSA_benchmarks/fully_observed_echonet_seg",
        help="Directory to save run dirs.",
    )
    parser.add_argument(
        "--us_agent_config",
        type=str,
        default="/latent-ultrasound-diffusion/active_sampling/configs/ultrasound/agent_v3_echonet_3_frames.yaml",
        help="Path to US agent config yaml (for io_config, etc).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for segmentation model."
    )
    parser.add_argument(
        "--limit_n_samples",
        type=int,
        default=None,
        help="Num samples to take from the target_dir.",
    )
    parser.add_argument(
        "--limit_n_frames",
        type=int,
        default=100,
        help="Num frames per sample. If less than limit_n_frames, will just take the max available frames.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    target_dir = Path(args.target_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load and save the US agent config as config.yaml in the root save_dir
    agent_config = Config.from_yaml(args.us_agent_config)
    config_save_path = save_dir / "config.yaml"
    agent_config.save_to_yaml(config_save_path)

    # Load dataset (each sample is a video: (1, H, W, num_frames))
    dataset = Dataset(target_dir, key="data/image")

    # Initialize segmentation model
    seg_model = EchoNetSegmentation(batch_size=args.batch_size)

    denormalize = lambda x: map_range(x, (-1, 1), (0, 255)).astype(np.uint8)

    if args.limit_n_samples is None:
        args.limit_n_samples = len(dataset)

    for i in range(args.limit_n_samples):
        file = dataset[i]
        video = file.load_data(dataset.key, slice(args.limit_n_frames))
        video = translate(video, (-60, 0), (-1, 1))
        targets = video[..., None]
        reconstructions = targets.copy()

        frames = seg_model.scan_convert_batch(targets)
        masks = batched_map(seg_model, frames, args.batch_size)
        metadata = {
            "masks": ops.convert_to_numpy(masks),
            "x_scan_converted": ops.convert_to_numpy(ops.squeeze(frames, axis=-1)),
        }

        # Save in run dir
        run_dir = save_dir / f"run_{i:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save the same config.yaml in each run dir for replay compatibility
        agent_config.save_to_yaml(run_dir / "config.yaml")

        np.savez(
            run_dir / "metrics.npz",
            targets=denormalize(targets),
            reconstructions=denormalize(reconstructions),
            metadata=metadata,
        )

        with open(run_dir / "target_filepath.yaml", "w") as f:
            yaml.dump({"target_filepath": str(file.path)}, f)

        print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
