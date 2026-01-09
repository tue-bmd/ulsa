import argparse
import os

os.environ["KERAS_BACKEND"] = "jax"

import sys

import zea

sys.path.append("/ulsa")
import keras
import numpy as np

from active_sampling_temporal import active_sampling_single_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark inference speed of diffusion models."
    )
    parser.add_argument(
        "--case",
        type=int,
        choices=[1, 2, 3, 4],
        required=True,
        help="Benchmark case to run.",
    )
    parser.add_argument(
        "--drop_n_frames",
        type=int,
        default=3,
        help="Number of timing frames to drop for warmup.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # defaults
    # keras.mixed_precision.set_global_policy("float32")
    override_config = {
        "diffusion_inference": {
            "initial_step": 475,
        },
        "io_config": {
            "frame_cutoff": 20,
        },
    }

    if args.case == 1:
        override_config["diffusion_inference"]["initial_step"] = 0
        jit_mode = "posterior_sample"
    elif args.case == 2:
        jit_mode = "posterior_sample"
    elif args.case == 3:
        jit_mode = "recover"
    elif args.case == 4:
        jit_mode = "recover"
        keras.mixed_precision.set_global_policy("mixed_float16")

    zea.init_device()

    timings = active_sampling_single_file(
        "configs/echonet_3_frames.yaml",
        override_config=override_config,
        jit_mode=jit_mode,
        return_timings=True,
        map_type="vmap",
    )[-1]

    timings = np.array(timings[args.drop_n_frames :])  # drop warmup
    total_time = np.sum(timings)

    fps = len(timings) / total_time
    frame_time_ms = 1000 / fps

    print(f"Case {args.case}")
    print(f"Frame freq.: {fps:.2f} Hz")
    print(f"Frame time: {frame_time_ms:.1f} ms")
