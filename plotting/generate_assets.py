"""This file generates assets for the documentation and website."""

import sys

sys.path.append("/ulsa")

import keras

import zea

zea.init_device()
keras.mixed_precision.set_global_policy("mixed_float16")
import shutil
from pathlib import Path

from active_sampling_temporal import active_sampling_single_file, save_results


def generate_assets(
    file_path="/mnt/USBMD_datasets/echonet_legacy/val/0X10A5FC19152B50A5.hdf5",
    n_actions: int = 7,
):
    run_dir, run_id = save_results(
        *active_sampling_single_file(
            "configs/echonet_3_frames.yaml",
            file_path,
            override_config={
                "downstream_task": None,
                "io_config": {
                    "gif_fps": 30,
                    "frame_cutoff": 100,
                    "plot_frames_for_presentation_kwargs": {
                        "file_type": "webm",
                        "context": "styles/website.mplstyle",
                        "drop_first_n_frames": 3,
                        "fill_value": "black",
                        "no_measurement_color": "black",
                    },
                },
                "action_selection": {"n_actions": n_actions},
            },
        ),
        "output/assets_for_docs",
    )
    return run_dir, run_id


def glob_copy_files(run_dir, pattern, dest_dir, dest_stem):
    files = list(run_dir.glob(pattern))
    assert len(files) == 1, f"Expected one file matching {pattern}, found {len(files)}"
    file = files[0]
    ext = file.suffix
    shutil.copy(file, dest_dir / f"{dest_stem}{ext}")


if __name__ == "__main__":
    new_assets_dir = Path("output/assets_for_docs/gathered")
    new_assets_dir.mkdir(parents=True, exist_ok=True)

    # For slider
    for n_actions in [2, 4, 7, 14, 28, 56, 112]:
        run_dir, run_id = generate_assets(n_actions=n_actions)

        glob_copy_files(
            run_dir,
            "measurements_reconstruction_*",
            new_assets_dir,
            f"measurements_reconstruction_{n_actions}",
        )

    # For teaser
    run_dir, run_id = generate_assets(
        "/mnt/USBMD_datasets/echonet_legacy/test/0X329210815212AA6A.hdf5", n_actions=14
    )
    glob_copy_files(
        run_dir,
        "heatmap_reconstruction_*",
        new_assets_dir,
        "heatmap_reconstruction_example_14",
    )
