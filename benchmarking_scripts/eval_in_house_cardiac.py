import sys

import zea

sys.path.append("/ulsa")  # for relative imports

zea.init_device(allow_preallocate=True)

import argparse
from pathlib import Path

import numpy as np

from active_sampling_temporal import active_sampling_single_file
from in_house_cardiac.cardiac_scan import cardiac_scan
from plotting.plot_in_house_cardiac import get_arrow, plot_from_npz


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate in-house cardiac data.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/mnt/z/usbmd/Wessel/eval_in_house_cardiac",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=None,
        help="Number of frames to process (None for all).",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="/mnt/USBMD_datasets/2024_USBMD_cardiac_S51/HDF5/",
        help="Folder containing the HDF5 files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_A4CH_*.hdf5",
        help="Pattern to match HDF5 files in the folder.",
    )
    return parser.parse_args()


def eval_in_house_data(
    file,
    save_dir,
    n_frames,
    override_config,
    visualize=True,
    fps=8,
    image_range=None,  # auto-dynamic range
    seed=42,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    zea.log.info(f"Processing {file.stem}...")
    # Run active sampling on focused waves
    results, _, _, _, _, agent, _, _ = active_sampling_single_file(
        "configs/cardiac_112_3_frames.yaml",
        target_sequence=str(file),
        override_config=override_config,
        image_range=image_range,
        seed=seed,
    )

    # Unpack results
    squeezed_results = results.squeeze(-1)

    # Run diverging waves (full dynamic range)
    zea.log.info("Running diverging waves...")
    diverging, diverging_scan = cardiac_scan(
        file,
        n_frames=n_frames,
        grid_width=90,
        resize_height=112,
        type="diverging",
    )

    # Run focused waves (full dynamic range)
    zea.log.info("Running focused waves...")
    focused, focused_scan = cardiac_scan(
        file,
        n_frames=n_frames,
        grid_width=90,
        resize_height=112,
        type="focused",
    )

    # Save results (all as floats and not scan converted)
    save_path = save_dir / f"{file.stem}_results.npz"
    np.savez(
        save_path,
        focused=focused,
        diverging=diverging,
        reconstructions=squeezed_results.reconstructions,
        measurements=squeezed_results.measurements,
        masks=squeezed_results.masks,
        targets=squeezed_results.target_imgs,
        theta_range=focused_scan.theta_range,  # assumes theta range is the same for focused and diverging!
        reconstruction_range=agent.input_range,
        focused_dynamic_range=focused_scan.dynamic_range,
        diverging_dynamic_range=diverging_scan.dynamic_range,
    )

    if not visualize:
        return save_path

    plot_from_npz(save_path, save_path, gif_fps=fps)

    return save_path


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    folder = Path(args.folder)
    files = list(folder.glob(args.pattern))
    n_frames = args.n_frames  # all frames if None

    override_config = dict(io_config=dict(frame_cutoff=n_frames))

    for file in files:
        eval_in_house_data(file, save_dir, n_frames, override_config)


def run_single_example():
    path = eval_in_house_data(
        Path(
            "/mnt/USBMD_datasets/2024_USBMD_cardiac_S51/HDF5/20240701_P1_A4CH_0001.hdf5"
        ),
        Path("/mnt/z/usbmd/Wessel/ulsa_paper_plots"),
        n_frames=None,
        override_config=dict(io_config=dict(frame_cutoff=None)),
        visualize=False,
        image_range=[-65, -20],
        seed=0,
    )
    plot_from_npz(
        path,
        "output/in_house_cardiac.png",
        gif=False,
        context="styles/ieee-tmi.mplstyle",
        diverging_dynamic_range=[-70, -30],
        focused_dynamic_range=[-68, -20],
        arrow=get_arrow(),
    )


if __name__ == "__main__":
    main()  # run all a4ch

    # run_single_example()
