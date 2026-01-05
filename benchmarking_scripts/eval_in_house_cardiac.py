"""
Evaluate in-house (cardiac) data using different sampling strategies.
Also saves focused and diverging wave reconstructions.

To run on phantom:
    ./launch/start_container.sh \
        python benchmarking_scripts/eval_in_house_cardiac.py \
        --save_dir "/mnt/z/usbmd/Wessel/eval_phantom2/" \
        --folder "/mnt/z/usbmd/Wessel/Verasonics/2025-11-18_zea" \
        --pattern "*.hdf5" --frame_idx 19

To run on in-house cardiac data:
    ./launch/start_container.sh \
        python benchmarking_scripts/eval_in_house_cardiac.py
"""

import sys

import zea

sys.path.append("/ulsa")  # for relative imports

zea.init_device(allow_preallocate=True)

import argparse
from pathlib import Path

import numpy as np

from active_sampling_temporal import active_sampling_single_file
from in_house_cardiac.cardiac_scan import cardiac_scan
from in_house_cardiac.to_itk import npz_to_itk
from plotting.plot_in_house_cardiac import get_arrow, plot_from_npz


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate in-house cardiac data.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/mnt/z/usbmd/Wessel/eval_in_house_cardiac_v3/",
        # default="/mnt/z/usbmd/Wessel/eval_phantom",
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
        nargs="+",
        default=[
            "/mnt/z/usbmd/Wessel/Verasonics/2026_USBMD_A4CH_S51_V2/",
            # "/mnt/USBMD_datasets/2024_USBMD_cardiac_S51/HDF5/",
            # "/mnt/z/usbmd/Wessel/Verasonics/2025-11-18_zea",
        ],
        help="Folder(s) containing the HDF5 files. Can specify multiple folders.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_a4ch_line_dw_*.hdf5",
        # default="*.hdf5",
        help="Pattern to match HDF5 files in the folder.",
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=24,
        # default=19,
        help="Frame index to plot.",
    )
    parser.add_argument(
        "--low_pct",
        type=float,
        # default=18,
        default=44,
        help="Low percentile for dynamic range calculation.",
    )
    parser.add_argument(
        "--high_pct",
        type=float,
        # default=95,
        default=99.99,
        help="High percentile for dynamic range calculation.",
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
    selection_strategies=None,
    frame_idx=24,
    low_pct=18,
    high_pct=95,
):
    zea.log.info(f"Processing {file.stem}...")

    if selection_strategies is None:
        selection_strategies = ["greedy_entropy", "equispaced", "uniform_random"]

    save_dir = Path(save_dir) / file.stem
    save_dir.mkdir(parents=True, exist_ok=True)

    # TODO: make this neater
    with zea.File(file) as f:
        n_focused_tx = np.where(f.scan().focus_distances > 0)[0].size
        grid_width = n_focused_tx * 6

    # Run diverging waves (full dynamic range)
    zea.log.info("Running diverging waves...")
    diverging, diverging_scan = cardiac_scan(
        file,
        n_frames=n_frames,
        grid_width=grid_width,
        type="diverging",
        resize_to=(112, 112),
        low_pct=low_pct,
        high_pct=high_pct,
    )
    np.savez(
        save_dir / f"diverging.npz",
        reconstructions=diverging,
        theta_range=diverging_scan.theta_range,
        dynamic_range=diverging_scan.dynamic_range,
    )

    # Run focused waves (full dynamic range)
    zea.log.info("Running focused waves...")
    focused, focused_scan = cardiac_scan(
        file,
        n_frames=n_frames,
        grid_width=grid_width,
        type="focused",
        resize_to=(112, 112),
        low_pct=low_pct,
        high_pct=high_pct,
    )
    np.savez(
        save_dir / f"focused.npz",
        reconstructions=focused,
        theta_range=focused_scan.theta_range,
        dynamic_range=focused_scan.dynamic_range,
    )

    # For annotation purposes, also save as itk
    npz_to_itk(
        save_dir / f"focused.npz",
        save_dir / f"focused.nii.gz",
        dynamic_range=focused_scan.dynamic_range,
    )

    override_config = zea.Config(override_config)
    if "action_selection" not in override_config:
        override_config["action_selection"] = {}

    for selection_strategy in selection_strategies:
        print(f"Running active perception with {selection_strategy}...")
        _override_config = override_config.copy()

        _override_config["action_selection"]["selection_strategy"] = selection_strategy

        if selection_strategy == "equispaced":
            _override_config["action_selection"]["kwargs"] = {
                "assert_equal_spacing": False
            }

        # Run active sampling on focused waves
        results, _, _, _, _, agent, _, _ = active_sampling_single_file(
            "configs/cardiac_112_3_frames.yaml",
            target_sequence=str(file),
            override_config=_override_config,
            image_range=image_range,
            seed=seed,
            low_pct=low_pct,
            high_pct=high_pct,
        )

        # Unpack results
        squeezed_results = results.squeeze(-1)

        # Save results (all as floats and not scan converted)
        np.savez(
            save_dir / f"{selection_strategy}.npz",
            reconstructions=squeezed_results.reconstructions,
            measurements=squeezed_results.measurements,
            masks=squeezed_results.masks,
            targets=squeezed_results.target_imgs,
            belief_distributions=squeezed_results.belief_distributions,
            dynamic_range=agent.input_range,
            theta_range=diverging_scan.theta_range,
        )

    print(f"Saved results to {save_dir}")

    if visualize:
        print("Creating plots...")
        plot_from_npz(save_dir, save_dir, gif_fps=fps, frame_idx=frame_idx)

    return save_dir


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    folders = [Path(f) for f in args.folder]
    files = []
    for folder in folders:
        files += list(folder.glob(args.pattern, case_sensitive=False))
    n_frames = args.n_frames  # all frames if None

    override_config = dict(io_config=dict(frame_cutoff=n_frames))

    for file in sorted(files):
        eval_in_house_data(
            file,
            save_dir,
            n_frames,
            override_config,
            frame_idx=args.frame_idx,
            low_pct=args.low_pct,
            high_pct=args.high_pct,
        )


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
        "output/in_house_cardiac",
        gif=False,
        context="styles/ieee-tmi.mplstyle",
        diverging_dynamic_range=[-70, -30],
        focused_dynamic_range=[-68, -20],
        arrow=get_arrow(),
    )


def run_harmonic_example():
    acq = "20251222_s3_a4ch_line_dw_0000"
    path = eval_in_house_data(
        Path(f"/mnt/z/usbmd/Wessel/Verasonics/2026_USBMD_A4CH_S51_V2/{acq}.hdf5"),
        Path("/mnt/z/usbmd/Wessel/ulsa_paper_plots_v2"),
        n_frames=None,
        override_config=dict(io_config=dict(frame_cutoff=None)),
        visualize=False,
        image_range=[-60, -10],
        seed=0,
    )
    plot_from_npz(
        path,
        "output/in_house_cardiac",
        gif=False,
        context="styles/ieee-tmi.mplstyle",
        diverging_dynamic_range=[-60, -10],
        focused_dynamic_range=[-60, -10],
    )


if __name__ == "__main__":
    main()  # run all a4ch

    # run_single_example()
    # run_harmonic_example()
