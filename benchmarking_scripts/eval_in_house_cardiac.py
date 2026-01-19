"""
Evaluate in-house (cardiac) data using different sampling strategies.
Also saves focused and diverging wave reconstructions.
"""

import zea

if __name__ == "__main__":
    zea.init_device(allow_preallocate=True)

import argparse
from pathlib import Path

import numpy as np

from ulsa.in_house_cardiac.to_itk import npz_to_itk
from ulsa.active_sampling_temporal import active_sampling_single_file
from ulsa.cardiac_scan import cardiac_scan


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate in-house cardiac data.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/mnt/z/usbmd/Wessel/ulsa/eval_in_house/cardiac_harmonic/",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=None,
        help="Number of frames to process (None for all).",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        default=[
            "/mnt/datasets/2026_USBMD_A4CH_S51_V2/20251222_s1_a4ch_line_dw_0000.hdf5",
            "/mnt/datasets/2026_USBMD_A4CH_S51_V2/20251222_s2_a4ch_line_dw_0000.hdf5",
            "/mnt/datasets/2026_USBMD_A4CH_S51_V2/20251222_s3_a4ch_line_dw_0000.hdf5",
        ],
        help="Can be a list of folders and/or files containing in-house cardiac data HDF5 files.",
    )
    parser.add_argument(
        "--agent_config_path",
        type=str,
        default="./configs/cardiac_112_frames_harmonic.yaml",
        help="Path to agent configuration file.",
    )
    parser.add_argument(
        "--low_pct",
        type=float,
        default=44,
        help="Low percentile for dynamic range calculation.",
    )
    parser.add_argument(
        "--high_pct",
        type=float,
        default=99.99,
        help="High percentile for dynamic range calculation.",
    )
    return parser.parse_args()


def eval_in_house_data(
    file,
    save_dir,
    n_frames,
    agent_config_path,
    override_config,
    image_range=None,  # auto-dynamic range
    seed=42,
    selection_strategies=None,
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

        # e.g. (112,90) for 90 tx, (112,112) for 56 tx.
        resize_to = (112, (112 // n_focused_tx) * n_focused_tx)

    # Run focused waves (stores full dynamic range)
    zea.log.info("Running focused waves...")
    focused, focused_scan = cardiac_scan(
        file,
        n_frames=n_frames,
        grid_width=grid_width,
        type="focused",
        resize_to=resize_to,
        low_pct=low_pct,
        high_pct=high_pct,
    )
    np.savez(
        save_dir / f"focused.npz",
        reconstructions=focused,
        theta_range=focused_scan.theta_range,
        dynamic_range=focused_scan.dynamic_range,
    )

    # Run diverging waves (stores full dynamic range)
    zea.log.info("Running diverging waves...")
    diverging, diverging_scan = cardiac_scan(
        file,
        n_frames=n_frames,
        grid_width=grid_width,
        type="diverging",
        resize_to=resize_to,
        low_pct=low_pct,
        high_pct=high_pct,
    )
    np.savez(
        save_dir / f"diverging.npz",
        reconstructions=diverging,
        theta_range=diverging_scan.theta_range,
        dynamic_range=diverging_scan.dynamic_range,
    )

    # For annotation purposes, also save as itk
    npz_to_itk(save_dir / f"focused.npz", save_dir / f"focused.nii.gz")

    override_config = zea.Config(override_config)
    if "action_selection" not in override_config:
        override_config["action_selection"] = {}

    for selection_strategy in selection_strategies:
        print(f"Running active perception with {selection_strategy}...")
        _override_config = override_config.copy()

        _override_config["action_selection"]["selection_strategy"] = selection_strategy

        # Run active sampling on focused waves
        results, _, _, _, _, agent, agent_config, _ = active_sampling_single_file(
            agent_config_path,
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
            n_possible_actions=agent_config.action_selection.n_possible_actions,
            n_actions=agent_config.action_selection.n_actions,
        )

    print(f"Saved results to {save_dir}")

    return save_dir


def find_hdf5_files(files_and_folders: list):
    for file in files_and_folders:
        path = Path(file)
        if path.is_file() and path.suffix == ".hdf5":
            yield path
        elif path.is_dir():
            folder = path
            for f in folder.rglob("*.hdf5"):
                yield f


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    files = list(find_hdf5_files(args.files))
    assert len(files) > 0, "No HDF5 files found."

    override_config = dict(io_config=dict(frame_cutoff=args.n_frames))
    for file in sorted(files):
        eval_in_house_data(
            file,
            save_dir,
            args.n_frames,
            args.agent_config_path,
            override_config,
            low_pct=args.low_pct,
            high_pct=args.high_pct,
        )


if __name__ == "__main__":
    main()
