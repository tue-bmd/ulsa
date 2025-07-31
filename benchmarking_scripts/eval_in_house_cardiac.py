import sys

import zea

sys.path.append("/ulsa")  # for relative imports

zea.init_device(allow_preallocate=True)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from active_sampling_temporal import active_sampling_single_file
from in_house_cardiac.cardiac_scan import cardiac_scan
from ulsa.io_utils import gaussian_sharpness

save_dir = Path("/mnt/z/usbmd/Wessel/eval_in_house_cardiac")
save_dir.mkdir(parents=True, exist_ok=True)

folder = Path("/mnt/USBMD_datasets/2024_USBMD_cardiac_S51/HDF5/")
files = list(folder.glob("*_A4CH_*.hdf5"))
n_frames = None  # all frames
frame_idx = 23  # example frame index to visualize

override_config = dict(io_config=dict(frame_cutoff=n_frames))

for file in files:
    zea.log.info(f"Processing {file.stem}...")
    # Run active sampling on focused waves
    results, _, _, _, _, agent, _, _ = active_sampling_single_file(
        "configs/cardiac_112_3_frames.yaml",
        target_sequence=str(file),
        override_config=override_config,
        image_range=None,  # auto-dynamic range
    )

    # Unpack results
    squeezed_results = results.squeeze(-1)
    targets = squeezed_results.target_imgs
    reconstructions = squeezed_results.reconstructions

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
    np.savez(
        save_dir / f"{file.stem}_results.npz",
        focused=focused,
        diverging=diverging,
        reconstructions=reconstructions,
        theta_range=focused_scan.theta_range,  # assumes theta range is the same for focused and diverging!
        reconstruction_range=agent.input_range,
        focused_dynamic_range=focused_scan.dynamic_range,
        diverging_dynamic_range=diverging_scan.dynamic_range,
    )

    #########################
    # Example visualization #
    #########################
    def scan_convert(image):
        sc, _ = zea.display.scan_convert_2d(
            image,
            rho_range=(0, image.shape[0]),
            theta_range=focused_scan.theta_range,
            resolution=0.1,
            fill_value=np.nan,
            order=0,
        )
        return sc

    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    axs[0].set_title("Focused")
    axs[0].imshow(
        scan_convert(focused[frame_idx]),
        cmap="gray",
        vmin=focused_scan.dynamic_range[0],
        vmax=focused_scan.dynamic_range[1],
    )
    axs[1].set_title("Reconstruction")
    axs[1].imshow(
        scan_convert(gaussian_sharpness(reconstructions[frame_idx], 0.04)),
        cmap="gray",
        vmin=agent.input_range[0],
        vmax=agent.input_range[1],
    )
    axs[2].set_title("Diverging")
    axs[2].imshow(
        scan_convert(diverging[frame_idx]),
        cmap="gray",
        vmin=diverging_scan.dynamic_range[0],
        vmax=diverging_scan.dynamic_range[1],
    )
    axs[3].set_title("Target")
    axs[3].imshow(
        scan_convert(targets[frame_idx]),
        cmap="gray",
        vmin=agent.input_range[0],
        vmax=agent.input_range[1],
    )
    axs[0].axis("off")
    axs[1].axis("off")
    axs[2].axis("off")
    axs[3].axis("off")
    plt.tight_layout()
    example_path = save_dir / f"{file.stem}_example_frame_{frame_idx}.png"
    plt.savefig(save_dir / f"{file.stem}_frame_{frame_idx}.png")

    zea.log.info(
        f"Processed {file.stem} with {len(reconstructions)} frames, "
        f"saved example visualization to {zea.log.yellow(example_path)}"
    )
