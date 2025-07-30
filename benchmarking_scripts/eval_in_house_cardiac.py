import sys

import zea

sys.path.append("/ulsa")  # for relative imports

zea.init_device(allow_preallocate=False)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from active_sampling_temporal import active_sampling_single_file
from in_house_cardiac.diverging import diverging_waves

save_dir = Path("/mnt/z/usbmd/Wessel/eval_in_house_cardiac")
save_dir.mkdir(parents=True, exist_ok=True)

folder = Path("/mnt/USBMD_datasets/2024_USBMD_cardiac_S51/HDF5/")
files = list(folder.glob("*_A4CH_*.hdf5"))
n_frames = None  # all frames
frame_idx = 23  # example frame index to visualize

override_config = dict(io_config=dict(frame_cutoff=n_frames))

# TODO: do not clip to this range
# TODO: histogram matching for diffusion model range
dynamic_range = [-65, -20]

for file in files:
    # Run active sampling on focused waves
    results, _, _, _, _, agent, _, _ = active_sampling_single_file(
        "configs/cardiac_112_3_frames.yaml",
        target_sequence=str(file),
        override_config=override_config,
        image_range=dynamic_range,
    )

    # Unpack results
    squeezed_results = results.squeeze(-1)
    targets = squeezed_results.target_imgs
    reconstructions = squeezed_results.reconstructions

    # Translate from diffusion model range to original dynamic range
    targets = zea.utils.translate(targets, agent.input_range, dynamic_range)
    reconstructions = zea.utils.translate(
        reconstructions, agent.input_range, dynamic_range
    )

    # Run diverging waves
    images, scan = diverging_waves(
        file,
        n_frames=n_frames,
        dynamic_range=dynamic_range,
        grid_width=90,
        resize_height=112,
    )

    # Save results (all as floats and not scan converted)
    np.savez(
        save_dir / f"{file.stem}_results.npz",
        focused=targets,
        reconstructions=reconstructions,
        diverging=images,
        theta_range=scan.theta_range,  # assumes theta range is the same for focused and diverging!
        dynamic_range=dynamic_range,
    )

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(
        targets[frame_idx], cmap="gray", vmin=dynamic_range[0], vmax=dynamic_range[1]
    )
    axs[0].set_title("Target")
    axs[1].imshow(
        reconstructions[frame_idx],
        cmap="gray",
        vmin=dynamic_range[0],
        vmax=dynamic_range[1],
    )
    axs[1].set_title("Reconstruction")
    axs[2].imshow(
        images[frame_idx], cmap="gray", vmin=dynamic_range[0], vmax=dynamic_range[1]
    )
    axs[2].set_title("Diverging Waves")
    axs[0].axis("off")
    axs[1].axis("off")
    axs[2].axis("off")
    plt.tight_layout()
    plt.savefig(save_dir / f"{file.stem}_frame_{frame_idx}.png")
