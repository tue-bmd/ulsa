import os

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "jax"
import zea

if __name__ == "__main__":
    zea.init_device()
import sys
from itertools import product
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from keras import ops
from matplotlib.animation import FuncAnimation

from zea.internal.cache import cache_output
from zea.tools.selection_tool import remove_masks_from_axs
from zea.visualize import plot_shape_from_mask

sys.path.append("/ulsa")
from in_house_cardiac.gcnr import (
    METRIC_LABEL,
    filter_empty,
    plot_gcnr_over_time,
    sort_by_names,
    swap_layer,
)
from plotting.plot_utils import ViolinPlotter, write_roman
from ulsa.metrics import gcnr_per_frame

SAVE_DIR = Path("output/gcnr")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def update_imshow_with_masks(
    frame_no: int,
    axs: matplotlib.axes.Axes,
    imshow_obj: matplotlib.image.AxesImage,
    images: np.ndarray,
    masks: np.ndarray,
):
    colors = ["red", "blue", "green", "yellow", "cyan", "magenta"]
    imshow_obj.set_array(images[frame_no])

    remove_masks_from_axs(axs)

    for _masks, color in zip(masks, colors):
        plot_shape_from_mask(
            axs,
            _masks[frame_no],
            alpha=0.5,
            facecolor=color,
            edgecolor=color,
            linewidth=2.0,
        )


def visualize_masks(images, valve, myocardium, ventricle, filepath, fps=10):
    def update(frame_no):
        update_imshow_with_masks(
            frame_no,
            axs,
            imshow_obj,
            images,
            [
                ventricle,
                myocardium,
                valve,
            ],
        )

    fig, axs = plt.subplots()
    imshow_obj = axs.imshow(images[0], cmap="gray")
    interval = 1000 / fps  # milliseconds
    ani = FuncAnimation(
        fig,
        update,
        frames=len(images),
        interval=interval,
    )
    ani.save(filepath, writer="pillow")


@cache_output(verbose=True)
def load_results():
    DATA_ROOT = Path("/mnt/z/usbmd/Wessel/ulsa/eval_in_house_cardiac_v3/")

    subjects = [
        "20251222_s1_a4ch_line_dw_0000",
        "20251222_s2_a4ch_line_dw_0000",
        "20251222_s3_a4ch_line_dw_0000",
    ]

    group_names = {
        "greedy_entropy": "Active Perception",
        # "uniform_random": "Random",
        # "equispaced": "Equispaced",
        "focused": "Focused",
        "diverging": "Diverging",
    }
    relative_to = "focused"
    skip_first_n_frames = 4

    gcnr_valve_all = {}
    gcnr_all = {}
    for i, subject in enumerate(subjects):
        # Calculate gCNR for each reconstruction type
        gcnr_valve_results = {}
        gcnr_results = {}
        for type in group_names.keys():
            data = np.load(DATA_ROOT / subject / f"{type}.npz", allow_pickle=True)
            images = data["reconstructions"][skip_first_n_frames:]
            images, _ = zea.display.scan_convert_2d(
                images,
                (0, images.shape[1]),
                theta_range=data["theta_range"],
                resolution=0.3,  # TODO: check if resolution impacts gCNR
                order=0,
                fill_value=images.min(),
            )

            masks = sitk.ReadImage(DATA_ROOT / subject / "focused_annotated.nii.gz")
            masks = sitk.GetArrayFromImage(masks)
            masks = masks[skip_first_n_frames:]

            # Check mask labels
            other = masks > 3
            assert np.all(other == False), "Unexpected label in mask"

            if images.shape != masks.shape:
                print(f"Shape mismatch for subject {subject}, type {type}")
                print(f"Images shape: {images.shape}, Masks shape: {masks.shape}")
                aspect_ratio_images = images.shape[2] / images.shape[1]
                aspect_ratio_masks = masks.shape[2] / masks.shape[1]
                if np.abs(aspect_ratio_images - aspect_ratio_masks) < 0.01:
                    print("Aspect ratios are similar, resizing masks...")
                    masks = ops.squeeze(
                        ops.image.resize(
                            masks[..., None], images.shape[1:], interpolation="nearest"
                        ),
                        axis=-1,
                    )

            # background = masks == 0
            ventricle = masks == 1
            myocardium = masks == 2
            valve = masks == 3

            gcnr_valve_results[type] = gcnr_per_frame(images, ventricle, valve)
            gcnr_results[type] = gcnr_per_frame(images, ventricle, myocardium)

            if type == relative_to:
                visualize_masks(
                    np.clip(images, -60, -10),
                    valve,
                    myocardium,
                    ventricle,
                    f"debug_masks_{subject}.gif",
                    fps=10,
                )

        # Store results relative to the focused reconstruction
        gcnr_relative = {}
        gcnr_valve_relative = {}
        for k, v in gcnr_results.items():
            if k == relative_to:
                continue
            gcnr_relative[k] = v - gcnr_results[relative_to]
        for k, v in gcnr_valve_results.items():
            if k == relative_to:
                continue
            gcnr_valve_relative[k] = v - gcnr_valve_results[relative_to]

        gcnr_all[subject] = gcnr_relative
        gcnr_valve_all[subject] = gcnr_valve_relative

    gcnr_valve_all = filter_empty(gcnr_valve_all)

    return subjects, group_names, gcnr_all, gcnr_valve_all


def main():
    subjects, group_names, gcnr_all, gcnr_valve_all = load_results()

    # Convert subject keys to Roman numerals
    subjects_ids = {s: write_roman(i + 1) for i, s in enumerate(subjects)}

    # Violin plot & over time plot for all
    violin = ViolinPlotter(group_names, xlabel="Subjects")
    for ext, (_gcnr, key) in product(
        [".png", ".pdf"], zip([gcnr_all, gcnr_valve_all], ["gcnr", "gcnr_valve"])
    ):
        _gncr_roman = {subjects_ids[k]: v for k, v in _gcnr.items()}
        violin.plot(
            sort_by_names(swap_layer(_gncr_roman), group_names.keys()),
            SAVE_DIR / f"{key}_violin{ext}",
            x_label_values=_gncr_roman.keys(),
            metric_name=METRIC_LABEL,
            context="styles/ieee-tmi.mplstyle",
        )
        with plt.style.context("styles/ieee-tmi.mplstyle"):
            for subject in subjects:
                if subject not in _gcnr:
                    continue
                plot_gcnr_over_time(
                    None,
                    _gcnr[subject],
                    group_names,
                    SAVE_DIR / f"{subject}_{key}_over_time{ext}",
                )


if __name__ == "__main__":
    main()
