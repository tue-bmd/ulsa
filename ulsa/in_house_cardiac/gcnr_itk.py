"""Script to calculate and plot gCNR (Generalized Contrast-to-Noise Ratio)
for harmonic cardiac imaging data (annotated using ITK-SNAP)."""

import os

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "jax"
import zea

if __name__ == "__main__":
    zea.init_device()

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from keras import ops

from ulsa.in_house_cardiac.gcnr import (
    METRIC_LABEL,
    filter_empty,
    plot_gcnr_over_time,
    sort_by_names,
    swap_layer,
)
from ulsa.metrics import gcnr_per_frame
from ulsa.plotting.masks import visualize_masks
from ulsa.plotting.plot_utils import ViolinPlotter, write_roman
from zea.internal.cache import cache_output

SAVE_DIR = Path("output/gcnr")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


@cache_output(verbose=True)
def load_results():
    DATA_ROOT = Path("/mnt/z/usbmd/ulsa/eval_in_house/cardiac_harmonic/")
    ANNOTATION_ROOT = Path(
        "/mnt/z/usbmd/ulsa/eval_in_house/cardiac_harmonic_annotations/"
    )

    subjects = [
        "20251222_s1_a4ch_line_dw_0000",
        "20251222_s2_a4ch_line_dw_0000",
        "20251222_s3_a4ch_line_dw_0000",
    ]

    group_names = {
        "greedy_entropy": "Cognitive",
        # "uniform_random": "Random",
        # "equispaced": "Equispaced",
        "focused": "Focused",
        "diverging": "Diverging",
    }
    relative_to = "focused"
    skip_first_n_frames = 4

    relative_gcnr_valve_all = {}
    relative_gcnr_all = {}

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

            masks = sitk.ReadImage(
                ANNOTATION_ROOT / subject / "focused_annotated.nii.gz"
            )
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

        # Store the unmodified gCNR results as well
        gcnr_all[subject] = gcnr_results
        gcnr_valve_all[subject] = gcnr_valve_results

        relative_gcnr_all[subject] = gcnr_relative
        relative_gcnr_valve_all[subject] = gcnr_valve_relative

    relative_gcnr_valve_all = filter_empty(relative_gcnr_valve_all)
    gcnr_valve_all = filter_empty(gcnr_valve_all)

    return (
        subjects,
        group_names,
        relative_gcnr_all,
        relative_gcnr_valve_all,
        gcnr_all,
        gcnr_valve_all,
    )


def main():
    subjects, group_names, relative_gcnr, relative_gcnr_valve, _, _ = load_results()

    # Convert subject keys to Roman numerals
    subjects_ids = {s: write_roman(i + 1) for i, s in enumerate(subjects)}

    # Violin plot & over time plot for all
    violin = ViolinPlotter(group_names, xlabel="Subjects")
    for ext, (_gcnr, key) in product(
        [".png", ".pdf"],
        zip([relative_gcnr, relative_gcnr_valve], ["gcnr", "gcnr_valve"]),
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
