"""Script to calculate and plot gCNR (Generalized Contrast-to-Noise Ratio)
for cardiac imaging data."""

import os

os.environ["KERAS_BACKEND"] = "numpy"
import sys
from collections import OrderedDict
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import zea

sys.path.append("/ulsa")
from plotting.plot_utils import ViolinPlotter


def write_roman(num):
    roman = OrderedDict()
    roman[1000] = "M"
    roman[900] = "CM"
    roman[500] = "D"
    roman[400] = "CD"
    roman[100] = "C"
    roman[90] = "XC"
    roman[50] = "L"
    roman[40] = "XL"
    roman[10] = "X"
    roman[9] = "IX"
    roman[5] = "V"
    roman[4] = "IV"
    roman[1] = "I"

    def roman_num(num):
        for r in roman.keys():
            x, y = divmod(num, r)
            yield roman[r] * x
            num -= r * x
            if num <= 0:
                break

    return "".join([a for a in roman_num(num)])


def gcnr_per_frame(images, mask1, mask2):
    """
    Calculate gCNR for each frame in the images array.

    Parameters:
    - images: numpy array of shape (frames, h, w)
    - mask1: boolean mask for the first region of shape (frames, h, w)
    - mask2: boolean mask for the second region of shape (frames, h, w)

    Returns:
    - List of gCNR values for each frame
    """

    def single_gcnr(img, m1, m2):
        return zea.metrics.gcnr(img[m1], img[m2])

    vectorized_gcnr = np.vectorize(single_gcnr, signature="(h,w),(h,w),(h,w)->()")
    return vectorized_gcnr(images, mask1, mask2)


def gcnr_valve(images, blacks, valve, selected_frames):
    images = images[selected_frames]
    blacks = blacks[selected_frames]

    print("gCNR for targets:")
    gcnr = gcnr_per_frame(images, blacks, valve)
    print(np.mean(gcnr))
    return gcnr


def add_layer_to_dict(one_layer: dict, key):
    two_layer = {}
    for k, v in one_layer.items():
        two_layer[k] = {key: v}
    return two_layer


def sort_by_names(combined_results, names):
    """Sort combined results by strategy names."""
    return {k: combined_results[k] for k in names if k in combined_results}


# with plt.style.context("styles/ieee-tmi.mplstyle"):
def plot_gcnr_over_time(
    selected_frames: np.ndarray,
    relative_gcnr: dict,
    group_names: dict,
    save_path: str,
    smoothness: int = 5,
    alpha: float = 0.5,
):
    fig = plt.figure()
    markers = ["x", "o", "v", "s", "d", "+"]
    ls = ["-", "--", ":", "-.", [5, [10, 3]], [0, [3, 1, 1, 1]]]
    color = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]
    for i, (k, gcnr) in enumerate(
        sort_by_names(relative_gcnr, group_names.keys()).items()
    ):
        plt.plot(
            selected_frames,
            gcnr,
            linestyle="",
            alpha=alpha,
            color=color[i % len(color)],
            marker=markers[i % len(markers)],
        )
        # smooth gcnr line
        plt.plot(
            selected_frames,
            np.convolve(gcnr, np.ones(smoothness) / smoothness, mode="same"),
            linestyle=ls[i % len(ls)],
            color=color[i % len(color)],
            # no marker
            marker="",
        )
        # empty line just for legend
        plt.plot(
            [],
            [],
            label=group_names[k],
            linestyle=ls[i % len(ls)],
            color=color[i % len(color)],
            marker=markers[i % len(markers)],
        )

    plt.xlabel("Frame index [-]")
    plt.ylabel("Relative gCNR [-]")
    # plt.title("gCNR per Frame")
    fig.legend(
        loc="outside upper center",
        ncol=2,
        frameon=False,
    )
    plt.grid()
    plt.savefig(save_path)
    zea.log.info(f"Saved gCNR plot as {zea.log.yellow(save_path)}")


def main():
    ANNOTATIONS_ROOT = Path("/mnt/z/usbmd/Wessel/cardiac_annotations/")
    DATA_ROOT = Path("/mnt/z/usbmd/Wessel/eval_in_house_cardiac/")
    subjects = ["20240701_P1_A4CH_0001"]
    group_names = {
        "reconstructions": "Active Perception",
        "focused": "Focused",
        "diverging": "Diverging",
    }
    relative_to = "focused"

    gcnr_valve_all = {}
    gcnr_all = {}
    for i, subject in enumerate(subjects):
        # Load annotations and results
        subject_name = write_roman(i + 1)
        wf = ANNOTATIONS_ROOT / f"{subject}_white_annotations.npy"
        bf = ANNOTATIONS_ROOT / f"{subject}_black_annotations.npy"
        vf = ANNOTATIONS_ROOT / "selected_frames" / f"{subject}_valve_annotations.npy"
        sf = ANNOTATIONS_ROOT / "selected_frames" / f"{subject}_selected_frames.npy"
        rf = DATA_ROOT / f"{subject}_results.npz"
        white_masks = np.load(wf) > 0
        black_masks = np.load(bf) > 0
        valve_masks = np.load(vf) > 0
        selected_frames = np.load(sf)[:-1]  # Exclude last frame
        results = np.load(rf, allow_pickle=True)

        if not (white_masks.shape == black_masks.shape == valve_masks.shape):
            raise ValueError("Masks must have the same shape.")

        # Calculate gCNR for each reconstruction type
        gcnr_valve_results = {}
        gcnr_results = {}
        for type in group_names.keys():
            images = results[type]

            gcnr_valve_results[type] = gcnr_valve(
                images, black_masks, valve_masks, selected_frames
            )
            gcnr_results[type] = gcnr_per_frame(images, black_masks, white_masks)

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

        gcnr_all[subject_name] = gcnr_relative
        gcnr_valve_all[subject_name] = gcnr_valve_relative

    # Violin plot & over time plot
    violin = ViolinPlotter(group_names, xlabel="Subjects")
    for ext, (_gcnr, key) in product(
        [".png", ".pdf"], zip([gcnr_all, gcnr_valve_all], ["gcnr", "gcnr_valve"])
    ):
        violin.plot(
            sort_by_names(_gcnr, group_names.keys()),
            f"output/{key}_violin{ext}",
            x_label_values=["I"],
            metric_name="gCNR",
            context="styles/ieee-tmi.mplstyle",
        )
        plot_gcnr_over_time(
            selected_frames,
            _gcnr,
            group_names,
            f"output/{key}_over_time{ext}",
        )


if __name__ == "__main__":
    main()
