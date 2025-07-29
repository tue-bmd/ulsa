import os

os.environ["KERAS_BACKEND"] = "numpy"
import sys

sys.path.append("/ulsa")

import numpy as np

import zea

DATA_ROOT = "/mnt/z/usbmd/Wessel/cardiac_annotations/"

from plotting.plot_utils import ViolinPlotter


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


def add_layer_to_dict(one_layer: dict, key):
    two_layer = {}
    for k, v in one_layer.items():
        two_layer[k] = {key: v}
    return two_layer


# Load data
in_house_cardiac = np.load(f"{DATA_ROOT}/in_house_cardiac2.npz", allow_pickle=True)
targets = in_house_cardiac["targets"]
diverging_images = in_house_cardiac["diverging_images"]
reconstructions = in_house_cardiac["reconstructions"]

# Load masks
black_masks = np.load(
    f"{DATA_ROOT}/zea_20240701_P1_A4CH_0001_target_interpolated_masks_black.npy"
)
white_masks = np.load(
    f"{DATA_ROOT}/zea_20240701_P1_A4CH_0001_target_interpolated_masks_white.npy"
)
valve_masks = np.load(
    f"{DATA_ROOT}/zea_20240701_P1_A4CH_0001_target_selected_interpolated_masks_valve.npy"
)
selected_frames = np.load(f"{DATA_ROOT}/20240701_P1_A4CH_0001_selected_frames.npy")[:-1]
black_masks = black_masks > 0
white_masks = white_masks > 0
valve_masks = valve_masks > 0

case = "valve"  # "normal", or "valve"
if case == "valve":
    black_masks = black_masks[selected_frames]
    white_masks = white_masks[selected_frames]
    targets = targets[selected_frames]
    diverging_images = diverging_images[selected_frames]
    reconstructions = reconstructions[selected_frames]
    masks1 = black_masks
    masks2 = valve_masks
else:
    mask1 = black_masks
    mask2 = white_masks

gcnr_all = {}

print("gCNR for targets:")
gcnr = gcnr_per_frame(targets, masks1, masks2)
print(np.mean(gcnr))
gcnr_all["targets"] = gcnr

print("gCNR for diverging images:")
gcnr = gcnr_per_frame(diverging_images, masks1, masks2)
print(np.mean(gcnr))
gcnr_all["diverging_images"] = gcnr

print("gCNR for reconstructions:")
gcnr = gcnr_per_frame(reconstructions, masks1, masks2)
print(np.mean(gcnr))
gcnr_all["reconstructions"] = gcnr


gcnr_targets = gcnr_all.pop("targets")
relative_gcnr = {}
for k, v in gcnr_all.items():
    relative_gcnr[k] = v - gcnr_targets


def sort_by_names(combined_results, names):
    """Sort combined results by strategy names."""
    return {k: combined_results[k] for k in names if k in combined_results}


group_names = {
    "reconstructions": "Active Perception",
    "targets": "Focused",
    "diverging_images": "Diverging",
}
plotter = ViolinPlotter(group_names, xlabel="Subjects")
for ext in [".png", ".pdf"]:
    plotter.plot(
        sort_by_names(add_layer_to_dict(relative_gcnr, "I"), group_names.keys()),
        f"output/gcnr_violin{ext}",
        x_label_values=["I"],
        metric_name="gCNR",
        context="styles/ieee-tmi.mplstyle",
    )


import matplotlib.pyplot as plt

with plt.style.context("styles/ieee-tmi.mplstyle"):
    fig = plt.figure()
    for k, gcnr in sort_by_names(relative_gcnr, group_names.keys()).items():
        plt.plot(
            selected_frames,
            gcnr,
            label=group_names[k],
            # linestyle="",
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
    plt.savefig("output/gcnr_per_frame.png")
    plt.savefig("output/gcnr_per_frame.pdf")

zea.log.info(f"Saved gCNR plot as {zea.log.yellow('output/gcnr_per_frame.png')}")
