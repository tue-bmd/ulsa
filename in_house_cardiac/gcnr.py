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

METRIC_LABEL = "Relative gCNR [-]"


def filter_dict_of_arrays(d: dict, condition):
    return {k: condition(v) for k, v in d.items()}


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

    gcnr = gcnr_per_frame(images, blacks, valve)
    return gcnr


def swap_layer(d: dict):
    merged = {}
    labels = d.keys()
    ex_key = next(iter(d))
    keys_one_deep = d[ex_key].keys()  # subjects
    for key in keys_one_deep:
        merged[key] = {label: d[key] for label, d in zip(labels, d.values())}
    return merged


def filter_empty(d: dict):
    new_dict = {}
    for k, v in d.items():
        if v != {}:
            new_dict[k] = v
    return new_dict


def sort_by_names(combined_results, names):
    """Sort combined results by strategy names."""
    return {k: combined_results[k] for k in names if k in combined_results}


# with plt.style.context("styles/ieee-tmi.mplstyle"):
def plot_gcnr_over_time(
    selected_frames: np.ndarray,
    relative_gcnr: dict,
    group_names: dict,
    save_path: str = None,
    smoothness: int = 5,
    alpha: float = 0.5,
    markersize: int = 4,
    fig=None,
    zorder: dict = None,
):
    fig_was_given = fig is not None
    if not fig_was_given:
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
        # Just the markers
        plt.plot(
            selected_frames if selected_frames is not None else np.arange(len(gcnr)),
            gcnr,
            linestyle="",
            alpha=alpha,
            color=color[i % len(color)],
            marker=markers[i % len(markers)],
            markersize=markersize,
        )
        # Smooth gcnr line
        plt.plot(
            selected_frames if selected_frames is not None else np.arange(len(gcnr)),
            np.convolve(gcnr, np.ones(smoothness) / smoothness, mode="same"),
            linestyle=ls[i % len(ls)],
            color=color[i % len(color)],
            # no marker
            marker="",
            zorder=zorder[k] if zorder and k in zorder else None,
        )
        # Empty line just for legend
        plt.plot(
            [],
            [],
            label=group_names[k],
            linestyle=ls[i % len(ls)],
            color=color[i % len(color)],
            marker=markers[i % len(markers)],
        )

    if not fig_was_given:
        plt.xlabel("Frame index [-]")
        plt.ylabel(METRIC_LABEL)
        # plt.title("gCNR per Frame")
        fig.legend(
            loc="outside upper center",
            ncol=2,
            frameon=False,
        )
        plt.grid()
    if save_path is not None:
        plt.savefig(save_path)
        zea.log.info(f"Saved gCNR plot as {zea.log.yellow(save_path)}")
    return fig


def main():
    ANNOTATIONS_ROOT = Path("/mnt/z/usbmd/Wessel/cardiac_annotations_2/")
    DATA_ROOT = Path("/mnt/z/usbmd/Wessel/eval_in_house_cardiac/")
    SAVE_DIR = Path("output/gcnr")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    subjects = [
        "20240701_P1_A4CH_0001",
        "20241021_P9_A4CH_0000",
        "20240710_P7_A4CH_0000",
    ]

    group_names = {
        "reconstructions": "Active Perception",
        "focused": "Focused",
        "diverging": "Diverging",
    }
    relative_to = "focused"

    gcnr_valve_all = {}
    gcnr_all = {}
    selected_frames_all = {}
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
        results = np.load(rf, allow_pickle=True)

        if vf.exists():
            valve_masks = np.load(vf) > 0
            if sf.exists():
                selected_frames = np.load(sf)[:-1]  # Exclude last frame
            else:
                selected_frames = np.arange(valve_masks.shape[0])
            selected_frames_all[subject_name] = selected_frames
            assert len(selected_frames) == valve_masks.shape[0], (
                "Number of selected frames must match the number of valve masks."
            )

        assert (
            white_masks.shape[1:] == black_masks.shape[1:] == valve_masks.shape[1:]
        ), "White, black, and valve masks must have the same shape."

        # Calculate gCNR for each reconstruction type
        gcnr_valve_results = {}
        gcnr_results = {}
        for type in group_names.keys():
            images = results[type]
            images, _ = zea.display.scan_convert_2d(
                images,
                (0, images.shape[1]),
                theta_range=results["theta_range"],
                resolution=0.1,
                order=0,
            )

            if vf.exists():
                gcnr_valve_results[type] = gcnr_valve(
                    images, black_masks, valve_masks, selected_frames
                )
            gcnr_results[type] = gcnr_per_frame(images, black_masks, white_masks)

            # DEBUG
            # from matplotlib.animation import FuncAnimation
            # from zea.tools.selection_tool import update_imshow_with_mask
            # for mask_type in ["valve", "black", "white"]:
            #     if mask_type == "valve" and not vf.exists():
            #         continue
            #     elif mask_type == "valve":
            #         interpolated_masks = valve_masks
            #         _images = images[selected_frames]
            #     elif mask_type == "black":
            #         interpolated_masks = black_masks
            #         _images = images
            #     elif mask_type == "white":
            #         interpolated_masks = white_masks
            #         _images = images
            #     fig, axs = plt.subplots()
            #     imshow_obj = axs.imshow(_images[0], cmap="gray")
            #     ani = FuncAnimation(
            #         fig,
            #         update_imshow_with_mask,
            #         frames=len(_images),
            #         fargs=(axs, imshow_obj, _images, interpolated_masks, "lasso"),
            #         interval=1000 / 10,  # 10 FPS
            #     )
            #     ani.save(f"{subjects[i]}_{type}_{mask_type}.gif", writer="pillow")
            #     fig.close()

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

    gcnr_valve_all = filter_empty(gcnr_valve_all)

    # Violin plot & over time plot for all
    violin = ViolinPlotter(group_names, xlabel="Subjects")
    for ext, (_gcnr, key) in product(
        [".png", ".pdf"], zip([gcnr_all, gcnr_valve_all], ["gcnr", "gcnr_valve"])
    ):
        violin.plot(
            sort_by_names(swap_layer(_gcnr), group_names.keys()),
            SAVE_DIR / f"{key}_violin{ext}",
            x_label_values=_gcnr.keys(),
            metric_name=METRIC_LABEL,
            context="styles/ieee-tmi.mplstyle",
        )
        with plt.style.context("styles/ieee-tmi.mplstyle"):
            for i, subject in enumerate(subjects):
                subject_name = write_roman(i + 1)
                if subject_name not in _gcnr:
                    continue
                plot_gcnr_over_time(
                    selected_frames_all[subject_name] if key == "gcnr_valve" else None,
                    _gcnr[subject_name],
                    group_names,
                    SAVE_DIR / f"{subject}_{key}_over_time{ext}",
                )

    # Plots for the paper
    plt.close("all")
    save_dir = SAVE_DIR / "paper_plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    zorder = {
        "reconstructions": 20,
        "diverging": 10,
    }
    title_kwargs = {
        "y": 1.0,
        "pad": -10,
        "fontsize": 8,
        "alpha": 0.7,
        "loc": "left",
        "x": 0.03,
    }
    with plt.style.context("styles/ieee-tmi.mplstyle"):
        fig, axs = plt.subplots(2, 1, sharex=True)
        plt.sca(axs[0])
        _sel_frames = selected_frames_all["I"][selected_frames_all["I"] < 100]
        _gcnr = filter_dict_of_arrays(
            gcnr_valve_all["I"], lambda x: x[selected_frames_all["I"] < 100]
        )
        fig = plot_gcnr_over_time(
            _sel_frames,
            _gcnr,
            group_names,
            fig=fig,
            alpha=0.2,
            markersize=3,
            zorder=zorder,
        )
        plt.grid()
        plt.title("Subject I", **title_kwargs)
        plt.sca(axs[1])
        _sel_frames = selected_frames_all["II"][selected_frames_all["II"] < 100]
        _gcnr = filter_dict_of_arrays(
            gcnr_valve_all["II"], lambda x: x[selected_frames_all["II"] < 100]
        )
        fig = plot_gcnr_over_time(
            _sel_frames,
            _gcnr,
            group_names,
            fig=fig,
            alpha=0.2,
            markersize=3,
            zorder=zorder,
        )
        plt.xlabel("Frame index [-]")
        fig.supylabel(METRIC_LABEL)
        plt.grid()
        plt.title("Subject II", **title_kwargs)
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(
            h,
            l,
            loc="outside upper center",
            ncol=2,
            frameon=False,
        )
        plt.savefig(save_dir / "gcnr_valve_over_time.png")
        plt.savefig(save_dir / "gcnr_valve_over_time.pdf")
        zea.log.info(
            f"Saved gCNR valve over time plots to {zea.log.yellow(save_dir / 'gcnr_valve_over_time.png')}"
        )


if __name__ == "__main__":
    main()
