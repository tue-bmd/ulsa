"""Script to calculate and plot gCNR (Generalized Contrast-to-Noise Ratio)
for the fundamental en harmonic data which is present in the paper."""

import os

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "jax"
import zea

if __name__ == "__main__":
    zea.init_device()

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from zea import log

sys.path.append("/ulsa")


from in_house_cardiac.gcnr import METRIC_LABEL, sort_by_names, swap_layer
from in_house_cardiac.gcnr import load_results as load_fundamental_results
from in_house_cardiac.gcnr_itk import load_results as load_harmonic_results
from plotting.plot_utils import ViolinPlotter, write_roman

SAVE_DIR = Path("output/gcnr")


def get_mean_gcnr(gcnr_dict, method="diverging"):
    sum_gcnr = 0.0
    len_gcnr = 0
    for subject, values in gcnr_dict.items():
        sum_gcnr += sum(values[method])
        len_gcnr += len(values[method])
    return sum_gcnr, len_gcnr


if __name__ == "__main__":
    plt.rcdefaults()  # Reset to default matplotlib style
    subjects, group_names, relative_gncr_dict, _, _, gcnr_all, _ = (
        load_fundamental_results()
    )
    subjects_hi, group_names_hi, relative_gncr_dict_hi, _, gcnr_all_hi, _ = (
        load_harmonic_results()
    )

    # Compute mean gcnr for active perception and diverging waves
    sum_gcnr, len_gcnr = get_mean_gcnr(gcnr_all, method="diverging")
    sum_gcnr_hi, len_gcnr_hi = get_mean_gcnr(gcnr_all_hi, method="diverging")
    mean_gcnr1 = (sum_gcnr + sum_gcnr_hi) / (len_gcnr + len_gcnr_hi)
    print(
        f"Mean gCNR (diverging waves, both fundamental and harmonic): {mean_gcnr1:.4f}"
    )

    sum_gcnr, len_gcnr = get_mean_gcnr(gcnr_all, method="reconstructions")
    sum_gcnr_hi, len_gcnr_hi = get_mean_gcnr(gcnr_all_hi, method="greedy_entropy")
    mean_gcnr2 = (sum_gcnr + sum_gcnr_hi) / (len_gcnr + len_gcnr_hi)
    print(
        f"Mean gCNR (greedy entropy, both fundamental and harmonic): {mean_gcnr2:.4f}"
    )

    gcnr_improvement = ((mean_gcnr2 - mean_gcnr1) / mean_gcnr1) * 100.0
    print(f"gCNR improvement: {gcnr_improvement:.2f}%")

    group_names.update(group_names_hi)

    subjects_all = subjects + subjects_hi
    subject_ids_all = {s: write_roman(i + 1) for i, s in enumerate(subjects_all)}
    subject_ids = []
    for s in subjects:
        subject_ids.append(subject_ids_all[s])
    subject_ids_hi = []
    for s in subjects_hi:
        subject_ids_hi.append(subject_ids_all[s])

    title_kwargs = {
        "y": 1.0,
        "pad": -10,
        "fontsize": 8,
        "alpha": 0.7,
        "loc": "left",
        "x": 0.03,
    }

    with plt.style.context("styles/ieee-tmi.mplstyle"):
        relative_gncr_dict_roman = {
            subject_ids_all[k]: v for k, v in relative_gncr_dict.items()
        }
        violin = ViolinPlotter(group_names, xlabel=None, ylabel=None)
        fig, axs = plt.subplots(1, 2, sharey=True)
        violin.plot(
            sort_by_names(swap_layer(relative_gncr_dict_roman), group_names.keys()),
            save_path=None,
            x_label_values=relative_gncr_dict_roman.keys(),
            metric_name=METRIC_LABEL,
            context="styles/ieee-tmi.mplstyle",
            ax=axs[0],
            legend_kwargs=None,
        )
        axs[0].set_title("Fundamental imaging", **title_kwargs)

        relative_gncr_dict_hi_roman = {
            subject_ids_all[k]: v for k, v in relative_gncr_dict_hi.items()
        }
        violin.plot(
            sort_by_names(swap_layer(relative_gncr_dict_hi_roman), group_names.keys()),
            save_path=None,
            x_label_values=relative_gncr_dict_hi_roman.keys(),
            metric_name=None,
            context="styles/ieee-tmi.mplstyle",
            ax=axs[1],
            legend_kwargs=None,
        )
        axs[1].set_title("Harmonic imaging", **title_kwargs)
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(
            h,
            l,
            loc="outside upper center",
            ncol=3,
            frameon=False,
        )
        fig.supxlabel("Subjects")
        for ext in [".pdf", ".png"]:
            save_path = f"./output/gcnr/gcnr_violin_combined{ext}"
            plt.savefig(save_path)
            log.info(f"Saved combined violin plot to {log.yellow(save_path)}")
