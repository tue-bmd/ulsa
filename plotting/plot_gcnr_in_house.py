"""Script to calculate and plot gCNR (Generalized Contrast-to-Noise Ratio)
for the fundamental en harmonic data which is present in the paper."""

import os

if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "jax"
import zea

if __name__ == "__main__":
    zea.init_device()

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt

from ulsa.in_house_cardiac.gcnr import METRIC_LABEL, sort_by_names, swap_layer
from ulsa.in_house_cardiac.gcnr import load_results as load_fundamental_results
from ulsa.in_house_cardiac.gcnr_itk import load_results as load_harmonic_results
from ulsa.plotting.plot_utils import ViolinPlotter, write_roman
from zea import log

SAVE_DIR = Path("output/gcnr")


def get_mean_gcnr(gcnr_dict, method="diverging"):
    sum_gcnr = 0.0
    len_gcnr = 0
    for subject, values in gcnr_dict.items():
        sum_gcnr += sum(values[method])
        len_gcnr += len(values[method])
    return sum_gcnr, len_gcnr


def print_mean_gcnr(
    gcnr_dict_fund,
    gcnr_dict_harm,
    rec_methods=("greedy_entropy", "diverging"),
    acq_methods=("fundamental", "harmonic"),
):
    if isinstance(rec_methods, str):
        rec_methods = [rec_methods]
    if isinstance(acq_methods, str):
        acq_methods = [acq_methods]

    sum_gcnr = 0.0
    len_gcnr = 0
    for rec_method, acq_method in product(rec_methods, acq_methods):
        if acq_method == "fundamental" and rec_method == "greedy_entropy":
            rec_method = "reconstructions"
        if acq_method == "fundamental":
            _sum_gcnr, _len_gcnr = get_mean_gcnr(gcnr_dict_fund, method=rec_method)
        else:
            _sum_gcnr, _len_gcnr = get_mean_gcnr(gcnr_dict_harm, method=rec_method)
        sum_gcnr += _sum_gcnr
        len_gcnr += _len_gcnr

    mean_gcnr = sum_gcnr / len_gcnr

    print(
        f"Mean gCNR ({' & '.join(acq_methods)} acquisition for {', '.join(rec_methods)}): {mean_gcnr:.4f}"
    )

    return sum_gcnr, len_gcnr, mean_gcnr


if __name__ == "__main__":
    plt.rcdefaults()  # Reset to default matplotlib style
    subjects, group_names, relative_gncr_dict, _, _, gcnr_all, _ = (
        load_fundamental_results()
    )
    subjects_hi, group_names_hi, relative_gncr_dict_hi, _, gcnr_all_hi, _ = (
        load_harmonic_results()
    )

    for acq_method in [("fundamental", "harmonic"), ("harmonic",), ("fundamental",)]:
        # mean gcnr for diverging
        _, _, mean_gcnr_div = print_mean_gcnr(
            gcnr_all,
            gcnr_all_hi,
            rec_methods=("diverging"),
            acq_methods=acq_method,
        )

        # mean gcnr for greedy entropy
        _, _, mean_gcnr_act = print_mean_gcnr(
            gcnr_all,
            gcnr_all_hi,
            rec_methods=("greedy_entropy"),
            acq_methods=acq_method,
        )

        gcnr_improvement = ((mean_gcnr_act - mean_gcnr_div) / mean_gcnr_div) * 100.0
        print(f"-- gCNR improvement ({acq_method}): {gcnr_improvement:.2f}%")

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
