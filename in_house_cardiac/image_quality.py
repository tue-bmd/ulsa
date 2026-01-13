"""Script to evaluate image quality of in-house cardiac ultrasound data."""

import os

os.environ["KERAS_BACKEND"] = "jax"
import zea

zea.init_device()
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import exposure  # pip install scikit-image

sys.path.append("/ulsa")

import ulsa.metrics  # for nrmse
from plotting.plot_utils import METRIC_NAMES, ViolinPlotter, write_roman
from zea.metrics import Metrics


def df_to_dict(df: pd.DataFrame, metric_name: str):
    result = {}
    for _, row in df.iterrows():
        strategy = row["strategy"]
        subject = row["subject"]
        metric = row[metric_name]
        if strategy not in result:
            result[strategy] = {}
        result[strategy][subject] = metric

    return result


def main():
    DATA_ROOT = Path("/mnt/z/usbmd/Wessel/ulsa/eval_in_house/cardiac_fundamental/")
    SAVE_DIR = Path("output/in_house_cardiac/")
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

    metric_names = ["psnr", "lpips", "nrmse"]
    drop_first_n_frames = 3  # filter startup artifacts of temporal model
    metrics = Metrics(metric_names, image_range=[0, 255])
    group_names.pop("focused")

    subject_names = []
    data_frame = []
    for i, subject in enumerate(subjects):
        # Load annotations and results
        subject_name = write_roman(i + 1)
        subject_names.append(subject_name)
        rf = DATA_ROOT / f"{subject}_results.npz"
        results = np.load(rf, allow_pickle=True)

        focused = results["focused"][drop_first_n_frames:]
        focused_uint8 = zea.display.to_8bit(
            focused, results["focused_dynamic_range"], pillow=False
        )
        diverging = results["diverging"][drop_first_n_frames:]
        diverging = exposure.match_histograms(diverging, focused)
        diverging_uint8 = zea.display.to_8bit(
            diverging, results["focused_dynamic_range"], pillow=False
        )
        reconstructions = results["reconstructions"][drop_first_n_frames:]
        reconstructions = zea.display.to_8bit(
            reconstructions, results["reconstruction_range"], pillow=False
        )

        assert focused_uint8.shape == diverging_uint8.shape == reconstructions.shape, (
            "Shapes of focused, diverging and reconstructions must be the same."
        )
        div_res = metrics(
            focused_uint8[..., None], diverging_uint8[..., None], average_batches=False
        )
        rec_res = metrics(
            focused_uint8[..., None], reconstructions[..., None], average_batches=False
        )
        data_frame.append({"subject": subject_name, "strategy": "diverging", **div_res})
        data_frame.append(
            {"subject": subject_name, "strategy": "reconstructions", **rec_res}
        )
    data_frame = pd.DataFrame(data_frame)

    for metric_name in metric_names:
        data_dict = df_to_dict(data_frame, metric_name)

        # Violin plot & over time plot for all
        violin = ViolinPlotter(group_names, xlabel="Subjects")
        for ext in [".png", ".pdf"]:
            violin.plot(
                data_dict,
                SAVE_DIR / f"in_house_{metric_name}_violin{ext}",
                metric_name=METRIC_NAMES[metric_name],
                context="styles/ieee-tmi.mplstyle",
            )


if __name__ == "__main__":
    main()
