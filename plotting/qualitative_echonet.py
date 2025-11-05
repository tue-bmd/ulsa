"""This script exports qualitative results on the EchoNet-Dynamic dataset."""

import os

os.environ["KERAS_BACKEND"] = "numpy"

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import ops
from mpl_toolkits.axes_grid1 import ImageGrid

sys.path.append("/ulsa")

from plotting.index import random_patients
from plotting.plot_utils import get_inset
from ulsa.entropy import pixelwise_entropy
from ulsa.io_utils import postprocess_agent_results
from zea import Config

DATA_ROOT = "/mnt/z/usbmd/Wessel/"
DATA_FOLDER = Path(DATA_ROOT) / "eval_echonet_dynamic_test_set"
SUBSAMPLED_PATHS = [
    DATA_FOLDER / "sharding_sweep_2025-08-05_14-35-11",
    DATA_FOLDER / "sharding_sweep_2025-08-05_14-42-40",
]

N_PATIENTS = 3
# FIGSIZE = (3.5, 2.0 * N_PATIENTS / 3)  # single column
FIGSIZE = (7.16, 2.5 * N_PATIENTS / 2)  # two columns
FRAME_IDX = 20
N_ACTIONS = [7, 14, 28]
METHOD = "greedy_entropy"


plt.style.use("styles/ieee-tmi.mplstyle")

patients = list(random_patients(SUBSAMPLED_PATHS, N_PATIENTS, seed=0))
results = []
for run_dirs, name in patients:
    for run_dir in run_dirs:
        agent_config = Config.from_yaml(run_dir / "config.yaml")
        if agent_config.action_selection.selection_strategy != METHOD:
            continue
        if agent_config.action_selection.n_actions not in N_ACTIONS:
            continue
        results.append(
            {
                "name": name,
                "run_dir": run_dir,
                "n_actions": agent_config.action_selection.n_actions,
            }
        )
patients = pd.DataFrame(results)
patient_names = patients["name"].unique()


columns = [
    # "Acquisitions",
    *[f"{n} / 122" for n in N_ACTIONS],
    # "Entropy",
    "Target",
]
interpolation = "nearest"
vmin = 0
vmax = 255

io_config = Config(scan_convert=True, scan_conversion_angles=(-45, 45))
scan_convert_order = 0
image_range = (vmin, vmax)
reconstruction_sharpness_std = 0.025

imshow_kwargs = {
    "cmap": "gray",
    "vmin": vmin,
    "vmax": vmax,
    "interpolation": interpolation,
}
fig = plt.figure(figsize=FIGSIZE)
grid = ImageGrid(
    fig,
    111,
    nrows_ncols=(N_PATIENTS, len(columns)),
    axes_pad=(0.1, 0.1),  # (horizontal, vertical) padding between axes in inches
)
axs = np.array(grid.axes_row)

for idx, column in enumerate(columns):
    axs[0, idx].set_title(column)

for ax in axs.flat:
    ax.axis("off")

for patient_id, patient_name in enumerate(patient_names):
    patient = patients[patients["name"] == patient_name]
    for i, n_actions in enumerate(N_ACTIONS):
        p = patient[patient["n_actions"] == n_actions].iloc[0]
        path = p["run_dir"] / "metrics.npz"
        data = np.load(path)

        psnr = data["psnr"][FRAME_IDX].item()  # TODO: add to plot

        reconstruction = postprocess_agent_results(
            data["reconstructions"][FRAME_IDX].squeeze(-1),
            io_config=io_config,
            scan_convert_order=scan_convert_order,
            image_range=image_range,
            reconstruction_sharpness_std=reconstruction_sharpness_std,
            fill_value="white",
        )
        target = postprocess_agent_results(
            data["targets"][FRAME_IDX].squeeze(-1),
            io_config=io_config,
            scan_convert_order=scan_convert_order,
            image_range=image_range,
            fill_value="white",
        )
        axs[patient_id, -1].imshow(target, **imshow_kwargs)
        axs[patient_id, i].imshow(reconstruction, **imshow_kwargs)


fig.canvas.draw()  # needed for inset positioning

# INSETS
for patient_id, patient_name in enumerate(patient_names):
    patient = patients[patients["name"] == patient_name]
    for i, n_actions in enumerate(N_ACTIONS):
        p = patient[patient["n_actions"] == n_actions].iloc[0]
        path = p["run_dir"] / "metrics.npz"
        data = np.load(path)
        measurement = postprocess_agent_results(
            data["measurements"][FRAME_IDX].squeeze(-1),
            io_config=io_config,
            scan_convert_order=0,
            image_range=image_range,
            fill_value="white",
        )
        belief_distributions = data["belief_distributions"][FRAME_IDX].squeeze(-1)
        entropy = ops.squeeze(
            pixelwise_entropy(belief_distributions[None], entropy_sigma=255), axis=0
        )
        entropy = postprocess_agent_results(
            entropy,
            io_config=io_config,
            scan_convert_order=1,
            image_range=[0, entropy.max()],
            fill_value="transparent",
        )
        # TODO: entropy not normalized to same scale for all images
        entropy = np.clip(entropy, None, np.nanpercentile(entropy, 99.5))

        inset_ax = get_inset(
            fig, axs[patient_id, i], axs[patient_id, i + 1], entropy.shape, height=0.6
        )
        inset_ax.imshow(entropy, cmap="inferno", vmin=0, interpolation=interpolation)
        # axs[patient_id, 0].imshow(measurement, **imshow_kwargs)

# exts = [".png", ".pdf"]
exts = [".png"]
for ext in exts:
    save_path = f"./qualitative_results_echonet{ext}"
    fig.savefig(save_path, dpi=300)
    print(f"Saved qualitative results to {save_path}")
