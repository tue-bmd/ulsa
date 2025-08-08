"""This script exports qualitative results on the EchoNet-Dynamic dataset."""

import os

os.environ["KERAS_BACKEND"] = "numpy"

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from keras import ops

sys.path.append("/ulsa")

from ulsa.io_utils import postprocess_agent_results
from zea import Config

DATA_ROOT = "/mnt/z/prjs0966"
DATA_FOLDER = Path(DATA_ROOT) / "oisin/ULSA_out/eval_echonet_dynamic_test_set"
N_PATIENTS = 3
FIGSIZE = (3.5, 2.0 * N_PATIENTS / 3)  # single column
# FIGSIZE = (7.16, 2.5 * N_PATIENTS / 2)  # two columns
FRAME_IDX = 20


plt.style.use("styles/ieee-tmi.mplstyle")

sweep_dir = DATA_FOLDER / "sharding_sweep_2025-08-05_14-42-40"
run_dirs = sweep_dir.glob("run_*")

patients = []
while len(patients) < N_PATIENTS:
    run_dir = next(run_dirs)
    agent_config = Config.from_yaml(run_dir / "config.yaml")
    if agent_config.action_selection.selection_strategy != "greedy_entropy":
        continue
    if agent_config.action_selection.n_actions != 7:
        continue
    if (run_dir / "metrics.npz").exists():
        patients.append(run_dir)

columns = [
    "Acquisitions",
    "Reconstruction",
    "Variance",
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
fig, axs = plt.subplots(N_PATIENTS, len(columns), figsize=FIGSIZE)
for p in range(N_PATIENTS):
    path = patients[p] / "metrics.npz"
    data = np.load(path)
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
    measurement = postprocess_agent_results(
        data["measurements"][FRAME_IDX].squeeze(-1),
        io_config=io_config,
        scan_convert_order=0,
        image_range=image_range,
        fill_value="white",
    )
    belief_distributions = data["belief_distributions"][FRAME_IDX].squeeze(-1)
    variance = ops.var(belief_distributions, axis=0)
    variance = postprocess_agent_results(
        variance,
        io_config=io_config,
        scan_convert_order=1,
        image_range=[0, variance.max()],
        fill_value="transparent",
    )
    variance = np.clip(variance, None, np.nanpercentile(variance, 99))

    axs[p, 3].imshow(target, **imshow_kwargs)
    axs[p, 2].imshow(
        variance,
        cmap="inferno",
        vmin=0,
        interpolation=interpolation,
    )
    axs[p, 1].imshow(reconstruction, **imshow_kwargs)
    axs[p, 0].imshow(measurement, **imshow_kwargs)

for idx, column in enumerate(columns):
    axs[0, idx].set_title(column)

for ax in axs.flat:
    ax.axis("off")

exts = [".png", ".pdf"]
for ext in exts:
    save_path = f"./qualitative_results_echonet{ext}"
    fig.savefig(save_path, dpi=300)
    print(f"Saved qualitative results to {save_path}")
