"""This script exports qualitative results on the EchoNet-Dynamic dataset."""

import os

os.environ["KERAS_BACKEND"] = "numpy"
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import ops
from keras.utils import Progbar

sys.path.append("/ulsa")

from plotting.index import load_patients_by_name, random_patients
from plotting.plot_utils import get_inset
from ulsa.entropy import pixelwise_entropy
from ulsa.io_utils import color_to_value, postprocess_agent_results
from zea import Config

DATA_ROOT = "/mnt/z/usbmd/Wessel/ulsa"
DATA_FOLDER = Path(DATA_ROOT) / "eval_echonet_dynamic_test_set"
SUBSAMPLED_PATHS = [
    DATA_FOLDER / "sharding_sweep_2025-08-05_14-35-11",
    DATA_FOLDER / "sharding_sweep_2025-08-05_14-42-40",
]

N_PATIENTS = 3
FRAME_IDX = 20
N_ACTIONS = [7, 14, 28]
METHOD = "greedy_entropy"
QUICK_MODE = False

if not QUICK_MODE:
    plt.style.use("styles/ieee-tmi.mplstyle")
    plt.rcParams.update({"figure.constrained_layout.use": False})

# Loading same randomly selected patients as before
patients = list(
    load_patients_by_name(
        SUBSAMPLED_PATHS,
        [
            "0X15C904A855E4FF2B.hdf5",
            "0X15BA82A1F6BF8B6.hdf5",
            "0X33646D65192ECB1B.hdf5",
        ],
    )
)

# If not enough patients, sample some more randomly
if len(patients) < N_PATIENTS:
    patients += list(
        random_patients(SUBSAMPLED_PATHS, len(patients) - N_PATIENTS, seed=0)
    )

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

grid_shape = (N_PATIENTS, 4)
figsize = (7.16, grid_shape[0])  # (width, height) in inches
wspace = 0.04
wspace_inner = -0.2
hspace = 0.07
hspace_inner = 0.02
inner_grid_shape = (2, 2)
fig = plt.figure(figsize=figsize)

outer = gridspec.GridSpec(
    *grid_shape,
    figure=fig,
    wspace=wspace,
    hspace=hspace,
    left=0.0,
    right=1.0,
    top=0.93,  # <1.0 to leave space for titles
    bottom=0.0,
    width_ratios=[1.5, 1.5, 1.5, 1.0],
)

print("Generating figure...")
pbar = Progbar(len(patient_names) * len(N_ACTIONS))

results = []
for patient_id, patient_name in enumerate(patient_names):
    patient = patients[patients["name"] == patient_name]
    for i, n_actions in enumerate(N_ACTIONS):
        p = patient[patient["n_actions"] == n_actions].iloc[0]
        path = p["run_dir"] / "metrics.npz"
        data = np.load(path)

        if patient_id == 0:
            ax = fig.add_subplot(outer[patient_id, i])
            ax.axis("off")
            ax.set_title(f"{n_actions} / 112")

        psnr = data["psnr"][FRAME_IDX].item()

        reconstruction = postprocess_agent_results(
            data["reconstructions"][FRAME_IDX].squeeze(-1),
            io_config=io_config,
            scan_convert_order=scan_convert_order,
            image_range=image_range,
            reconstruction_sharpness_std=reconstruction_sharpness_std,
            fill_value="transparent",
        )

        measurement = data["measurements"][FRAME_IDX].squeeze(-1)
        mask = data["masks"][FRAME_IDX].squeeze(-1)
        no_measurement_color = "gray"
        no_measurement_color = color_to_value(image_range, no_measurement_color)
        measurement = np.where(mask, measurement, no_measurement_color)
        measurement = postprocess_agent_results(
            measurement,
            io_config=io_config,
            scan_convert_order=0,
            image_range=image_range,
            fill_value="transparent",
        )
        belief_distributions = data["belief_distributions"][FRAME_IDX].squeeze(-1)
        entropy = ops.squeeze(
            pixelwise_entropy(belief_distributions[None], entropy_sigma=255), axis=0
        )
        results.append((patient_id, i, reconstruction, measurement, entropy, psnr))

    target = postprocess_agent_results(
        data["targets"][FRAME_IDX].squeeze(-1),
        io_config=io_config,
        scan_convert_order=scan_convert_order,
        image_range=image_range,
        fill_value="transparent",
    )
    target_ax = fig.add_subplot(outer[patient_id, -1])
    target_ax.imshow(target, **imshow_kwargs)
    target_ax.axis("off")
    if patient_id == 0:
        target_ax.set_title("Target")

max_percentile = -np.inf
for result in results:
    patient_id, i, reconstruction, measurement, entropy, psnr = result
    curr_percentile = np.nanpercentile(entropy, 98.5)
    print(curr_percentile)
    max_percentile = max(max_percentile, curr_percentile)
print(f"Max: {max_percentile}")


for patient_id, i, reconstruction, measurement, entropy, psnr in results:
    inner = gridspec.GridSpecFromSubplotSpec(
        *inner_grid_shape,
        subplot_spec=outer[patient_id, i],
        width_ratios=[2, 1],
        height_ratios=[1, 1],
        wspace=wspace_inner,
        hspace=hspace_inner * inner_grid_shape[0],
    )

    ax_big = fig.add_subplot(inner[:, 0])
    ax_big.imshow(reconstruction, **imshow_kwargs)
    ax_big.axis("off")
    ax_big.text(
        0.0,  # left=0.0
        0.5,  # upper=1.0
        f"{psnr:.1f} dB",
        transform=ax_big.transAxes,
        fontsize=7,
        rotation=45,
        color="gray",
    )

    entropy = postprocess_agent_results(
        entropy,
        io_config=io_config,
        scan_convert_order=1,
        image_range=[0, max_percentile],
        fill_value="transparent",
    )

    ax_top = fig.add_subplot(inner[0, 1])
    ax_top.imshow(
        entropy,
        cmap="inferno",
        vmin=0,
        vmax=255,
        interpolation=interpolation,
    )
    ax_top.axis("off")

    ax_bottom = fig.add_subplot(inner[1, 1])
    ax_bottom.imshow(measurement, **imshow_kwargs)
    ax_bottom.axis("off")

    pbar.add(1)

if not QUICK_MODE:
    exts = [".png", ".pdf"]
    dpi = 600
else:
    exts = [".png"]
    dpi = 150
for ext in exts:
    print("Saving figure...")
    save_path = f"./qualitative_results_echonet{ext}"
    fig.savefig(save_path, dpi=dpi)
    print(f"Saved qualitative results to {save_path}")
