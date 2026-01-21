import os

os.environ["KERAS_BACKEND"] = "jax"
import zea

zea.init_device()
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ulsa.io_utils import color_to_value, postprocess_agent_results, side_by_side_gif
from ulsa.plotting.index import random_patients
from ulsa.utils import find_best_cine_loop

# Parameters
save_dir = "./output/echonet-examples/"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
METHOD = "greedy_entropy"
N_ACTIONS = [7, 14, 28]
interpolation = "nearest"
vmin = 0
vmax = 255
drop_first_n_frames = 4
scan_convert_resolution = 0.8
n_samples = 20

io_config = zea.Config(scan_convert=True, scan_conversion_angles=(-45, 45))
scan_convert_order = 0
image_range = (vmin, vmax)
reconstruction_sharpness_std = 0.025
no_measurement_color = "gray"
no_measurement_color = color_to_value(image_range, no_measurement_color)

# Get random patients
patients = random_patients(
    [
        "/mnt/z/usbmd/ulsa/Np_2/eval_echonet_dynamic_test_set/sweep_2026_01_08_225505_654881"
    ],
    n_samples=n_samples,
)

already_done = list(save_dir.glob("*.webm"))
already_done_dict = []
for done in already_done:
    stem = done.stem
    parts = stem.split("_")
    if len(parts) < 2:
        continue
    name = parts[0]
    n_actions = int(parts[-1])

    already_done_dict.append({"name": name, "n_actions": n_actions})
already_done_df = pd.DataFrame(already_done_dict)


def _check_if_done(name, n_actions):
    if already_done_df.empty:
        return False

    done = already_done_df[
        (already_done_df["name"] == name) & (already_done_df["n_actions"] == n_actions)
    ]
    return not done.empty


# Preload relevant data
print("Preloading relevant data...")
results = []
for run_dirs, name in patients:
    for run_dir in run_dirs:
        agent_config = zea.Config.from_yaml(run_dir / "config.yaml")
        if agent_config.action_selection.selection_strategy != METHOD:
            continue
        n_actions = agent_config.action_selection.n_actions
        if n_actions not in N_ACTIONS:
            continue
        stem = str(name).split(".")[0]
        if _check_if_done(stem, agent_config.action_selection.n_actions):
            print(f"Skipping already done: {name} with {n_actions} actions")
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
patient_stems = [str(n).split(".")[0] for n in patient_names]
print("Patients to process:")
print(patient_stems)

# Generate gifs
print("Generating gifs...")
pbar = tqdm(total=len(patient_names) * len(N_ACTIONS))
for patient_id, patient_name in enumerate(patient_names):
    patient = patients[patients["name"] == patient_name]
    for i, n_actions in enumerate(N_ACTIONS):
        p = patient[patient["n_actions"] == n_actions].iloc[0]
        path = p["run_dir"] / "metrics.npz"

        data = np.load(path)
        reconstructions = data["reconstructions"].squeeze(-1)
        masks = data["masks"].squeeze(-1)
        measurements = data["measurements"].squeeze(-1)

        if i == 0 and patient_id == 0:
            _, height, width = reconstructions.shape
            coordinates, _ = zea.display.compute_scan_convert_2d_coordinates(
                (height, width),
                (0, height),
                np.deg2rad(io_config.scan_conversion_angles),
                resolution=scan_convert_resolution,
            )

        if i == 0:
            targets = data["targets"].squeeze(-1)
            last_frame = (
                find_best_cine_loop(
                    targets[drop_first_n_frames:],
                    min_sequence_length=50,
                    visualize=False,
                )
                + drop_first_n_frames
            )
            targets = targets[drop_first_n_frames:last_frame]
            targets = postprocess_agent_results(
                targets,
                io_config=io_config,
                scan_convert_order=scan_convert_order,
                image_range=image_range,
                fill_value="black",
                coordinates=coordinates,
            )

        reconstructions = reconstructions[drop_first_n_frames:last_frame]
        measurements = measurements[drop_first_n_frames:last_frame]
        masks = masks[drop_first_n_frames:last_frame]

        reconstructions = postprocess_agent_results(
            reconstructions,
            io_config=io_config,
            scan_convert_order=scan_convert_order,
            image_range=image_range,
            reconstruction_sharpness_std=reconstruction_sharpness_std,
            fill_value="black",
            coordinates=coordinates,
        )

        measurements = np.where(masks, measurements, no_measurement_color)
        measurements = postprocess_agent_results(
            measurements,
            io_config=io_config,
            scan_convert_order=0,
            image_range=image_range,
            fill_value="black",
            coordinates=coordinates,
        )

        patient_stem = str(patient_name).split(".")[0]
        side_by_side_gif(
            save_dir / f"{patient_stem}_{n_actions}.webm",
            measurements,
            reconstructions,
            targets,
            fps=30,
            interpolation=interpolation,
            context="styles/website.mplstyle",
            labels=["Measurements", "Cognitive", "Target"],
        )
        pbar.update(1)
pbar.close()
