"""
Generates PNGs for teaser figure.
"""

import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

import zea
from ulsa.in_house_cardiac.load_results import load_from_run_dir

harmonic_dir = "/mnt/z/usbmd/ulsa/eval_in_house/teaser/harmonic/"
harmonic_file = "20251222_s3_a4ch_line_dw_0000"
subprocess.call(
    [
        "python3",
        "benchmarking_scripts/eval_in_house_cardiac.py",
        "--save_dir",
        harmonic_dir,
        "--files",
        f"/mnt/z/usbmd/ulsa/2026_ULSA_A4CH_S51/harmonic/{harmonic_file}.hdf5",
        "--selection_strategies",
        "greedy_entropy",
        "--agent_config_path",
        "./configs/cardiac_112_3_frames_harmonic.yaml",
        "--low_pct",
        "44",
        "--high_pct",
        "99.99",
        "--n_transmits",
        "5",  # actually 10 for pulse inversion
    ]
)

fundamental_dir = "/mnt/z/usbmd/ulsa/eval_in_house/teaser/fundamental/"
fundamental_file = "20240701_P1_A4CH_0001"
subprocess.call(
    [
        "python3",
        "benchmarking_scripts/eval_in_house_cardiac.py",
        "--save_dir",
        fundamental_dir,
        "--files",
        f"/mnt/z/usbmd/ulsa/2026_ULSA_A4CH_S51/fundamental/{fundamental_file}.hdf5",
        "--selection_strategies",
        "greedy_entropy",
        "--agent_config_path",
        "./configs/cardiac_112_3_frames.yaml",
        "--low_pct",
        "18",
        "--high_pct",
        "95",
        "--n_transmits",
        "10",
    ],
)


os.environ["KERAS_BACKEND"] = "jax"
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
zea.init_device()
plt.rcdefaults()

_, diverging_harmonic, cognitive_harmonic, _, _, _, _, heatmap, _ = load_from_run_dir(
    Path(harmonic_dir) / harmonic_file,
    frame_idx=68,
    selection_strategy="greedy_entropy",
    scan_convert_resolution=0.5,
    fill_value="transparent",
    no_measurement_color="transparent",
)

_, diverging_fundamental, _, _, _, _, _, _, _ = load_from_run_dir(
    Path(fundamental_dir) / fundamental_file,
    frame_idx=24,
    selection_strategy="greedy_entropy",
    scan_convert_resolution=0.5,
    fill_value="transparent",
    no_measurement_color="transparent",
)


cognitive_harmonic = cognitive_harmonic[0]
diverging_harmonic = diverging_harmonic[0]
diverging_fundamental = diverging_fundamental[0]
heatmap = heatmap[0]

save_dir = Path("./output/teaser")
save_dir.mkdir(parents=True, exist_ok=True)
plt.imsave(
    save_dir / "cognitive.png", cognitive_harmonic, cmap="gray", vmin=0, vmax=255
)
plt.imsave(
    save_dir / "diverging_harmonic.png",
    diverging_harmonic,
    cmap="gray",
    vmin=0,
    vmax=255,
)
plt.imsave(
    save_dir / "diverging_fundamental.png",
    diverging_fundamental,
    cmap="gray",
    vmin=0,
    vmax=255,
)
plt.imsave(save_dir / "heatmap.png", heatmap)
