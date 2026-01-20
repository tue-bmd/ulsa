"""Plot qualitative 3D results as biplanes."""

import os

os.environ["KERAS_BACKEND"] = "numpy"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import random
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from keras import ops

from ulsa.elevation_interpolation.tools import postprocess_3d_data
from ulsa.entropy import pixelwise_entropy
from zea import Config
from zea.visualize import plot_biplanes

# Set reproducible seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

plt.style.use("styles/ieee-tmi.mplstyle")
## Padding (in inches) around axes
plt.rcParams["figure.constrained_layout.h_pad"] = -0.04167
plt.rcParams["figure.constrained_layout.w_pad"] = -0.04167
## Spacing between subplots, relative to the subplot sizes.  Much smaller than for
## tight_layout (figure.subplot.hspace, figure.subplot.wspace) as constrained_layout
## already takes surrounding texts (titles, labels, # ticklabels) into account.
plt.rcParams["figure.constrained_layout.hspace"] = -0.02
plt.rcParams["figure.constrained_layout.wspace"] = -0.02
plt.rcParams["axes.titlepad"] = 10.0  # default is 6.0


# --- CONFIG ---
data_root = Path("/mnt/z/usbmd/Wessel/ulsa/Np_2/eval_3d/qualitative")
n_patients = 3
frame_idx = 4  # or whichever frame you want to visualize
normalization_range = (0, 255)

# Find valid patient runs
run_dirs = list(data_root.glob("run_*"))
patients = []
for run_dir in run_dirs:
    try:
        agent_config = Config.from_yaml(run_dir / "config.yaml")
        if agent_config.action_selection.selection_strategy != "greedy_entropy":
            continue
        if agent_config.action_selection.n_actions != 6:
            continue
        metrics_path = run_dir / "metrics.npz"
        if metrics_path.exists():
            patients.append(run_dir)
    except Exception:
        continue

if len(patients) < n_patients:
    raise RuntimeError("Not enough valid patient metrics.npz files found.")

# Randomly select patients
patients = random.sample(patients, n_patients)

# Output directory
output_dir = Path("./qualitative_3d_biplane")
output_dir.mkdir(exist_ok=True, parents=True)

fig, axs = plt.subplots(n_patients, 4, subplot_kw={"projection": "3d"})

for p, run_dir in enumerate(patients):
    path = run_dir / "metrics.npz"
    data = np.load(path)

    # Load the target file path from target_filepath.yaml
    target_filepath_yaml = run_dir / "target_filepath.yaml"
    with open(target_filepath_yaml, "r") as f:
        target_info = yaml.safe_load(f)
    target_filepath = Path(target_info["target_filepath"])

    # Load the grid from the target file (assuming npz with keys 'rho', 'theta', 'phi')
    # with h5py.File(target_filepath, "r") as f:
    #     rho = f["/scan/frustum/rho"][()]
    #     theta = f["/scan/frustum/theta"][()]
    #     phi = f["/scan/frustum/phi"][()]
    grid = np.load("./styles/grid.npz")
    rho, theta, phi = grid["rho"], grid["theta"], grid["phi"]
    rho_range = (float(np.min(rho)), float(np.max(rho)))
    theta_range = (float(np.min(theta)), float(np.max(theta)))
    phi_range = (float(np.min(phi)), float(np.max(phi)))

    scan_convert_kwargs = {
        "rho_range": rho_range,
        "theta_range": theta_range,
        "phi_range": phi_range,
        "fill_value": np.nan,
    }
    postprocess_fn = partial(
        postprocess_3d_data,
        normalization_range=normalization_range,
        scan_convert_mode="cartesian",
        swap_axes=True,
    )

    # Get 3D arrays for the selected frame
    # Assumed shapes: (frames, ax, az, el, 1) or (frames, ax, az, el)
    def get_vol(key):
        arr = data[key][frame_idx]
        if arr.ndim == 4:
            arr = arr.squeeze(-1)
        return arr

    acquisitions = get_vol("measurements")
    reconstruction = get_vol("reconstructions")
    target = get_vol("targets")
    belief_distributions = data["belief_distributions"][frame_idx].squeeze(-1)
    entropy = pixelwise_entropy(belief_distributions[None], entropy_sigma=255)[0]

    # az, ax, el
    acquisitions = postprocess_fn(
        acquisitions.astype(np.float32),
        scan_convert_kwargs={**scan_convert_kwargs, "order": 0},
    )[0]
    reconstruction = postprocess_fn(
        reconstruction.astype(np.float32),
        scan_convert_kwargs={**scan_convert_kwargs, "order": 3},
    )[0]
    entropy = postprocess_fn(
        entropy.astype(np.float32),
        scan_convert_kwargs={**scan_convert_kwargs, "order": 1},
    )[0]
    entropy = np.clip(entropy, None, np.nanpercentile(entropy, 99.5))
    target = postprocess_fn(
        target.astype(np.float32),
        scan_convert_kwargs={**scan_convert_kwargs, "order": 3},
    )[0]

    n_ax, n_az, n_elev = ops.shape(acquisitions)
    # --> az, el, ax
    acquisitions = np.transpose(acquisitions, (0, 2, 1))
    reconstruction = np.transpose(reconstruction, (0, 2, 1))
    entropy = np.transpose(entropy, (0, 2, 1))
    target = np.transpose(target, (0, 2, 1))

    # List of (volume, cmap, title)
    vols = [
        (acquisitions, "grey", "Acquisitions"),
        (reconstruction, "gray", "Cognitive"),
        (entropy, "inferno", "Entropy"),
        (target, "gray", "Target"),
    ]

    for c, (vol, cmap, title) in enumerate(vols):
        ax = axs[p, c] if n_patients > 1 else axs[c]
        fig, ax = plot_biplanes(
            vol,
            slice_x=n_elev // 2,
            slice_y=n_az // 2,
            cmap=cmap,
            fig=fig,
            ax=ax,
            rasterized=True,
        )
        if p == 0:
            ax.set_title(title, y=0.87)
        ax.axis("off")
        crop_fraction = 0.8
        crop_start = lambda shape: shape - (shape * crop_fraction)
        crop_end = lambda shape: (shape * crop_fraction)
        ax.set_xlim(crop_start(vol.shape[1]), crop_end(vol.shape[1]))
        ax.set_ylim(crop_start(vol.shape[2]), crop_end(vol.shape[2]))
        ax.set_zlim(crop_start(vol.shape[0]), crop_end(vol.shape[0]))

for ax in axs.flat:
    ax.axis("off")

exts = [".png", ".pdf"]
for ext in exts:
    save_path = f"./qualitative_3d_biplane{ext}"
    fig.savefig(save_path)
    print(f"Saved qualitative 3D biplane results to {save_path}")
