"""Compares normal diffusion and sequential diffusion PSNR vs diffusion steps."""

import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_yaml(filepath):
    """Load YAML file."""
    with open(filepath, "r") as f:
        # return yaml.safe_load(f)
        return yaml.unsafe_load(f)


def extract_sweep_data(sweep_dir, keys_to_extract=["mse", "psnr"]):
    """Copy of extract_sweep_data to stay compatible with the timings run."""
    sweep_details_path = os.path.join(sweep_dir, "sweep_details.yaml")
    if not os.path.exists(sweep_details_path):
        raise FileNotFoundError(f"Missing sweep_details.yaml in {sweep_dir}")

    sweep_details = load_yaml(sweep_details_path)
    agent_type = sweep_details["agent_type"]

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for run_dir in sorted(os.listdir(sweep_dir)):
        run_path = os.path.join(sweep_dir, run_dir)
        if not os.path.isdir(run_path):
            continue

        config_path = os.path.join(run_path, "config.yaml")
        metrics_path = os.path.join(run_path, "metrics.npz")

        if not os.path.exists(config_path) or not os.path.exists(metrics_path):
            continue

        config = load_yaml(config_path)
        metrics = np.load(metrics_path)

        num_lines = config.get("action_selection", {}).get("num_lines_to_sample")
        initial_step = config.get("diffusion_inference", {}).get("initial_step")
        total_num_steps = config.get("diffusion_inference", {}).get("num_steps")
        n_steps = total_num_steps - initial_step
        selection_strategy = config.get("action_selection", {}).get(
            "selection_strategy"
        )

        if num_lines is None or selection_strategy is None:
            continue

        for metric_name in keys_to_extract:
            metric_values = metrics[metric_name]

            if isinstance(metric_values, np.ndarray) and metric_values.size > 0:
                # Compute mean per sequence (to handle varying frame counts)
                sequence_means = np.mean(metric_values, axis=-1)  # Average over frames
                results[metric_name][selection_strategy][n_steps].append(sequence_means)
            else:
                print(f"Skipping empty metric '{metric_name}' in {run_path}")

    return results, agent_type


def plot_timings(
    results: list,
    names: list,
    save_root=None,
    metric="psnr",
    context=None,
    dark=False,
):
    """Plots all sweeps on the same figure with error bars showing SEM."""

    if context is None:
        context = {}

    markers = ["o", "x", "D", "^", "v", "x", "*"]
    for i, (method, name) in enumerate(zip(results, names)):
        result = method[0][metric]["greedy_entropy"]
        psnr = np.stack(list(result.values()), axis=1)
        psnr_mean = np.mean(psnr, axis=0)
        sem_values = np.std(psnr, axis=0) / np.sqrt(len(psnr))

        steps = np.array(list(result.keys()))

    with plt.style.context(context):
        plt.figure(figsize=(3.5, 2.5))
        # Create broken axis plot: left axis for steps < 100, right axis for step 500
        fig, (ax, ax2) = plt.subplots(
            1, 2, sharey=True, gridspec_kw={"width_ratios": [7, 1]}
        )
        for i, (method, name) in enumerate(zip(results, names)):
            result = method[0][metric]["greedy_entropy"]
            psnr = np.stack(list(result.values()), axis=1)
            psnr_mean = np.mean(psnr, axis=0)
            sem_values = np.std(psnr, axis=0) / np.sqrt(len(psnr))
            steps = np.array(list(result.keys()))
            idx_left = steps > 0
            idx_right = steps > 0
            ax.errorbar(
                steps[idx_left],
                psnr_mean[idx_left],
                yerr=sem_values[idx_left],
                marker=markers[i],
                capsize=3,
                label=name,
            )
            ax2.errorbar(
                steps[idx_right],
                psnr_mean[idx_right],
                yerr=sem_values[idx_right],
                marker=markers[i],
                capsize=3,
            )
        ax.set_xlim(0, 105)
        ax2.set_xlim(490, 510)
        fig.supxlabel("Diffusion Steps [-]")
        ax.set_ylabel("PSNR [dB]")
        ax.grid(True)
        ax2.grid(True)
        ax.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax.yaxis.tick_left()
        ax2.tick_params(left=False, labelleft=False)

        # Draw break marks on the x-axis
        d = 0.5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(
            marker=[(-d, -1), (d, 1)],
            markersize=6,
            linestyle="none",
            color="k",
            mec="k",
            mew=1,
            clip_on=False,
        )
        ax.plot([1, 1], [0, 1], transform=ax.transAxes, **kwargs)
        ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)

        fig.legend(loc="outside upper center", ncol=2, frameon=False)
        ax2.set_xticks([500])  # Hide x ticks on the right axis
        ax.set_xticks([10, 50, 100])

        for ext in [".pdf", ".png"]:
            save_path = os.path.join(save_root, "ulsa_timings" + ext)
            plt.savefig(
                save_path,
                dpi=300,
                transparent=True,
            )
            print(f"Saved to {save_path}")


if __name__ == "__main__":
    seqdiff = "/mnt/z/Ultrasound-BMD/Ultrasound-BMd/data/Wessel/output/lud/active_sampling/recon_vs_fps2/sweep_2025_04_08_124401"
    normal_diff = "/mnt/z/Ultrasound-BMD/Ultrasound-BMd/data/Wessel/output/lud/active_sampling/normal_diffusion_num_steps2/sweep_2025_04_08_115535"
    SAVE_ROOT = "."

    x = extract_sweep_data(seqdiff)
    y = extract_sweep_data(normal_diff)

    # context = Path("styles/nvmu.mplstyle")
    context = Path("styles/ieee-tmi.mplstyle")
    assert context.exists(), f"Context file {context} does not exist."
    plot_timings(
        [x, y],
        ["SeqDiff", "Regular"],
        save_root=SAVE_ROOT,
        context=context,
        dark=False,
    )
