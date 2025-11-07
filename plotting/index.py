import concurrent.futures
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import scipy
import tqdm

from benchmark_active_sampling_ultrasound import get_config_value
from ulsa.downstream_task import compute_dice_score
from zea import Config, log
from zea.internal.cache import cache_output

DATA_ROOT = "/mnt/z/usbmd/Wessel/"
DATA_FOLDER = Path(DATA_ROOT) / "eval_echonet_dynamic_test_set"


@cache_output(verbose=True)
def index_sweep_data(sweep_dirs: str | List[str]):
    if isinstance(sweep_dirs, str):
        sweep_dirs = [sweep_dirs]

    print("Discovering runs in sweep directories...")
    run_dirs = []
    for sweep_dir in sweep_dirs:
        sweep_dir = Path(sweep_dir)
        with os.scandir(sweep_dir) as it:
            run_dirs.extend(Path(entry.path) for entry in it if entry.is_dir())
    print(f"Found {len(run_dirs)} runs.")

    def process_run(run_path):
        config_path = run_path / "config.yaml"
        metrics_path = run_path / "metrics.npz"
        filepath_yaml = run_path / "target_filepath.yaml"
        if not all(p.exists() for p in [config_path, metrics_path, filepath_yaml]):
            print(f"Skipping incomplete run: {run_path}")
            return None
        target_file = Config.from_yaml(str(filepath_yaml))["target_filepath"]
        target_file = str(target_file).replace(
            "/projects/0/prjs0966/data", "/mnt/z/Ultrasound-BMd/data"
        )
        filename = Path(target_file).name
        return (run_path, target_file, filename)

    lookup_table = []
    print("Indexing runs...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(
            tqdm.tqdm(executor.map(process_run, run_dirs), total=len(run_dirs))
        )
        lookup_table = [r for r in results if r is not None]

    return lookup_table


def random_patients(sweep_dirs, n_samples: int, seed=42):
    generator = index_sweep_data(sweep_dirs)

    data_frame = pd.DataFrame(generator, columns=["run_path", "filepath", "filename"])
    unique_filenames = data_frame["filename"].unique()
    rng = np.random.default_rng(seed)
    random_filenames = rng.choice(unique_filenames, size=n_samples, replace=False)

    for random_filename in random_filenames:
        sample_rows = data_frame[data_frame["filename"] == random_filename]
        yield sample_rows["run_path"].tolist(), random_filename


def load_patients_by_name(sweep_dirs, patient_names: List[str]):
    generator = index_sweep_data(sweep_dirs)

    data_frame = pd.DataFrame(generator, columns=["run_path", "filepath", "filename"])

    for patient_name in patient_names:
        sample_rows = data_frame[data_frame["filename"] == patient_name]
        yield sample_rows["run_path"].tolist(), patient_name


def mask_has_too_many_blobs(mask_sequence, max_blobs=1, max_bad_frames=5):
    """
    Returns True if the mask sequence has more than max_blobs in more than max_bad_frames.
    mask_sequence: np.ndarray or list, shape (T, H, W) or (T, H, W, 1) or list of (H, W)
    """
    if isinstance(mask_sequence, np.ndarray):
        if mask_sequence.ndim == 4 and mask_sequence.shape[-1] == 1:
            mask_sequence = [
                mask_sequence[t, ..., 0] for t in range(mask_sequence.shape[0])
            ]
        elif mask_sequence.ndim == 3:
            mask_sequence = [mask_sequence[t] for t in range(mask_sequence.shape[0])]
        else:
            mask_sequence = [mask_sequence]
    bad_frame_count = 0
    for mask in mask_sequence:
        mask_arr = np.squeeze(mask)
        labeled, num_blobs = scipy.ndimage.label(mask_arr > 0)
        if num_blobs > max_blobs:
            bad_frame_count += 1
    return bad_frame_count > max_bad_frames


def extract_run_dir(
    run,
    keys_to_extract=["mse", "psnr"],
    x_axis_key="action_selection.n_actions",
    strategies_to_plot=None,
    include_only_these_files=None,
    ef_lookup=None,
    max_blobs=1,
    max_bad_frames=5,
):
    run_path, target_file, _ = run
    config_path = os.path.join(run_path, "config.yaml")
    metrics_path = os.path.join(run_path, "metrics.npz")
    config = Config.from_yaml(config_path)
    metrics = np.load(metrics_path, allow_pickle=True)

    if (
        include_only_these_files is not None
        and target_file not in include_only_these_files
    ):
        return

    x_value = get_config_value(config, x_axis_key)
    if x_value is None:
        log.warning(f"Skipping {run_path} due to missing x_axis_key: {x_axis_key}.")
        return

    selection_strategy = config.get("action_selection", {}).get("selection_strategy")
    if selection_strategy is None:
        log.warning(
            f"Skipping {run_path} due to missing selection_strategy: {selection_strategy}."
        )
        return

    if strategies_to_plot is not None and selection_strategy not in strategies_to_plot:
        return

    filestem = Path(target_file).stem
    ef_value = (
        ef_lookup[filestem] if ef_lookup is not None and filestem in ef_lookup else None
    )

    # Use in-file ground truth and predicted masks for DICE
    if (
        "segmentation_mask_targets" in metrics
        and "segmentation_mask_reconstructions" in metrics
    ):
        gt_masks = np.array(metrics["segmentation_mask_targets"])
        # Only keep if gt_masks pass the blob filter
        if mask_has_too_many_blobs(
            gt_masks, max_blobs=max_blobs, max_bad_frames=max_bad_frames
        ):
            mean_dice = None
            too_many_blobs = True
        else:
            pred_masks = np.array(metrics["segmentation_mask_reconstructions"])
            dice_score = compute_dice_score(pred_masks, gt_masks)
            mean_dice = np.mean(dice_score)
            too_many_blobs = False
    else:
        mean_dice = None

    metric_results = {}
    for metric_name in keys_to_extract:
        if metric_name not in metrics:
            continue
        metric_values = metrics[metric_name]
        if isinstance(metric_values, np.ndarray) and metric_values.size > 0:
            sequence_means = np.mean(metric_values, axis=-1)
            metric_results[metric_name] = sequence_means

    return {
        "EF": ef_value,
        "selection_strategy": selection_strategy,
        "x_value": x_value,
        "filepath": target_file,
        "filestem": filestem,
        "dice": mean_dice,
        "too_many_blobs": too_many_blobs,
        **metric_results,
    }


@cache_output(verbose=True)
def extract_sweep_data(sweep_dirs: str, **kwargs):
    """Load all the metrics from the run_benchmark function, using in-file ground truth masks."""

    generator = index_sweep_data(sweep_dirs)
    _extract_run_dir = lambda run: extract_run_dir(run, **kwargs)

    print("Extracting sweep data...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(
            tqdm(executor.map(_extract_run_dir, generator), total=len(generator))
        )
    results = [r for r in results if r is not None]
    return pd.DataFrame(results)
