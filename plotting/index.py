import concurrent.futures
import os
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tqdm

from zea import Config
from zea.internal.cache import ZEA_CACHE_DIR
from zea.internal.core import hash_elements

DATA_ROOT = "/mnt/z/usbmd/Wessel/"
DATA_FOLDER = Path(DATA_ROOT) / "eval_echonet_dynamic_test_set"


def index_sweep_data(sweep_dirs: str | List[str], cache=True):
    if isinstance(sweep_dirs, str):
        sweep_dirs = [sweep_dirs]

    hashed = hash_elements(sweep_dirs)
    cache_path = ZEA_CACHE_DIR / f".index_sweep_data_{hashed}.pkl"
    if cache and cache_path.exists():
        print("Loading cached sweep index...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

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

    if cache:
        with open(cache_path, "wb") as f:
            pickle.dump(lookup_table, f)

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
