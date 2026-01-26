"""
Scan Line Sampling Simulation

This script simulates equispaced and uniform random sampling strategies
over a predefined number of time steps, recording the average spatiotemporal
distance to the nearest sampled line.
"""

import os

os.environ["MPLBACKEND"] = "Agg"

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde


def initial_equispaced_lines(n_actions: int, n_possible_actions: int) -> np.ndarray:
    """Generate initial equispaced line indices.

    Args:
        n_actions: Number of lines to select per time step
        n_possible_actions: Total number of possible line locations

    Returns:
        k-hot vector of shape (n_possible_actions,)
    """
    spacing = n_possible_actions // n_actions
    indices = np.arange(0, n_actions * spacing, spacing)
    k_hot = np.zeros(n_possible_actions, dtype=np.float32)
    k_hot[indices] = 1.0
    return k_hot


def next_equispaced_lines(current_lines: np.ndarray) -> np.ndarray:
    """Roll the equispaced lines by one position.

    Args:
        current_lines: Current k-hot vector of selected lines

    Returns:
        New k-hot vector rolled by one position
    """
    return np.roll(current_lines, 1)


def random_uniform_lines(
    n_actions: int, n_possible_actions: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate random uniform line selection.

    Args:
        n_actions: Number of lines to select
        n_possible_actions: Total number of possible line locations
        rng: NumPy random generator

    Returns:
        k-hot vector of shape (n_possible_actions,)
    """
    indices = rng.choice(n_possible_actions, size=n_actions, replace=False)
    k_hot = np.zeros(n_possible_actions, dtype=np.float32)
    k_hot[indices] = 1.0
    return k_hot


def compute_spatiotemporal_distances(
    sampling_history: np.ndarray,
    spatial_weight: float = 1.0,
    temporal_weight: float = 1.0,
) -> dict:
    """Compute spatiotemporal distances to nearest sampled line for each point.

    For each (location, time) point, find the minimum distance to any sampled
    line in the history, where distance is defined as:
        d = sqrt((spatial_weight * delta_x)^2 + (temporal_weight * delta_t)^2)

    Args:
        sampling_history: Binary array of shape (n_time_steps, n_possible_actions)
                         where 1 indicates a sample was taken
        spatial_weight: Weight for spatial distance (default: 1.0)
        temporal_weight: Weight for temporal distance (default: 1.0)

    Returns:
        Dictionary with distance statistics
    """
    n_time_steps, n_possible_actions = sampling_history.shape

    # Get all sampled points as (time, location) pairs
    sampled_times, sampled_locations = np.where(sampling_history > 0)
    sampled_points = np.stack([sampled_times, sampled_locations], axis=1)  # (N, 2)

    if len(sampled_points) == 0:
        return {
            "mean_distance": np.nan,
            "std_distance": np.nan,
            "max_distance": np.nan,
            "median_distance": np.nan,
            "distance_grid": np.full((n_time_steps, n_possible_actions), np.nan),
        }

    # Create grid of all (time, location) points
    times = np.arange(n_time_steps)
    locations = np.arange(n_possible_actions)
    time_grid, loc_grid = np.meshgrid(times, locations, indexing="ij")

    # Compute distance from each grid point to nearest sampled point
    distance_grid = np.full((n_time_steps, n_possible_actions), np.inf)

    for t_samp, x_samp in sampled_points:
        # Compute weighted distance to this sampled point
        dt = temporal_weight * np.abs(time_grid - t_samp)
        dx = spatial_weight * np.abs(loc_grid - x_samp)
        d = np.sqrt(dt**2 + dx**2)
        distance_grid = np.minimum(distance_grid, d)

    # Flatten for statistics
    all_distances = distance_grid.flatten()

    return {
        "mean_distance": np.mean(all_distances),
        "std_distance": np.std(all_distances),
        "max_distance": np.max(all_distances),
        "median_distance": np.median(all_distances),
        "distance_grid": distance_grid,
    }


def simulate_sampling(
    n_possible_actions: int,
    n_actions: int,
    n_time_steps: int,
    strategy: str,
    seed: int = 42,
    compute_distances: bool = True,
    spatial_weight: float = 1.0,
    temporal_weight: float = 1.0,
) -> dict:
    """Simulate sampling strategy and record inter-sample intervals.

    Args:
        n_possible_actions: Total number of possible line locations
        n_actions: Number of lines to select per time step
        n_time_steps: Number of time steps to simulate
        strategy: Either 'equispaced' or 'uniform_random'
        seed: Random seed for reproducibility
        compute_distances: Whether to compute spatiotemporal distances
        spatial_weight: Weight for spatial distance
        temporal_weight: Weight for temporal distance

    Returns:
        Dictionary with simulation results
    """
    rng = np.random.default_rng(seed)

    # Track the last time each location was sampled
    last_sampled = np.full(n_possible_actions, -1, dtype=np.int32)

    # Store all inter-sample intervals for each location
    intervals_per_location = defaultdict(list)

    # Store full sampling history for distance computation
    sampling_history = np.zeros((n_time_steps, n_possible_actions), dtype=np.float32)

    # Initialize for equispaced
    current_lines = None
    if strategy == "equispaced":
        current_lines = initial_equispaced_lines(n_actions, n_possible_actions)

    # Run simulation
    for t in range(n_time_steps):
        if strategy == "equispaced":
            if t == 0:
                selected_lines = current_lines
            else:
                selected_lines = next_equispaced_lines(current_lines)
                current_lines = selected_lines
        elif strategy == "uniform_random":
            selected_lines = random_uniform_lines(n_actions, n_possible_actions, rng)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Record in history
        sampling_history[t] = selected_lines

        # Get indices of selected lines
        selected_indices = np.where(selected_lines > 0)[0]

        # Record intervals
        for idx in selected_indices:
            if last_sampled[idx] >= 0:
                interval = t - last_sampled[idx]
                intervals_per_location[idx].append(interval)
            last_sampled[idx] = t

    # Compute statistics
    mean_intervals = np.zeros(n_possible_actions)
    std_intervals = np.zeros(n_possible_actions)
    sample_counts = np.zeros(n_possible_actions, dtype=np.int32)

    for idx in range(n_possible_actions):
        intervals = intervals_per_location[idx]
        sample_counts[idx] = len(intervals) + (1 if last_sampled[idx] >= 0 else 0)
        if len(intervals) > 0:
            mean_intervals[idx] = np.mean(intervals)
            std_intervals[idx] = np.std(intervals) if len(intervals) > 1 else 0
        else:
            mean_intervals[idx] = np.nan
            std_intervals[idx] = np.nan

    # Compute global statistics (excluding NaNs)
    valid_means = mean_intervals[~np.isnan(mean_intervals)]
    all_intervals = [v for vals in intervals_per_location.values() for v in vals]

    result = {
        "strategy": strategy,
        "n_actions": n_actions,
        "mean_intervals": mean_intervals,
        "std_intervals": std_intervals,
        "sample_counts": sample_counts,
        "global_mean_interval": np.mean(valid_means)
        if len(valid_means) > 0
        else np.nan,
        "global_std_interval": np.std(valid_means) if len(valid_means) > 0 else np.nan,
        "all_intervals": all_intervals,
        "intervals_per_location": intervals_per_location,
        "sampling_history": sampling_history,
    }

    # Compute spatiotemporal distances
    if compute_distances:
        distance_stats = compute_spatiotemporal_distances(
            sampling_history, spatial_weight, temporal_weight
        )
        result.update(
            {
                "mean_distance": distance_stats["mean_distance"],
                "std_distance": distance_stats["std_distance"],
                "max_distance": distance_stats["max_distance"],
                "median_distance": distance_stats["median_distance"],
                "distance_grid": distance_stats["distance_grid"],
            }
        )

    return result


def run_sweep_over_n_actions(
    n_possible_actions: int = 112,
    n_actions_range: list = None,
    n_time_steps: int = 100,
    seed: int = 42,
    spatial_weight: float = 1.0,
    temporal_weight: float = 1.0,
):
    """Run simulation sweep over different numbers of actions per time step.

    Args:
        n_possible_actions: Total number of possible line locations
        n_actions_range: List of n_actions values to test
        n_time_steps: Number of time steps to simulate
        seed: Random seed
        spatial_weight: Weight for spatial distance
        temporal_weight: Weight for temporal distance
    """
    if n_actions_range is None:
        n_actions_range = [
            k
            for k in range(1, n_possible_actions // 2 + 1)
            if n_possible_actions % k == 0
        ]

    print(f"Simulation Parameters:")
    print(f"  Possible locations: {n_possible_actions}")
    print(f"  N_actions values: {n_actions_range}")
    print(f"  Time steps: {n_time_steps}")
    print(f"  Spatial weight: {spatial_weight}")
    print(f"  Temporal weight: {temporal_weight}")
    print(f"  Seed: {seed}")
    print()

    # Run simulations for all n_actions values
    results_equispaced = []
    results_random = []

    for n_actions in n_actions_range:
        res_eq = simulate_sampling(
            n_possible_actions,
            n_actions,
            n_time_steps,
            "equispaced",
            seed,
            compute_distances=True,
            spatial_weight=spatial_weight,
            temporal_weight=temporal_weight,
        )
        res_rand = simulate_sampling(
            n_possible_actions,
            n_actions,
            n_time_steps,
            "uniform_random",
            seed,
            compute_distances=True,
            spatial_weight=spatial_weight,
            temporal_weight=temporal_weight,
        )
        results_equispaced.append(res_eq)
        results_random.append(res_rand)
        print(f"  Completed n_actions={n_actions}")

    # Extract summary statistics
    n_actions_arr = np.array(n_actions_range)

    eq_mean_distances = np.array([r["mean_distance"] for r in results_equispaced])
    eq_max_distances = np.array([r["max_distance"] for r in results_equispaced])
    eq_std_distances = np.array([r["std_distance"] for r in results_equispaced])
    eq_median_distances = np.array([r["median_distance"] for r in results_equispaced])

    rand_mean_distances = np.array([r["mean_distance"] for r in results_random])
    rand_max_distances = np.array([r["max_distance"] for r in results_random])
    rand_std_distances = np.array([r["std_distance"] for r in results_random])
    rand_median_distances = np.array([r["median_distance"] for r in results_random])

    # =========================================================================
    # Distance distribution heatmap as a function of K
    # =========================================================================
    print("Creating distance distribution heatmaps...")

    # Define bins for distance histogram
    max_distance = max(
        max(r["max_distance"] for r in results_equispaced),
        max(r["max_distance"] for r in results_random),
    )
    n_bins = 100
    bin_edges = np.linspace(0, max_distance, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Build 2D histogram: rows = distance bins, cols = K values
    eq_dist_heatmap = np.zeros((n_bins, len(n_actions_range)))
    rand_dist_heatmap = np.zeros((n_bins, len(n_actions_range)))

    for i, (res_eq, res_rand) in enumerate(zip(results_equispaced, results_random)):
        # Equispaced - histogram of all distances
        distances_eq = res_eq["distance_grid"].flatten()
        counts_eq, _ = np.histogram(distances_eq, bins=bin_edges, density=True)
        eq_dist_heatmap[:, i] = counts_eq

        # Random - histogram of all distances
        distances_rand = res_rand["distance_grid"].flatten()
        counts_rand, _ = np.histogram(distances_rand, bins=bin_edges, density=True)
        rand_dist_heatmap[:, i] = counts_rand

    # Find shared color scale
    vmax_shared = max(eq_dist_heatmap.max(), rand_dist_heatmap.max())

    # Clip y-axis display
    y_max = 20.0
    y_max_idx = np.searchsorted(bin_centers, y_max)

    # Create figure with two heatmaps side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Equispaced heatmap
    ax = axes[0]
    im = ax.imshow(
        eq_dist_heatmap[:y_max_idx, :],
        aspect="auto",
        origin="lower",
        extent=[0, len(n_actions_range), 0, y_max],
        cmap="Blues",
        vmin=0,
        vmax=vmax_shared,
    )
    # Overlay mean and median lines
    ax.plot(
        np.arange(len(n_actions_arr)) + 0.5,
        eq_mean_distances,
        "r-",
        linewidth=2,
        label="Mean",
    )
    ax.plot(
        np.arange(len(n_actions_arr)) + 0.5,
        eq_median_distances,
        "r--",
        linewidth=1.5,
        label="Median",
    )
    ax.set_xlabel("Lines per Time Step ($K$)")
    ax.set_ylabel("Spatiotemporal Distance")
    ax.set_xticks(np.arange(len(n_actions_range)) + 0.5)
    ax.set_xticklabels(n_actions_range)
    ax.set_title("Equispaced: Distance Distribution vs $K$")
    ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, label="Density")

    # Random heatmap
    ax = axes[1]
    im = ax.imshow(
        rand_dist_heatmap[:y_max_idx, :],
        aspect="auto",
        origin="lower",
        extent=[0, len(n_actions_range), 0, y_max],
        cmap="Oranges",
        vmin=0,
        vmax=vmax_shared,
    )
    ax.plot(
        np.arange(len(n_actions_arr)) + 0.5,
        rand_mean_distances,
        "r-",
        linewidth=2,
        label="Mean",
    )
    ax.plot(
        np.arange(len(n_actions_arr)) + 0.5,
        rand_median_distances,
        "r--",
        linewidth=1.5,
        label="Median",
    )
    ax.set_xlabel("Lines per Time Step ($K$)")
    ax.set_ylabel("Spatiotemporal Distance")
    ax.set_xticks(np.arange(len(n_actions_range)) + 0.5)
    ax.set_xticklabels(n_actions_range)
    ax.set_title("Uniform Random: Distance Distribution vs $K$")
    ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, label="Density")

    plt.tight_layout()
    plt.savefig("strategy_sim_dist_heatmap.png", dpi=300, bbox_inches="tight")
    print("Saved plot to strategy_sim_dist_heatmap.png")
    plt.close()

    # =========================================================================
    # Combined comparison heatmap (difference)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Equispaced
    ax = axes[0]
    im = ax.imshow(
        eq_dist_heatmap[:y_max_idx, :],
        aspect="auto",
        origin="lower",
        extent=[0, len(n_actions_range), 0, y_max],
        cmap="Blues",
        vmin=0,
        vmax=vmax_shared,
    )
    ax.set_xlabel("Lines per Time Step ($K$)")
    ax.set_ylabel("Spatiotemporal Distance")
    ax.set_title("Equispaced")
    ax.set_xticks(np.arange(len(n_actions_range)) + 0.5)
    ax.set_xticklabels(n_actions_range)
    plt.colorbar(im, ax=ax, label="Density")

    # Random
    ax = axes[1]
    im = ax.imshow(
        rand_dist_heatmap[:y_max_idx, :],
        aspect="auto",
        origin="lower",
        extent=[0, len(n_actions_range), 0, y_max],
        cmap="Oranges",
        vmin=0,
        vmax=vmax_shared,
    )
    ax.set_xlabel("Lines per Time Step ($K$)")
    ax.set_ylabel("Spatiotemporal Distance")
    ax.set_title("Uniform Random")
    ax.set_xticks(np.arange(len(n_actions_range)) + 0.5)
    ax.set_xticklabels(n_actions_range)
    plt.colorbar(im, ax=ax, label="Density")

    # Difference (Random - Equispaced)
    ax = axes[2]
    diff = rand_dist_heatmap[:y_max_idx, :] - eq_dist_heatmap[:y_max_idx, :]
    vabs = np.abs(diff).max()
    im = ax.imshow(
        diff,
        aspect="auto",
        origin="lower",
        extent=[0, len(n_actions_range), 0, y_max],
        cmap="RdBu_r",
        vmin=-vabs,
        vmax=vabs,
        # interpolation="bilinear",
    )
    ax.set_xlabel("Lines per Time Step ($K$)")
    ax.set_ylabel("Spatiotemporal Distance")
    ax.set_title("Difference (Random - Equispaced)")
    ax.set_xticks(np.arange(len(n_actions_range)) + 0.5)
    ax.set_xticklabels(n_actions_range)
    plt.colorbar(im, ax=ax, label="Density Difference")

    plt.tight_layout()
    plt.savefig(
        "strategy_sim_dist_heatmap_comparison.png", dpi=300, bbox_inches="tight"
    )
    print("Saved plot to strategy_sim_dist_heatmap_comparison.png")
    plt.close()

    # =========================================================================
    # Sampling patterns over time for each K
    # =========================================================================
    print("Creating sampling pattern visualizations...")

    n_k_values = len(n_actions_range)
    fig, axes = plt.subplots(2, n_k_values, figsize=(3 * n_k_values, 8))

    # Limit time steps shown for clarity
    t_max_display = min(50, n_time_steps)

    for i, (res_eq, res_rand, n_actions) in enumerate(
        zip(results_equispaced, results_random, n_actions_range)
    ):
        # Equispaced pattern (top row)
        ax = axes[0, i] if n_k_values > 1 else axes[0]
        history_eq = res_eq["sampling_history"][:t_max_display, :]
        ax.imshow(
            history_eq,
            aspect="auto",
            origin="upper",
            cmap="Blues",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_xlabel("Scan Line Location")
        if i == 0:
            ax.set_ylabel("Time Step")
        ax.set_title(f"Equispaced, K={n_actions}")

        # Random pattern (bottom row)
        ax = axes[1, i] if n_k_values > 1 else axes[1]
        history_rand = res_rand["sampling_history"][:t_max_display, :]
        ax.imshow(
            history_rand,
            aspect="auto",
            origin="upper",
            cmap="Oranges",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_xlabel("Scan Line Location")
        if i == 0:
            ax.set_ylabel("Time Step")
        ax.set_title(f"Random, K={n_actions}")

    plt.tight_layout()
    plt.savefig("strategy_sim_sampling_patterns.png", dpi=300, bbox_inches="tight")
    print("Saved plot to strategy_sim_sampling_patterns.png")
    plt.close()

    # =========================================================================
    # Distance grids for each K (distance to nearest sample)
    # =========================================================================
    print("Creating distance grid visualizations...")

    fig, axes = plt.subplots(2, n_k_values, figsize=(3 * n_k_values, 8))

    for i, (res_eq, res_rand, n_actions) in enumerate(
        zip(results_equispaced, results_random, n_actions_range)
    ):
        # Find shared color scale for this K (across equispaced and random)
        dist_eq = res_eq["distance_grid"][:t_max_display, :]
        dist_rand = res_rand["distance_grid"][:t_max_display, :]
        vmax_k = max(dist_eq.max(), dist_rand.max())

        # Equispaced distance grid (top row)
        ax = axes[0, i] if n_k_values > 1 else axes[0]
        im_eq = ax.imshow(
            dist_eq,
            aspect="auto",
            origin="upper",
            # cmap="viridis",
            vmin=0,
            vmax=vmax_k,
            interpolation="nearest",
        )
        ax.set_xlabel("Scan Line Location")
        if i == 0:
            ax.set_ylabel("Time Step")
        ax.set_title(f"Equispaced, K={n_actions}")
        plt.colorbar(im_eq, ax=ax, label="Distance")

        # Random distance grid (bottom row)
        ax = axes[1, i] if n_k_values > 1 else axes[1]
        im_rand = ax.imshow(
            dist_rand,
            aspect="auto",
            origin="upper",
            # cmap="viridis",
            vmin=0,
            vmax=vmax_k,
            interpolation="nearest",
        )
        ax.set_xlabel("Scan Line Location")
        if i == 0:
            ax.set_ylabel("Time Step")
        ax.set_title(f"Random, K={n_actions}")
        plt.colorbar(im_rand, ax=ax, label="Distance")

    plt.tight_layout()
    plt.savefig("strategy_sim_distance_grids.png", dpi=300, bbox_inches="tight")
    print("Saved plot to strategy_sim_distance_grids.png")
    plt.close()

    ratio = rand_mean_distances / eq_mean_distances

    # =========================================================================
    # Print summary table
    # =========================================================================
    print("\n" + "=" * 100)
    print("Summary Table: Spatiotemporal Distances")
    print("=" * 100)
    print(
        f"{'K':>4} | {'Eq Mean':>10} | {'Eq Max':>10} | {'Eq Std':>10} | "
        f"{'Rand Mean':>10} | {'Rand Max':>10} | {'Rand Std':>10} | {'Ratio':>8}"
    )
    print("-" * 100)
    for i, n_actions in enumerate(n_actions_range):
        print(
            f"{n_actions:>4} | "
            f"{eq_mean_distances[i]:>10.2f} | "
            f"{eq_max_distances[i]:>10.2f} | "
            f"{eq_std_distances[i]:>10.2f} | "
            f"{rand_mean_distances[i]:>10.2f} | "
            f"{rand_max_distances[i]:>10.2f} | "
            f"{rand_std_distances[i]:>10.2f} | "
            f"{ratio[i]:>8.2f}"
        )

    return results_equispaced, results_random


def run_simulation_and_plot(
    n_possible_actions: int = 112,
    n_actions: int = 8,
    n_time_steps: int = 1000,
    seed: int = 42,
):
    """Run simulation for both strategies and create visualization.

    Args:
        n_possible_actions: Total number of possible line locations
        n_actions: Number of lines to select per time step
        n_time_steps: Number of time steps to simulate
        seed: Random seed
    """
    print(f"Simulation Parameters:")
    print(f"  Possible locations: {n_possible_actions}")
    print(f"  Lines per time step: {n_actions}")
    print(f"  Time steps: {n_time_steps}")
    print(f"  Seed: {seed}")
    print()

    # Run simulations
    results_equispaced = simulate_sampling(
        n_possible_actions, n_actions, n_time_steps, "equispaced", seed
    )
    results_random = simulate_sampling(
        n_possible_actions, n_actions, n_time_steps, "uniform_random", seed
    )

    # Print summary statistics
    print("=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    for results in [results_equispaced, results_random]:
        print(f"\n{results['strategy'].upper()}:")
        print(f"  Global mean interval: {results['global_mean_interval']:.2f}")
        print(
            f"  Std of mean intervals across locations: {results['global_std_interval']:.2f}"
        )
        print(f"  Min mean interval: {np.nanmin(results['mean_intervals']):.2f}")
        print(f"  Max mean interval: {np.nanmax(results['mean_intervals']):.2f}")
        print(f"  Total samples: {np.sum(results['sample_counts'])}")

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Top row: Equispaced
    # Mean interval per location
    ax = axes[0, 0]
    ax.bar(
        range(n_possible_actions),
        results_equispaced["mean_intervals"],
        alpha=0.7,
        color="tab:blue",
    )
    ax.axhline(
        results_equispaced["global_mean_interval"],
        color="red",
        linestyle="--",
        label=f"Global mean: {results_equispaced['global_mean_interval']:.2f}",
    )
    ax.set_xlabel("Line Location")
    ax.set_ylabel("Mean Inter-sample Interval")
    ax.set_title("Equispaced: Mean Interval per Location")
    ax.legend()
    ax.set_xlim(-1, n_possible_actions)

    # Histogram of all intervals
    ax = axes[0, 1]
    all_intervals_eq = [
        v for vals in results_equispaced["all_intervals"].values() for v in vals
    ]
    if all_intervals_eq:
        ax.hist(
            all_intervals_eq, bins=30, alpha=0.7, color="tab:blue", edgecolor="black"
        )
    ax.set_xlabel("Inter-sample Interval")
    ax.set_ylabel("Frequency")
    ax.set_title("Equispaced: Distribution of Intervals")

    # Sample count per location
    ax = axes[0, 2]
    ax.bar(
        range(n_possible_actions),
        results_equispaced["sample_counts"],
        alpha=0.7,
        color="tab:blue",
    )
    ax.set_xlabel("Line Location")
    ax.set_ylabel("Sample Count")
    ax.set_title("Equispaced: Samples per Location")
    ax.set_xlim(-1, n_possible_actions)

    # Bottom row: Uniform Random
    # Mean interval per location
    ax = axes[1, 0]
    ax.bar(
        range(n_possible_actions),
        results_random["mean_intervals"],
        alpha=0.7,
        color="tab:orange",
    )
    ax.axhline(
        results_random["global_mean_interval"],
        color="red",
        linestyle="--",
        label=f"Global mean: {results_random['global_mean_interval']:.2f}",
    )
    ax.set_xlabel("Line Location")
    ax.set_ylabel("Mean Inter-sample Interval")
    ax.set_title("Uniform Random: Mean Interval per Location")
    ax.legend()
    ax.set_xlim(-1, n_possible_actions)

    # Histogram of all intervals
    ax = axes[1, 1]
    all_intervals_rand = [
        v for vals in results_random["all_intervals"].values() for v in vals
    ]
    if all_intervals_rand:
        ax.hist(
            all_intervals_rand,
            bins=30,
            alpha=0.7,
            color="tab:orange",
            edgecolor="black",
        )
    ax.set_xlabel("Inter-sample Interval")
    ax.set_ylabel("Frequency")
    ax.set_title("Uniform Random: Distribution of Intervals")

    # Sample count per location
    ax = axes[1, 2]
    ax.bar(
        range(n_possible_actions),
        results_random["sample_counts"],
        alpha=0.7,
        color="tab:orange",
    )
    ax.set_xlabel("Line Location")
    ax.set_ylabel("Sample Count")
    ax.set_title("Uniform Random: Samples per Location")
    ax.set_xlim(-1, n_possible_actions)

    plt.tight_layout()
    plt.savefig("scan_line_sim_results.png", dpi=300, bbox_inches="tight")
    print("\nSaved plot to scan_line_sim_results.png")
    plt.show()

    # Additional comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Side-by-side mean intervals
    ax = axes[0]
    width = 0.4
    x = np.arange(n_possible_actions)
    ax.bar(
        x - width / 2,
        results_equispaced["mean_intervals"],
        width,
        label="Equispaced",
        alpha=0.7,
        color="tab:blue",
    )
    ax.bar(
        x + width / 2,
        results_random["mean_intervals"],
        width,
        label="Uniform Random",
        alpha=0.7,
        color="tab:orange",
    )
    ax.set_xlabel("Line Location")
    ax.set_ylabel("Mean Inter-sample Interval")
    ax.set_title("Comparison: Mean Interval per Location")
    ax.legend()
    ax.set_xlim(-1, n_possible_actions)

    # Box plot comparison
    ax = axes[1]
    data_to_plot = [all_intervals_eq, all_intervals_rand]
    bp = ax.boxplot(
        data_to_plot, labels=["Equispaced", "Uniform Random"], patch_artist=True
    )
    bp["boxes"][0].set_facecolor("tab:blue")
    bp["boxes"][1].set_facecolor("tab:orange")
    for box in bp["boxes"]:
        box.set_alpha(0.7)
    ax.set_ylabel("Inter-sample Interval")
    ax.set_title("Comparison: Distribution of All Intervals")

    plt.tight_layout()
    plt.savefig("scan_line_sim_comparison.png", dpi=300, bbox_inches="tight")
    print("Saved comparison plot to scan_line_sim_comparison.png")
    plt.show()

    return results_equispaced, results_random


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scan line sampling simulation")
    parser.add_argument(
        "--n-possible-actions",
        type=int,
        default=112,
        help="Total number of possible line locations (default: 112)",
    )
    parser.add_argument(
        "--n-actions",
        type=int,
        default=None,
        help="Number of lines per time step. If not set, runs sweep from 1 to N/2",
    )
    parser.add_argument(
        "--n-time-steps",
        type=int,
        default=224,
        help="Number of time steps to simulate (default: 100)",
    )
    parser.add_argument(
        "--spatial-weight",
        type=float,
        default=1.0,
        help="Weight for spatial distance (default: 1.0)",
    )
    parser.add_argument(
        "--temporal-weight",
        type=float,
        default=1.0,
        help="Weight for temporal distance (default: 1.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    if args.n_actions is None:
        # Run sweep over n_actions
        run_sweep_over_n_actions(
            n_possible_actions=args.n_possible_actions,
            n_time_steps=args.n_time_steps,
            seed=args.seed,
            spatial_weight=args.spatial_weight,
            temporal_weight=args.temporal_weight,
        )
    else:
        # Run single simulation
        run_simulation_and_plot(
            n_possible_actions=args.n_possible_actions,
            n_actions=args.n_actions,
            n_time_steps=args.n_time_steps,
            seed=args.seed,
        )
