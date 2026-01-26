from pathlib import Path

import jax.numpy as jnp
import keras
import numpy as np
from skimage.exposure import match_histograms

import zea
from ulsa.entropy import pixelwise_entropy
from ulsa.io_utils import color_to_value, get_heatmap, postprocess_agent_results
from ulsa.utils import find_best_cine_loop


def load_from_run_dir(
    run_dir: Path | str,
    frame_idx=None,
    selection_strategy="greedy_entropy",
    scan_convert_resolution=0.1,
    dynamic_range=None,
    fill_value="transparent",
    distance_to_apex=7.0,
    no_measurement_color="gray",
    drop_first_n_frames: int = 0,
):
    """Load and postprocess results from a run directory (in-house dataset)."""
    if drop_first_n_frames > 0:
        assert frame_idx is None, (
            "Cannot drop frames when a specific frame is selected."
        )

    run_dir = Path(run_dir)

    focused_results = np.load(run_dir / "focused.npz", allow_pickle=True)
    diverging_results = np.load(run_dir / "diverging.npz", allow_pickle=True)
    results = np.load(run_dir / f"{selection_strategy}.npz", allow_pickle=True)
    n_actions = results["n_actions"].item()
    n_possible_actions = results["n_possible_actions"].item()

    # Load into variables
    focused = focused_results["reconstructions"]
    diverging = diverging_results["reconstructions"]
    reconstructions = results["reconstructions"]
    measurements = results["measurements"]
    masks = results["masks"]
    belief_distributions = results["belief_distributions"]

    io_config = zea.Config(
        scan_convert=True,
        scan_conversion_angles=np.rad2deg(results["theta_range"]),
    )

    heatmap = get_heatmap(
        masks,
        io_config,
        window_size=7,
        resolution=scan_convert_resolution,
        distance_to_apex=distance_to_apex,
    )

    # Drop to single frame if selected
    if frame_idx is not None:
        focused = focused[frame_idx, None]
        diverging = diverging[frame_idx, None]
        reconstructions = reconstructions[frame_idx, None]
        measurements = measurements[frame_idx, None]
        masks = masks[frame_idx, None]
        belief_distributions = belief_distributions[frame_idx, None]
        heatmap = heatmap[frame_idx, None]
        frame_idx = 0

    # histogram match diverging to focused
    match_histograms_vectorized = np.vectorize(
        match_histograms, signature="(n,m),(n,m)->(n,m)"
    )
    diverging = match_histograms_vectorized(diverging, focused)

    # histogram match reconstructions to focused
    reconstructions = match_histograms_vectorized(reconstructions, focused)
    reconstruction_range = results["dynamic_range"]

    if dynamic_range is None:
        dynamic_range = focused_results["dynamic_range"]

    measurements = keras.ops.where(
        masks > 0,
        measurements,
        color_to_value(reconstruction_range, no_measurement_color),
    )

    _, height, width = focused.shape
    coordinates, _ = zea.display.compute_scan_convert_2d_coordinates(
        (height, width),
        (0, height),
        results["theta_range"],
        scan_convert_resolution,
        dtype=focused.dtype,
        distance_to_apex=distance_to_apex,
    )

    # Find best cine loop
    if frame_idx is None:
        last_frame = find_best_cine_loop(focused[drop_first_n_frames:], visualize=True)
        last_frame = last_frame + drop_first_n_frames
        focused = focused[:last_frame]
        diverging = diverging[:last_frame]
        reconstructions = reconstructions[:last_frame]
        measurements = measurements[:last_frame]
        belief_distributions = belief_distributions[:last_frame]
        heatmap = heatmap[:last_frame]

    print("Postprocessing focused...")
    focused = postprocess_agent_results(
        focused,
        io_config,
        scan_convert_order=0,
        image_range=dynamic_range,
        drop_first_n_frames=drop_first_n_frames,
        fill_value=fill_value,
        scan_convert_resolution=scan_convert_resolution,
        distance_to_apex=distance_to_apex,
        coordinates=coordinates,
    )
    print("Postprocessing diverging...")
    diverging = postprocess_agent_results(
        diverging,
        io_config,
        scan_convert_order=0,
        image_range=dynamic_range,
        drop_first_n_frames=drop_first_n_frames,
        fill_value=fill_value,
        scan_convert_resolution=scan_convert_resolution,
        distance_to_apex=distance_to_apex,
        coordinates=coordinates,
    )
    print("Postprocessing reconstructions...")
    reconstructions = postprocess_agent_results(
        reconstructions,
        io_config,
        scan_convert_order=0,
        image_range=dynamic_range,
        drop_first_n_frames=drop_first_n_frames,
        reconstruction_sharpness_std=0.02,
        fill_value=fill_value,
        scan_convert_resolution=scan_convert_resolution,
        distance_to_apex=distance_to_apex,
        coordinates=coordinates,
    )
    print("Postprocessing measurements...")
    measurements = postprocess_agent_results(
        measurements,
        io_config,
        scan_convert_order=0,
        image_range=reconstruction_range,
        drop_first_n_frames=drop_first_n_frames,
        fill_value=fill_value,
        scan_convert_resolution=scan_convert_resolution,
        distance_to_apex=distance_to_apex,
        coordinates=coordinates,
    )

    print("Postprocessing entropy...")
    entropy = pixelwise_entropy(belief_distributions, entropy_sigma=255)
    entropy = postprocess_agent_results(
        entropy,
        io_config=io_config,
        scan_convert_order=1,
        image_range=[0, jnp.nanpercentile(entropy, 98.5)],
        drop_first_n_frames=drop_first_n_frames,
        fill_value=fill_value,
        scan_convert_resolution=scan_convert_resolution,
        distance_to_apex=distance_to_apex,
        coordinates=coordinates,
    )
    entropy = np.squeeze(entropy)

    return (
        focused,
        diverging,
        reconstructions,
        measurements,
        entropy,
        n_actions,
        n_possible_actions,
        heatmap,
        frame_idx,
    )
