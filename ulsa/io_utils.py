import os
from pathlib import Path

import cv2
import keras
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from keras import ops
from matplotlib.animation import FuncAnimation

import zea
from zea import log
from zea.display import scan_convert_2d
from zea.visualize import plot_image_grid


def make_save_dir(path, prefix="run"):
    """
    Make a save dir with the current datetime as an ID
    """
    datestr = zea.utils.get_date_string("%Y_%m_%d_%H%M%S_%f")
    run_id = f"{prefix}_{datestr}"
    save_dir = path / Path(run_id)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir, run_id


def map_range(img, from_range=(-1, 1), to_range=(0, 255)):
    img = ops.convert_to_numpy(img)
    img = zea.func.translate(img, from_range, to_range)
    return np.clip(img, to_range[0], to_range[1])


def deg2rad(x: float):
    return x * (np.pi / 180)


def _scan_convert(
    img,
    scan_conversion_angles: tuple = (-45, 45),
    fill_value: float = 0.0,
    order: int = 1,
    resolution: float | None = 1.0,
    **kwargs,
):
    """Scan conversion helper function (will handle casting to float32 if needed). If possible,
    stay with floats and cast to int to avoid unnessesary quantization in between."""
    img_height = img.shape[-2]
    start_angle, end_angle = scan_conversion_angles

    orig_dtype = ops.dtype(img)
    int_type = "int" in orig_dtype
    if int_type:
        img = ops.cast(img, "float32")

    sc, _ = scan_convert_2d(
        img,
        rho_range=(0, img_height),
        theta_range=(deg2rad(start_angle), deg2rad(end_angle)),
        order=order,
        fill_value=fill_value,
        resolution=resolution,
        **kwargs,
    )

    # round sc to get rid of numerical errors leading to overflow
    if int_type:
        sc = ops.cast(ops.round(sc), orig_dtype)

    return sc


def gray_to_color_with_transparency(grayscale_image, transparency_mask=None):
    transparency_mask = (
        transparency_mask
        if transparency_mask is not None
        else np.ones_like(grayscale_image)
    )
    grayscale_image = ops.convert_to_numpy(grayscale_image)
    colour_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGRA)
    colour_image[:, :, 3] = colour_image[:, :, 3] * transparency_mask
    return colour_image


map_error = lambda x: map_range(x, (0, 2), (0, 1))


def apply_colormap_to_rgba(rgba_image: np.ndarray, cmap_name: str = "viridis"):
    """
    Apply a colormap to an RGBA image based on its grayscale values while preserving transparency.

    Parameters:
    - rgba_image (np.ndarray): Input image of shape (H, W, 4) where RGB channels define grayscale values.
    - cmap_name (str): Name of the colormap to apply (default is 'viridis').

    Returns:
    - np.ndarray: RGBA image with the colormap applied and transparency preserved.
    """

    if rgba_image.shape[-1] != 4:
        raise ValueError("Input image must have shape (H, W, 4) with an alpha channel.")

    # Convert RGB to grayscale (luminosity method: best perceptual results)
    gray = np.dot(rgba_image[..., :3], [0.2989, 0.5870, 0.1140])  # Weighted sum

    # Normalize grayscale values to range [0, 1]
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)  # Avoid div by zero

    # Get the specified colormap
    cmap = cm.get_cmap(cmap_name)

    # Apply colormap (returns RGBA image with shape (H, W, 4))
    color_mapped = cmap(gray)

    # Preserve the original alpha channel
    color_mapped[..., 3] = rgba_image[..., 3]  # Copy alpha values

    return color_mapped


def mask_heatmap_moving_average(masks, window_size=9):
    masks = ops.convert_to_numpy(masks).astype(np.float32)
    heatmap = np.zeros_like(masks)
    for i, mask in enumerate(masks):
        for j in range(window_size):
            if i + j < len(masks):
                heatmap[i + j] += mask
    return heatmap / window_size


def side_by_side_gif(
    save_path,
    *arrays,
    vmin=0,
    vmax=255,
    fps=30,
    interpolation="nearest",
    dpi=300,
    context=None,
    labels=None,
):
    """
    Generalized side-by-side gif for N arrays.
    Usage: side_by_side_gif(save_path, arr1, arr2, arr3, ..., labels=[...])
    """
    if context is None:
        context = {}

    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array must be provided.")

    num_frames = arrays[0].shape[0]
    for arr in arrays:
        if arr.shape[0] != num_frames:
            raise ValueError("All arrays must have the same number of frames.")

    if not isinstance(interpolation, (tuple, list)):
        interpolation = [interpolation] * n_arrays
    if labels is None:
        labels = [None] * n_arrays

    with plt.style.context(context):
        fig, axes = plt.subplots(
            1, n_arrays, figsize=(4 * n_arrays, 4), layout="constrained"
        )
        if n_arrays == 1:
            axes = [axes]
        ims = []
        for i, arr in enumerate(arrays):
            im = axes[i].imshow(
                arr[0],
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                interpolation=interpolation[i],
                animated=True,
            )
            if labels[i] is not None:
                axes[i].set_title(labels[i])
            axes[i].axis("off")
            ims.append(im)

        def update(frame):
            for i, arr in enumerate(arrays):
                ims[i].set_data(arr[frame])
            return ims

        writer = "pillow" if str(save_path).endswith(".gif") else "ffmpeg"
        anim = FuncAnimation(fig, update, frames=num_frames, blit=True)
        anim.save(save_path, writer=writer, fps=fps, dpi=dpi)
        plt.close(fig)
        print(f"Saved animation to {save_path}")


def get_heatmap(
    masks,
    io_config,
    sigma=1.0,
    resolution=0.1,
    normalize=True,
    cmap="inferno",
    sc_order=1,
    window_size=9,
):
    from scipy.ndimage import gaussian_filter

    heatmap = mask_heatmap_moving_average(masks, window_size=window_size)

    # Smooth
    if sigma is not None:
        # axis=-1 is the width axis
        heatmap = gaussian_filter(heatmap, sigma=sigma, axes=-1)

    # Scan convert
    if io_config.scan_convert:
        heatmap = _scan_convert(
            heatmap,
            io_config.scan_conversion_angles,
            order=sc_order,
            fill_value=np.nan,
            resolution=resolution,
        )

    if normalize:
        heatmap /= np.nanmax(heatmap)

    # Cmap
    cmap = plt.colormaps.get_cmap(cmap)
    heatmap = cmap(heatmap).astype(np.float32)
    heatmap = map_range(heatmap, from_range=(0, 1)).astype(np.uint8)
    return heatmap


def first_frames_for_slides(
    save_dir,
    targets,
    masks,
    measurements,
    io_config,
    dpi=150,
    scan_convert_order=0,  # fixed to 0 for measurements!
    scan_convert_resolution=0.1,
    interpolation_matplotlib="nearest",
    context="styles/nvmu.mplstyle",
    postfix_filename=None,
):
    # Save first target frame as PNG
    plt.figure(figsize=(6, 6), dpi=dpi)
    with plt.style.context(context):
        plt.imshow(targets[0], cmap="gray", interpolation=interpolation_matplotlib)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            save_dir / f"first_target{postfix_filename}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()
        log.info(log.yellow(save_dir / f"first_target{postfix_filename}.png"))

    plt.figure(figsize=(6, 6), dpi=dpi)
    with plt.style.context(context):
        plt.imshow(measurements[0], cmap="gray", interpolation=interpolation_matplotlib)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            save_dir / f"first_measurement{postfix_filename}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()
        log.info(log.yellow(save_dir / f"first_measurement{postfix_filename}.png"))

    # Save first mask as black/white PNG
    first_mask = masks[0]
    if io_config.scan_convert:
        first_mask_sc = _scan_convert(
            first_mask.astype(np.float32),
            io_config.scan_conversion_angles,
            order=scan_convert_order,
            fill_value=0.0,
            resolution=scan_convert_resolution,
        )
    else:
        first_mask_sc = first_mask

    # Convert mask to 0 (black) and 255 (white)
    first_mask_img = (first_mask_sc > 0.5).astype(np.uint8) * 255

    plt.figure(figsize=(6, 6), dpi=dpi)
    with plt.style.context(context):
        plt.imshow(
            first_mask_img, cmap="gray", interpolation="nearest", vmin=0, vmax=255
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            save_dir / f"first_mask{postfix_filename}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        log.info(log.yellow(save_dir / f"first_mask{postfix_filename}.png"))
        plt.close()


def gaussian_sharpness(data, std=0.025, image_range=(-1, 1)):
    """Apply Gaussian noise to the data to simulate sharpness."""
    if std > 0:
        total_image_range = image_range[1] - image_range[0]
        std *= total_image_range
        noised_data = data + keras.random.normal(
            data.shape,
            stddev=std,
            dtype=data.dtype,
        )
        data = ops.where(data > image_range[0], noised_data, data)
    return data


def color_to_value(image_range, color="gray"):
    color = color.lower()

    if color == "white":
        return image_range[1]
    elif color == "black":
        return image_range[0]
    elif color == "gray":
        return (image_range[0] + image_range[1]) / 2
    elif color == "transparent":
        return np.nan
    else:
        raise ValueError(
            f"Unknown color of type str: {color}. Use 'black', 'white', 'gray', or 'transparent'."
        )


def postprocess_agent_results(
    data,
    io_config,
    scan_convert_order,
    image_range,
    drop_first_n_frames=0,
    scan_convert_resolution=0.1,
    reconstruction_sharpness_std=0.0,  # advise: 0.025
    fill_value="black",
    to_uint8=True,
):
    """Postprocess agent results for visualization.
    Always return images in range [0, 255]."""
    # Cast to float32 because scan conversion is weird for float16
    data = ops.cast(data, "float32")

    # Drop first n frames
    if drop_first_n_frames > 0 and drop_first_n_frames < len(data):
        data = data[drop_first_n_frames:]

    # Add some noise (mainly for reconstructions)
    # Scaled based on the image range
    data = gaussian_sharpness(data, reconstruction_sharpness_std, image_range)

    if isinstance(fill_value, str):
        to_uint8 = fill_value.lower() != "transparent"
        fill_value = color_to_value(image_range, fill_value)

    if io_config.scan_convert:
        data = _scan_convert(
            data,
            io_config.scan_conversion_angles,
            order=scan_convert_order,
            fill_value=fill_value,
            resolution=scan_convert_resolution,
        )

    # To uint8
    data = map_range(data, image_range, (0, 255))
    if to_uint8:
        data = data.astype(np.uint8)

    return data


def postprocess_heatmap(
    heatmap,
    io_config,
    scan_convert_order=0,
    scan_convert_resolution=0.1,
    cmap="viridis",
):
    heatmap = _scan_convert(
        heatmap,
        io_config.scan_conversion_angles,
        order=scan_convert_order,
        fill_value=np.nan,
        resolution=scan_convert_resolution,
    )
    heatmap_min = np.nanmin(heatmap)
    heatmap_max = np.nanmax(heatmap)
    heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    cmap = plt.colormaps.get_cmap(cmap)
    heatmap = cmap(heatmap).astype(np.float32)
    heatmap = map_range(heatmap, from_range=(0, 1)).astype(np.uint8)
    return heatmap


def plot_frames_for_presentation(
    save_dir,
    targets,  # shape (frames, height, width)
    reconstructions,  # shape (frames, height, width)
    masks,  # shape (frames, height, width)
    measurements,  # shape (frames, height, width)
    io_config,
    dpi=150,
    scan_convert_order=0,  # fixed to 0 for measurements!
    scan_convert_resolution=0.1,
    interpolation_matplotlib="nearest",
    image_range=(0, 255),
    context="styles/nvmu.mplstyle",
    drop_first_n_frames=2,
    window_size=7,
    postfix_filename=None,
    sigma_heatmap=None,
    file_type="gif",  # 'gif' or 'mp4'
    fill_value="black",  # 'black', 'white', 'gray', 'transparent'
    no_measurement_color="gray",
):
    log.info("Plotting frames for presentation, this may take a while...")
    save_dir = Path(save_dir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    targets = postprocess_agent_results(
        targets,
        io_config,
        scan_convert_order,
        image_range,
        drop_first_n_frames,
        scan_convert_resolution,
        fill_value=fill_value,
    )
    reconstructions = postprocess_agent_results(
        reconstructions,
        io_config,
        scan_convert_order,
        image_range,
        drop_first_n_frames,
        scan_convert_resolution,
        reconstruction_sharpness_std=io_config.get("reconstruction_sharpness_std", 0.0),
        fill_value=fill_value,
    )
    measurements = keras.ops.where(
        masks > 0, measurements, color_to_value(image_range, no_measurement_color)
    )
    measurements = postprocess_agent_results(
        measurements,
        io_config,
        scan_convert_order=0,  # always 0 for masks!
        drop_first_n_frames=drop_first_n_frames,
        image_range=image_range,
        scan_convert_resolution=scan_convert_resolution,
        fill_value=fill_value,
    )

    if postfix_filename is None:
        postfix_filename = ""
    else:
        postfix_filename = "_" + postfix_filename

    first_frames_for_slides(
        save_dir,
        targets,
        masks,
        measurements,
        io_config,
        dpi,
        scan_convert_order,
        scan_convert_resolution,
        interpolation_matplotlib,
        context,
        postfix_filename,
    )

    # Target and reconstruction side by side
    side_by_side_gif(
        save_dir / f"target_reconstruction{postfix_filename}.{file_type}",
        targets,
        reconstructions,
        dpi=dpi,
        interpolation=interpolation_matplotlib,
        fps=io_config.gif_fps,
        context=context,
        labels=["Target", "Reconstruction"],
    )

    # Measurements and reconstruction side by side
    side_by_side_gif(
        save_dir / f"measurements_reconstruction{postfix_filename}.{file_type}",
        measurements,
        reconstructions,
        dpi=dpi,
        interpolation=interpolation_matplotlib,
        fps=io_config.gif_fps,
        context=context,
        labels=["Measurements", "Reconstruction"],
    )

    # Action heatmap and reconstruction side by side
    heatmap = get_heatmap(
        masks, io_config, cmap="inferno", window_size=window_size, sigma=sigma_heatmap
    )
    offset = max(window_size, drop_first_n_frames)
    if offset > len(heatmap):
        offset = 0
        log.warning(
            f"Heatmap sequence is not long enough to cut of {offset} frames. Setting to 0."
        )
    if offset > drop_first_n_frames:
        drop_extra_frames = offset - drop_first_n_frames
    side_by_side_gif(
        save_dir / f"heatmap_reconstruction{postfix_filename}.{file_type}",
        heatmap[offset:],
        reconstructions[drop_extra_frames:],
        dpi=dpi,
        interpolation=[None, interpolation_matplotlib],
        fps=io_config.gif_fps,
        context=context,
        labels=[f"Density over {window_size} frames", "Reconstruction"],
    )


def plot_downstream_task_beliefs(
    save_dir,
    belief_distribution,
    beliefs_dst,
    downstream_task,
    target,
    target_dst,
    io_config,
    frame_idx=0,
    dpi=150,
    interpolation_matplotlib="nearest",
    context="styles/darkmode.mplstyle",
):
    """
    Plots a row with:
      - Left: target_with_mask
      - Middle: grid of beliefs_with_mask (e.g. 2x2)
      - Right: mask agreement (sum of beliefs_dst masks)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    target_with_mask = downstream_task.postprocess_for_visualization(
        target[None, ...], target_dst[None, ...]
    )
    beliefs_with_mask = downstream_task.postprocess_for_visualization(
        belief_distribution, beliefs_dst
    )

    # Prepare mask agreement: sum of beliefs_dst masks
    mask_agreement = ops.sum(beliefs_dst, axis=0)

    # Prepare grid of beliefs_with_mask (e.g. 2x2)
    n_beliefs = len(beliefs_with_mask)
    grid_ncols = int(np.ceil(np.sqrt(n_beliefs)))
    grid_nrows = int(np.ceil(n_beliefs / grid_ncols))

    with plt.style.context(context):
        fig = plt.figure(
            figsize=(4 + 6 + 2, 6), dpi=dpi
        )  # wider grid, smaller mask agreement
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 2, 0.7], wspace=0.0)
        # Left: target_with_mask
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(
            np.squeeze(target_with_mask),
            cmap="gray",
            interpolation=interpolation_matplotlib,
        )
        ax0.set_title("Target", fontsize=12)
        ax0.axis("off")

        # Middle: grid of beliefs_with_mask
        grid_gs = gridspec.GridSpecFromSubplotSpec(
            grid_nrows, grid_ncols, subplot_spec=gs[0, 1], wspace=0.0, hspace=0.0
        )
        for idx, belief_img in enumerate(beliefs_with_mask):
            row = idx // grid_ncols
            col = idx % grid_ncols
            ax = fig.add_subplot(grid_gs[row, col])
            ax.imshow(
                np.squeeze(belief_img),
                cmap="gray",
                interpolation=interpolation_matplotlib,
            )
            ax.axis("off")
        # Add a title above the grid
        fig.text(
            0.525,
            0.9,
            "Beliefs",
            ha="center",
            va="top",
            fontsize=12,
        )

        # Right: mask agreement (smaller)
        ax2 = fig.add_subplot(gs[0, 2])
        im = ax2.imshow(
            mask_agreement, cmap="viridis", interpolation=interpolation_matplotlib
        )
        ax2.set_title("Mask Agreement", fontsize=12)
        ax2.axis("off")
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, shrink=0.7)

        save_path = save_dir / f"downstream_task_beliefs_{frame_idx}.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        log.info(log.yellow(f"Saved downstream task beliefs plot to {save_path}"))


def plot_downstream_task_output_for_presentation(
    save_dir,
    targets,  # shape (num_frames, H, W)
    measurements,  # shape (num_frames, H, W)
    reconstructions,  # shape (num_frames, H, W)
    posterior_std,  # shape (num_frames, H, W)
    downstream_task,
    reconstructions_dst,
    targets_dst,
    saliency_maps,  # shape (num_frames, H, W) or (num_frames, H, W, 1)
    io_config,
    dpi=150,
    scan_convert_order=0,
    scan_convert_resolution=0.1,
    interpolation_matplotlib="nearest",
    image_range=(-1, 1),
    context="styles/darkmode.mplstyle",
    gif_name="downstream_task_output.gif",
    drop_first_n_frames=0,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    num_frames = targets.shape[0]
    # expects RGB image output
    targets_with_mask = downstream_task.postprocess_for_visualization(
        targets, targets_dst
    )
    reconstructions_with_mask = downstream_task.postprocess_for_visualization(
        reconstructions, reconstructions_dst
    )

    # TODO: maybe check if DST output is same size as measurements etc and if not then resize?

    measurements = postprocess_agent_results(
        measurements,
        io_config,
        scan_convert_order=0,  # always 0 for masks!
        drop_first_n_frames=drop_first_n_frames,
        image_range=image_range,
        scan_convert_resolution=scan_convert_resolution,
    )

    # rescale DST outputs to get correct aspect ratio
    aspect_ratio = ops.shape(measurements)[2] / ops.shape(measurements)[1]
    new_shape = (
        ops.shape(targets_with_mask)[1],
        int(ops.shape(targets_with_mask)[1] * aspect_ratio),
    )
    targets_with_mask = ops.image.resize(
        targets_with_mask, new_shape, interpolation="nearest"
    )
    reconstructions_with_mask = ops.image.resize(
        reconstructions_with_mask, new_shape, interpolation="nearest"
    )

    # apply log for visualization
    saliency_maps = postprocess_heatmap(saliency_maps, io_config, cmap="magma_r")
    posterior_std = postprocess_heatmap(posterior_std, io_config, cmap="magma_r")

    arrays = [
        targets_with_mask,
        measurements,
        reconstructions_with_mask,
        posterior_std,
        saliency_maps,
    ]
    labels = [
        "Targets",
        "Measurements",
        "Reconstructions",
        "STD[X | Y]",
        "DST Saliency",
    ]

    # Make gif
    side_by_side_gif(
        save_dir / gif_name,
        *arrays,
        dpi=dpi,
        interpolation=interpolation_matplotlib,
        fps=io_config.gif_fps if hasattr(io_config, "gif_fps") else 10,
        context=context,
        labels=labels,
    )


def plot_belief_distribution_for_presentation(
    save_dir,
    belief_distribution,  # shape (num_beliefs, H, W, 1)
    masks,  # shape (num_beliefs, H, W) or (num_beliefs, H, W, 1)
    io_config,
    frame_idx=0,
    dpi=150,
    scan_convert_order=0,
    interpolation_matplotlib="nearest",
    image_range=(-1, 1),
    fill_value=np.nan,
    next_masks=None,
    context="styles/darkmode.mplstyle",
):
    """
    Plots a grid of scan-converted belief images, pixelwise variance, and a single data-space measurements image.

    Args:
        save_dir (str or Path): Directory to save the output images.
        belief_distribution (ndarray): Array of shape (num_beliefs, H, W, 1).
        masks (ndarray): Array of shape (num_beliefs, H, W) or (num_beliefs, H, W, 1).
        io_config (dict): IO configuration.
        dpi (int): Dots per inch for saved figures.
        scan_convert_order (int): Order for scan conversion.
        interpolation_matplotlib (str): Interpolation for matplotlib.
        image_range (tuple): Range of image values.
        context (str): Matplotlib style context.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    beliefs = ops.clip(belief_distribution, image_range[0], image_range[1])
    beliefs = ops.cast(beliefs, "float32")
    num_beliefs = beliefs.shape[0]

    # Remove channel dimension for scan conversion
    # beliefs = beliefs[..., 0]  # shape: (num_beliefs, H, W)

    # Prepare masks
    if masks.shape[-1] == 1:
        masks = masks[..., 0]
    masks = ops.cast(masks, "bool")

    # Compute measurements in data space (all beliefs share the same mask)
    # Use the first belief as the reference image for masking
    measurements = ops.where(
        masks[0], beliefs[0], ops.ones_like(beliefs[0]) * image_range[0]
    )
    measurements = measurements * masks[0]  # zero-out the unmeasured vals

    # Scan convert each belief image
    scan_converted_beliefs = []
    for i in range(num_beliefs):
        sc_belief = _scan_convert(
            beliefs[i],
            io_config.scan_conversion_angles,
            order=scan_convert_order,
            fill_value=fill_value,
            resolution=0.1,
        )
        scan_converted_beliefs.append(sc_belief)
    scan_converted_beliefs = np.stack(scan_converted_beliefs, axis=0)

    # Plot grid of beliefs
    with plt.style.context(context):
        fig_grid, _ = plot_image_grid(scan_converted_beliefs, ncols=2, figsize=(8, 8))
        fig_grid.suptitle("Belief Distribution (Scan Converted)", fontsize=18)
        plt.tight_layout()
        fig_grid.savefig(
            save_dir / f"belief_distribution_grid_{frame_idx}.png",
            dpi=dpi,
            transparent=True,
        )
        plt.close(fig_grid)
        log.info(log.yellow(save_dir / f"belief_distribution_grid_{frame_idx}.png"))

    pixelwise_variance = np.var(beliefs, axis=0)
    sc_variance = _scan_convert(
        pixelwise_variance,
        io_config.scan_conversion_angles,
        order=scan_convert_order,
        fill_value=fill_value,
        resolution=0.1,
    )

    # Plot pixelwise variance
    with plt.style.context(context):
        fig_var, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(
            sc_variance,
            cmap="magma",
            interpolation=interpolation_matplotlib,
        )
        ax.set_title("Pixelwise Variance", fontsize=18)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.6)
        plt.tight_layout()
        fig_var.savefig(
            save_dir / f"belief_distribution_variance_{frame_idx}.png",
            dpi=dpi,
            transparent=True,
        )
        plt.close(fig_var)
        log.info(log.yellow(save_dir / f"belief_distribution_variance_{frame_idx}.png"))

    sc_measurements = _scan_convert(
        measurements,
        io_config.scan_conversion_angles,
        order=scan_convert_order,
        fill_value=fill_value,
        resolution=0.1,
    )
    with plt.style.context(context):
        fig_meas, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(
            sc_measurements, cmap="gray", interpolation=interpolation_matplotlib
        )
        ax.set_title("Measurements", fontsize=18)
        ax.axis("off")
        plt.tight_layout()
        fig_meas.savefig(save_dir / f"measurements_{frame_idx}.png", dpi=dpi)
        plt.close(fig_meas)
        log.info(log.yellow(save_dir / f"measurements_{frame_idx}.png"))

    # Plot measurements (masked image, data space, no scan conversion)
    measurements_uint8 = map_range(measurements, image_range, (0, 255)).astype(np.uint8)
    with plt.style.context(context):
        fig_meas, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(
            measurements_uint8, cmap="gray", interpolation=interpolation_matplotlib
        )
        ax.set_title("Measurements (Data Space)", fontsize=18)
        ax.axis("off")
        plt.tight_layout()
        fig_meas.savefig(save_dir / f"measurements_data_space_{frame_idx}.png", dpi=dpi)
        plt.close(fig_meas)
        log.info(log.yellow(save_dir / f"measurements_data_space_{frame_idx}.png"))

    sc_selected = _scan_convert(
        ops.cast(next_masks, "float32") * ops.max(pixelwise_variance)
        + (pixelwise_variance * ~ops.cast(next_masks, "bool")),
        io_config.scan_conversion_angles,
        order=scan_convert_order,
        fill_value=fill_value,
        resolution=0.1,
    )
    with plt.style.context(context):
        fig_meas, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(
            sc_selected, cmap="magma", interpolation=interpolation_matplotlib
        )
        ax.set_title("Mask t", fontsize=18)
        ax.axis("off")
        plt.tight_layout()
        fig_meas.savefig(
            save_dir / f"selected_next_t_{frame_idx}.png", dpi=dpi, transparent=True
        )
        plt.close(fig_meas)
        log.info(log.yellow(save_dir / f"selected_next_t_{frame_idx}.png"))
