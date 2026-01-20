"""Tools for elevation interpolation visualization and processing."""

import functools
import inspect
from pathlib import Path
from typing import Union

import cv2
import h5py
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import ops
from matplotlib.patches import FancyArrow

from zea import log
from zea.agent.selection import EquispacedLines
from zea.display import scan_convert_2d, scan_convert_3d
from zea.func import (
    interpolate_data,
    split_volume_data_from_axis,
    stack_volume_data_along_axis,
    translate,
    vmap,
)
from zea.internal.cache import cache_output
from zea.io_lib import matplotlib_figure_to_numpy, save_to_gif
from zea.visualize import plot_frustum_vertices, plot_image_grid

TITLE_LOOKUP = {
    "masks": "Acquisitions",
    "nearest": "Nearest",
    "linear": "Linear",
    "supervised": "U-Net",
    "diffusion_vanilla": "Diffusion (w/o SeqDiff)",
    "diffusion": "Diffusion",  # Generative prior",
    "clean_images": "Ground truth",
    "variance": "Variance",
    "composite": "Composite",
}


def visualize_masks(masks, show_num_frames=4, show_num_elevations=4):
    assert masks.ndim == 4, "masks must have shape (frames, elevations, height, width)"
    masks = masks[:show_num_frames, :show_num_elevations]

    frames, depth, height, width = masks.shape

    # Flatten the masks for plotting
    masks_flat = ops.reshape(masks, (frames * depth, height, width))

    aspect = width / height

    fig, ax = plot_image_grid(
        masks_flat,
        ncols=depth,
        aspect=aspect,
        figsize=(depth * 3, frames * 3),
        remove_axis=False,
        text_color="white",
    )
    for ax in fig.axes:
        for spine in ax.spines.values():
            # spine.set_visible(False)
            spine.set_color("white")

    # now indicate on top of the plot a big arrow to the right
    # indicating the elevation
    # and a big arrow to the bottom indicating the frames to the left of the plot

    # Add arrows
    fig.text(
        0.5, 0.95, "Elevation", ha="center", va="center", fontsize=20, color="white"
    )
    fig.text(
        0.05,
        0.5,
        "Frames",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=20,
        color="white",
    )
    # Add arrows
    arrow_elevation = FancyArrow(
        0.4, 0.92, 0.2, 0, width=0.005, color="white", transform=fig.transFigure
    )
    arrow_frames = FancyArrow(
        0.08, 0.6, 0, -0.2, width=0.005, color="white", transform=fig.transFigure
    )

    fig.add_artist(arrow_elevation)
    fig.add_artist(arrow_frames)

    fig.savefig("test.png", bbox_inches="tight")


@cache_output()
def preprocess(data, image_shape, image_range, normalization_range):
    assert len(image_shape) == 3, (
        "image_shape must be a tuple of (height, width, channels)"
    )
    assert len(image_range) == 2, "image_range must be a tuple of (min, max)"
    assert len(normalization_range) == 2, (
        "normalization_range must be a tuple of (min, max)"
    )
    assert data.ndim == 4, "data must have shape (frames, azimuth, depth, elevation)"
    # go from depth, azimuth, elevation
    # to azimuth, depth, elevation
    data = ops.transpose(data, (0, 2, 1, 3))
    data = ops.expand_dims(data, axis=-1)

    # reshape so all outer dimensions become a single batch dimension
    shape = data.shape
    data = ops.reshape(data, (-1, *data.shape[-3:]))

    data = ops.image.resize(data, (image_shape[0], image_shape[1]))
    data = translate(data, image_range, normalization_range)
    data = ops.reshape(data, (*shape[:-3], *image_shape))
    data = ops.cast(data, "float32")
    return data


def batch_interpolation(measurements, masks, order=1):
    output = []
    batch_size = measurements.shape[0]
    assert all(m == 1 or m == n for m, n in zip(masks.shape, measurements.shape)), (
        "masks must be broadcastable to measurements"
    )

    # broadcast masks to measurements
    masks = ops.broadcast_to(masks, measurements.shape)

    for batch_idx in range(batch_size):
        measurement, mask = measurements[batch_idx], masks[batch_idx]
        output.append(
            interpolate_data(
                measurement[ops.cast(mask, "bool")],
                mask,
                order=order,
                axis=-1,
            )
        )

    return ops.stack(output)


@cache_output()
def interpolate_volumes(
    measurements,
    masks,
    method="linear",
    model_kwargs=None,
    diffusion_model=None,
    model=None,
    buffer_length=0,
    batch_size=None,
    seed_gen=None,
    average_posterior_samples=True,
):
    assert measurements.ndim == 5, (
        "volumes should have dimensions [batch, frames, azimuth, depth, elevation, 1]"
    )

    output = {
        "output": [],
    }

    masks = ops.broadcast_to(masks, measurements.shape)

    log.info(f"Interpolating volume with {log.green(method)}...")
    if method in ["linear", "nearest", "supervised"]:
        order = 1 if method == "linear" else 0
        measurements = measurements[buffer_length:]
        masks = masks[buffer_length:]

        progbar = keras.utils.Progbar(len(measurements), unit_name="frame")

        for measurement, mask in zip(measurements, masks):
            if method == "supervised":
                func = functools.partial(
                    model.predict,
                    verbose=False,
                )
                # make sure measurements at masks zero locations is center value
                center_value = (model.image_range[0] + model.image_range[1]) / 2
                measurement = ops.where(
                    mask,
                    measurement,
                    center_value,
                )
                outputs = vmap(
                    func,
                    batch_size=batch_size,
                    disable_jit=True,
                    fn_supports_batch=True,
                )(measurement)

            else:
                measurement = ops.squeeze(measurement, axis=-1)
                mask = ops.squeeze(mask, axis=-1)
                outputs = batch_interpolation(
                    measurement,
                    mask,
                    order=order,
                )

            progbar.add(1)

            # output["measurements"].append(measurement)
            output["output"].append(outputs)

        for key in output:
            output[key] = ops.stack(output[key], axis=0)
            if method in ["linear", "nearest"]:
                output[key] = ops.expand_dims(output[key], axis=-1)

    elif "diffusion" in method:
        assert model_kwargs is not None, "model_kwargs must be provided for diffusion"
        n_frames = diffusion_model.image_shape[-1]

        if n_frames > 1:
            padding = measurements.shape[1] % n_frames
            measurements = stack_volume_data_along_axis(
                measurements,
                batch_axis=1,
                stack_axis=-1,
                number=n_frames,
            )
            masks = stack_volume_data_along_axis(
                masks,
                batch_axis=1,
                stack_axis=-1,
                number=n_frames,
            )

        # to (batch, frames, height, width, channels)
        # where batch is the elevation and height is the azimuth
        # -> [batch(elevation), frames, height(depth), width(azimuth), channels]
        measurements = ops.swapaxes(measurements, 0, 1)
        masks = ops.swapaxes(masks, 0, 1)

        output = diffusion_model(
            measurements,
            masks=masks,
            num_steps=model_kwargs["num_steps"],
            min_num_steps=model_kwargs["min_num_steps"],
            num_particles=model_kwargs["num_particles"],
            buffer_length=buffer_length,
            init_guidance_kwargs=model_kwargs.get("init_guidance_kwargs"),
            batch_size=batch_size,
            seed=seed_gen,
            verbose=True,
            guidance_kwargs=model_kwargs["guidance_kwargs"],
        )
        posterior_samples = output["output"]

        posterior_samples = ops.clip(posterior_samples, *diffusion_model.image_range)
        output["posterior_samples"] = posterior_samples

        # posterior mean over all particles
        if average_posterior_samples:
            pred_images = ops.mean(posterior_samples, axis=-4)
        else:
            pred_images = ops.take(posterior_samples, axis=-4, indices=0)
        output["output"] = pred_images

        if output["initial_samples"] is not None:
            if average_posterior_samples:
                output["initial_samples"] = ops.mean(output["initial_samples"], axis=-4)
            else:
                output["initial_samples"] = ops.take(
                    output["initial_samples"], axis=-4, indices=0
                )

        # seqdiff has frames as dim 1, and elevation as dim 0
        # we need to swap these dimensions back to the original shape
        for key in output:
            # initial samples can be None if vanilla inference
            # in that case we don't stack
            if output[key] is None:
                continue

            output[key] = ops.swapaxes(output[key], 0, 1)

            if n_frames > 1:
                output[key] = split_volume_data_from_axis(
                    output[key],
                    batch_axis=1,
                    stack_axis=-1,
                    number=n_frames,
                    padding=padding,
                )
    else:
        raise ValueError(f"method {method} not recognized.")

    return output


def sweeping_frustum_visualization():
    # Specify the range of the frustum
    rho_range = (0.1, 1.0)  # in meters
    theta_range = (-np.pi / 4, np.pi / 4)  # in radians
    phi_range = (-np.pi / 4, np.pi / 4)  # in radians

    # Specify only one plane
    fig, ax = plot_frustum_vertices(rho_range, theta_range, phi_range, phi_plane=0.5)
    path = "temp/frustum_vertices.png"
    fig.savefig(path, bbox_inches="tight")
    log.success(f"Saved image to {log.yellow(path)}")
    # plot_frustum_vertices(rho_range, theta_range, phi_range, theta_plane=0.2)
    # plot_frustum_vertices(rho_range, theta_range, phi_range, rho_plane=0.5)


def postprocess_3d_data(
    data,
    normalization_range,
    scan_convert=True,
    scan_convert_kwargs=None,
    filename=None,
    scan_convert_mode="cartesian",
    swap_axes=False,
):
    """Postprocess 3D ultrasound volumes.

    Args:
        data (Tensor): 3D ultrasound data with either
            shape [frames, depth, azimuth, elevation]
            or shape [frames, azimuth, depth, elevation].
            See `swap_axes` for more information. Can also have an additional
            channel dimension at the end (has to be singleton).
        normalization_range (tuple): Range of the input data.
        scan_convert (bool): Whether to perform scan conversion.
        filename (str): Path to the HDF5 file containing the scan conversion parameters.
            These can also be passed directly via `scan_convert_kwargs`.
        scan_convert_kwargs (dict): Scan conversion parameters.
        scan_convert_mode (str): Scan conversion mode.
            Can be one of ["cartesian", "cartesian_phi", "cartesian_theta"].
            "cartesian" is the full 3D scan conversion, while "cartesian_phi" and
            "cartesian_theta" are 2D scan conversions along the elevation and
            azimuth axes, respectively.
        swap_axes (bool): Whether to swap the azimuth and depth axes. If True, the input
            is assumed to be in the shape [frames, azimuth, depth, elevation]. If False,
            the input is assumed to be in the shape [frames, depth, azimuth, elevation].
            i.e. the default configuration.

    Returns:
        np.ndarray: Postprocessed 3D ultrasound data. If `scan_convert` is True, the
            output will be in the shape [frames, z, x, y]. Otherwise, the output
            will be in the shape [frames, depth, azimuth, elevation]. Data is normalized to
            the range [0, 255] and converted to uint8 if possible (i.e. no NaNs).
        info (dict): A dictionary containing information about the scan conversion.
            Contains the resolution, x, y, and z limits, rho, theta, and phi ranges,
            and the fill value.
    """

    info = None

    if ops.shape(data)[-1] == 1:
        data = ops.squeeze(data, axis=-1)

    # go from azimuth, depth, elevation
    # to depth, azimuth, elevation
    if swap_axes:
        data = ops.swapaxes(data, -3, -2)

    # make sure -3 axis is depth (which is largest)
    assert ops.shape(data)[-3] == ops.max(ops.shape(data)), (
        "data should have dimensions [frames, depth, azimuth, elevation] "
        "but got dimensions "
        f"{ops.shape(data)}"
    )

    if scan_convert:
        assert scan_convert_mode in [
            "cartesian",
            "cartesian_phi",
            "cartesian_theta",
        ], f"scan_convert_mode {scan_convert_mode} not recognized."

        # assert all parameters are provided
        if filename is not None:
            if scan_convert_kwargs is None:
                scan_convert_kwargs = {}
            with h5py.File(filename, "r", locking=False) as f:
                rho = f["/scan/frustum/rho"][()]
                theta = f["/scan/frustum/theta"][()]
                phi = f["/scan/frustum/phi"][()]
                scan_convert_kwargs["rho_range"] = (ops.min(rho), ops.max(rho))
                scan_convert_kwargs["theta_range"] = (ops.min(theta), ops.max(theta))
                scan_convert_kwargs["phi_range"] = (ops.min(phi), ops.max(phi))

        assert scan_convert_kwargs is not None, "scan_convert_kwargs must be provided"

        assert all(
            key in scan_convert_kwargs
            for key in ["rho_range", "theta_range", "phi_range", "fill_value"]
        ), (
            "scan_convert_kwargs must contain `rho_range`, `theta_range`, "
            f"`phi_range`, and `fill_value`, but got {scan_convert_kwargs.keys()}"
        )

        if scan_convert_mode == "cartesian":
            data, info = scan_convert_3d(
                data,
                **scan_convert_kwargs,
            )
            # data is now [..., z, x, y] all in cartesian coordinates
        elif scan_convert_mode == "cartesian_phi":
            # from [frames, depth, azimuth, elevation]
            # to [frames, azimuth, depth, elevation]
            data = ops.swapaxes(data, -3, -2)
            data, info = scan_convert_2d(
                data,
                rho_range=scan_convert_kwargs["rho_range"],
                theta_range=scan_convert_kwargs["phi_range"],
                fill_value=scan_convert_kwargs["fill_value"],
            )
            data = ops.swapaxes(data, -3, -2)
            # reshapes to make sure we have [..., z, x, y] again
            # or in this case [..., z, azimuth, y]
        elif scan_convert_mode == "cartesian_theta":
            # from [frames, depth, azimuth, elevation]
            # to [frames, elevation, depth, azimuth]
            data = ops.swapaxes(data, -1, -2)
            data = ops.swapaxes(data, -2, -3)
            data, info = scan_convert_2d(
                data,
                rho_range=scan_convert_kwargs["rho_range"],
                theta_range=scan_convert_kwargs["theta_range"],
                fill_value=scan_convert_kwargs["fill_value"],
            )
            data = ops.swapaxes(data, -2, -3)
            data = ops.swapaxes(data, -1, -2)
            # reshapes to make sure we have [..., z, x, y] again
            # or in this case [..., z, x, elevation]
        else:
            raise ValueError(f"scan_convert_mode {scan_convert_mode} not recognized.")

    if normalization_range is None:
        normalization_range = ops.min(data), ops.max(data)

    image_range = (0, 255)

    def _clip_and_translate(data):
        out = []
        for i in range(data.shape[0]):  # loop for memory efficiency
            _data = data[i]
            _data = ops.clip(_data, *normalization_range)
            _data = translate(_data, normalization_range, image_range)
            _data = ops.clip(_data, *image_range)
            out.append(_data)
        return ops.convert_to_numpy(out)

    data = _clip_and_translate(data)

    if np.any(np.isnan(data)):
        return data, info
    else:
        return data.astype("uint8"), info


def auto_crop(image, pad=0, background_value=0, bounding_box=None):
    """Automatically crop empty space (background) from an image with optional padding.

    Args:
        image: RGB image to crop
        pad: Number of pixels to pad around the cropped area
        background_value: Pixel value to consider as background (0=black, 255=white)
        bounding_box: Optional bounding box to crop the image. If provided, it will be used
            instead of calculating the bounding box from the image. Should be a tuple
            (x1, y1, x2, y2) representing the coordinates of the bounding box.

    Returns:
        tuple: (cropped_image, bounding_box)
            - cropped_image: The cropped image
            - bounding_box: The coordinates (x1, y1, x2, y2) of the cropping box
    """
    if bounding_box is not None:
        # Use the provided bounding box directly
        x1, y1, x2, y2 = bounding_box

        # Ensure coordinates are within image bounds
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, image.shape[1])
        y2 = min(y2, image.shape[0])

        cropped = image[y1:y2, x1:x2]
        return cropped, (x1, y1, x2, y2)

    # Otherwise, calculate the bounding box automatically
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if background_value == 0:
        # For dark background, use regular thresholding
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    else:
        # For light background (e.g., 255=white), invert the threshold
        _, thresh = cv2.threshold(
            gray, background_value - 1, 255, cv2.THRESH_BINARY_INV
        )

    coords = cv2.findNonZero(thresh)  # Get nonzero pixels

    # If no non-background pixels found, return original image
    if coords is None:
        return image, (0, 0, image.shape[1], image.shape[0])

    x, y, w, h = cv2.boundingRect(coords)  # Find bounding box

    # Apply padding
    y1 = max(y - pad, 0)
    y2 = min(y + h + pad, image.shape[0])
    x1 = max(x - pad, 0)
    x2 = min(x + w + pad, image.shape[1])

    cropped = image[y1:y2, x1:x2]  # Crop with padding
    return cropped, (x1, y1, x2, y2)


def plot_slices_from_3d_volumes(
    data: dict,
    select_dims: dict[str, Union[int, list[int]]],  # e.g. {'t': [0,1,2], 'd': 5}
    broadcast_dims: list[str],  # e.g. ['a', 'e'] to broadcast azimuth and elevation
    dim_name_to_idx=None,
    aspect="equal",
    fig=None,
    fig_contents=None,
    save_path=None,
    verbose=True,
    show_frustum=False,
    frustum_kwargs=None,
    show_indices=True,
    show_indices_dim="all",
    dpi=300,
    hide_dims=None,
    # padding of labels
    labelpad=5,
    **kwargs,
):
    """Plot 2D slices from 3D volumes for comparison by selecting specific dimensions.

    Parameters:
        data (dict): Dictionary containing 4D volumes from which to plot 2D slices.
            Each value should have shape [frames(t), depth(d), azimuth(a), elevation(e)].
        select_dims (dict): Dictionary specifying which dimensions to select values from.
            Keys must be from ['t', 'd', 'a', 'e'] representing time, depth, azimuth, elevation.
            Values can be either:
            - An integer indicating which index to select
            - A list of integers (only one dimension can have a list)
        broadcast_dims (list): List of two dimension names to broadcast (use all values).
            Must be from ['t', 'd', 'a', 'e'] and cannot overlap with select_dims keys.
        aspect (str, optional): Aspect ratio for the plots. Defaults to "equal".
        fig (matplotlib.figure.Figure, optional): Existing figure to plot on.
        fig_contents (dict, optional): Existing figure contents.
        save_path (str, optional): Path to save the resulting figure.
        verbose (bool, optional): Whether to log progress. Defaults to True.
        show_frustum (bool, optional): Whether to show a frustum image. Defaults to False.
        frustum_kwargs (dict, optional): Keyword arguments for plot_frustum_vertices().
            Defaults are:
            - dim_mapping: Mapping of tensor dimensions to frustum angles.
                Defaults to {"a": "theta", "e": "phi"}.
            - title: Title of the frustum plot. Defaults to "Frustum".
            - fig: Existing figure to plot on.
            - ax: Existing axis to plot on.
            - rho_range: (0.1, 1.0)
            - theta_range: (-np.pi / 4, np.pi / 4)
            - phi_range: (-np.pi / 4, np.pi / 4)
            - color_frustum: Color of the frustum. Defaults to "gray".
        hide_dims (list[str], optional): List of dimension names to hide from axis labels.
            Must be from ['t', 'd', 'a', 'e']. Defaults to None.
        **kwargs: Additional keyword arguments passed to plot_image_grid().

    Returns:
        tuple: (matplotlib.figure.Figure, dict)
            - Figure object containing the plots
            - Dictionary containing figure contents

    Raises:
        AssertionError: If data dimensions, select_dims, or broadcast_dims are invalid.
    """

    if dim_name_to_idx is None:
        dim_name_to_idx = {"t": 0, "d": 1, "a": 2, "e": 3}

    dim_names = list(dim_name_to_idx.keys())

    for data_value in data.values():
        assert data_value.ndim == 4, (
            "data should have dimensions [frames, depth, azimuth, elevation]"
        )

    assert len(broadcast_dims) == 2, "broadcast_dims must contain exactly 2 dimensions"
    assert all(d in dim_names for d in broadcast_dims), (
        f"broadcast_dims must be from {dim_names}"
    )
    assert all(d in dim_names for d in select_dims), (
        f"select_dims keys must be from {dim_names}"
    )
    assert not any(d in broadcast_dims for d in select_dims), (
        "broadcast_dims cannot overlap with select_dims"
    )

    # Validate only one dimension has a list
    list_dims = [d for d, v in select_dims.items() if isinstance(v, list)]
    assert len(list_dims) <= 1, "Only one dimension can have multiple values"

    # Create base slicing tuple with broadcast dims as slice(None)
    slices = [slice(None)] * 4
    images = np.stack(list(data.values()))
    titles = list(data.keys())

    # Determine the list dimension and its indices
    list_dim = list_dims[0] if list_dims else next(iter(select_dims.keys()))
    list_indices = (
        select_dims[list_dim]
        if isinstance(select_dims[list_dim], list)
        else [select_dims[list_dim]]
    )
    list_dim_idx = dim_name_to_idx[list_dim]

    # Set other dimensions' slicing
    for dim, val in select_dims.items():
        if dim != list_dim:
            slices[dim_name_to_idx[dim]] = val

    # Create list of 2D images, one for each index in the list dimension
    all_images = []
    bounding_box = None
    for idx in list_indices:
        # create selectors to grab the images for this row from 'images' tensor
        dimension_selectors = slices.copy()
        dimension_selectors[list_dim_idx] = idx
        images_for_row = list(
            images[
                ...,
                dimension_selectors[0],  # time dimension
                dimension_selectors[1],  # depth dimension
                dimension_selectors[2],  # azimuth dimension
                dimension_selectors[3],  # elevation dimension
            ]
        )

        if show_frustum:
            default_frustum_kwargs = {
                "dim_mapping": {"a": "theta", "e": "phi"},
                "title": "Frustum",
                "fig": None,
                "ax": None,
                "rho_range": (0.1, 1.0),
                "theta_range": (-np.pi / 4, np.pi / 4),
                "phi_range": (-np.pi / 4, np.pi / 4),
                "color_frustum": "gray",
                "plotting_kwargs": {"lw": "6"},
            }
            if frustum_kwargs is None:
                frustum_kwargs = {}

            frustum_kwargs = {**default_frustum_kwargs, **frustum_kwargs}

            # Map selected dimensions to frustum angles

            for tensor_dim, angle_name in frustum_kwargs["dim_mapping"].items():
                if tensor_dim in select_dims:
                    dim_value = select_dims[tensor_dim]
                    # if the current dim is the list dim (i.e. we're making multiple rows)
                    # then we'll just want to make a frustum image for the current row,
                    # given by idx.
                    if tensor_dim == list_dim:
                        dim_value = idx
                    # Map the tensor index to angle range
                    angle_range = frustum_kwargs[angle_name + "_range"]

                    # size dim, using + 1 as we have an additional dimension due to the
                    # 2D grid plot layout
                    size_dim = images.shape[dim_name_to_idx[tensor_dim] + 1]
                    normalized_idx = dim_value / size_dim
                    angle = angle_range[0] + normalized_idx * (
                        angle_range[1] - angle_range[0]
                    )
                    # round angle to the nearest 10 degrees
                    angle = np.rad2deg(angle)
                    angle = np.round(angle / 10) * 10
                    if angle < -5:
                        angle -= 10

                    if angle > 5:
                        angle += 10

                    angle = np.deg2rad(angle)

                    frustum_kwargs[f"{angle_name}_plane"] = float(angle)

            # create frustum image
            sig = inspect.signature(plot_frustum_vertices)
            param_names = list(sig.parameters.keys())
            pass_frustum_kwargs = {
                k: v for k, v in frustum_kwargs.items() if k in param_names
            }

            frustum_fig, frustum_ax = plot_frustum_vertices(**pass_frustum_kwargs)

            # crop and resize frustum image to fit with other images
            frustum_ax.set_position([0, 0, 2, 2])
            frustum_image = matplotlib_figure_to_numpy(
                frustum_fig  # , savefig_kwargs={"dpi": 300, "bbox_inches": "tight"}
            )

            background_value = frustum_image[0, 0, 0]
            frustum_image, bounding_box = auto_crop(
                frustum_image,
                pad=10,
                background_value=background_value,
                bounding_box=bounding_box,
            )
            target_height = min(
                image.shape[0] for image in images_for_row
            )  # Use the smallest image height

            target_width = (
                frustum_image.shape[1] * target_height / frustum_image.shape[0]
            )

            frustum_image = cv2.resize(
                frustum_image,
                (
                    int(target_width),
                    int(target_height),
                ),
            )

            images_for_row = [frustum_image] + images_for_row
        all_images.extend(images_for_row)

    if show_frustum:
        titles = [frustum_kwargs["title"]] + titles
        num_columns = len(titles)
        # Set aspect=1 for frustum images (first column), aspect=param for others
        aspect_list = [
            1 if i % num_columns == 0 else aspect for i in range(len(all_images))
        ]
    else:
        aspect_list = aspect

    # Plot images
    if verbose:
        log.info("Plotting images...")

    fig, fig_contents = plot_image_grid(
        all_images,
        ncols=len(titles),
        aspect=aspect_list,
        titles=titles + [""] * (len(all_images) - len(titles)),
        remove_axis=False,
        fig=fig,
        fig_contents=fig_contents,
        **kwargs,
    )

    # Add labels
    axes = fig.axes
    axes = np.array(axes).reshape(-1, len(titles))

    # Determine the list dimension and its indices
    list_dim = list_dims[0] if list_dims else next(iter(select_dims.keys()))
    list_indices = (
        select_dims[list_dim]
        if isinstance(select_dims[list_dim], list)
        else [select_dims[list_dim]]
    )

    # Add labels to the left of each row indicating the 'select_dims' values
    if show_indices:
        for row_idx, idx in enumerate(list_indices):
            select_labels = []

            def _convert_dim_to_angle(dim, idx):
                if dim not in ["e", "a"]:
                    return idx
                dim_to_angle_name = {"e": "phi", "a": "theta"}
                angle_name = dim_to_angle_name[dim]
                angle_range = frustum_kwargs[f"{angle_name}_range"]
                size_dim = images.shape[dim_name_to_idx[dim] + 1]
                # Avoid division by zero for singleton dimensions
                max_idx = max(size_dim - 1, 1)
                idx = translate(idx, [0, max_idx], angle_range)
                idx = np.rad2deg(idx)

                # round to the nearest 10 degrees
                # for neat viz
                idx = np.round(idx / 10) * 10

                # round and add degrees symbol
                return f"${idx:.0f}°$"

            idx = _convert_dim_to_angle(list_dim, idx)

            # Add the current dimension's value if not hidden
            if hide_dims is None or list_dim not in hide_dims:
                select_labels.append(f"{list_dim} = {idx}")

            # Add other selected dimensions if not hidden
            for dim, val in select_dims.items():
                if show_indices_dim != "all":
                    assert isinstance(show_indices_dim, (list, str)), (
                        "show_indices_dim must be a list or a string"
                    )
                    if dim not in show_indices_dim:
                        continue

                if dim != list_dim and (hide_dims is None or dim not in hide_dims):
                    val = _convert_dim_to_angle(dim, val)
                    select_labels.append(f"{dim} = {val}")

            select_labels = "\n".join(select_labels)
            if select_labels:  # Only set ylabel if there are labels to show
                axes[row_idx, 0].set_ylabel(
                    select_labels,
                    labelpad=labelpad,
                    fontsize=12,
                    ha="left",
                    va="center",
                    y=0.2,
                )

    # Add labels for broadcast dimensions if not hidden
    broadcast_dims_sorted = sorted(broadcast_dims, key=lambda x: dim_name_to_idx[x])
    vertical_dim, horizontal_dim = broadcast_dims_sorted
    dim_labels = {"e": "Elevation", "a": "Azimuth", "d": "Depth", "t": "Time"}
    last_ax = axes[(len(axes) // 2) - 1, -1]  # Bottom right axis

    if hide_dims is None or vertical_dim not in hide_dims:
        last_ax.set_ylabel(
            rf"$\leftarrow \quad {dim_labels[vertical_dim]}$", fontsize=12
        )
        last_ax.yaxis.set_label_position("right")
        last_ax.yaxis.get_label().set_visible(True)

    if hide_dims is None or horizontal_dim not in hide_dims:
        last_ax.set_xlabel(
            rf"${dim_labels[horizontal_dim]} \quad \rightarrow$",
            labelpad=-5,
            fontsize=12,
        )

    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        log.success(f"Saved image to {log.yellow(save_path)}")
        plt.close()

    return fig, fig_contents, frustum_kwargs


def animate_slices_from_3d_volumes(
    data: dict,
    select_dims: dict[str, Union[int, list[int]]],  # e.g. {'t': 0, 'e': range(10)}
    broadcast_dims: list[str],  # e.g. ['a', 'd'] to broadcast azimuth and depth
    animate_dim: str,  # which dimension to animate (must be in select_dims and be a list/range)
    dim_name_to_idx={"t": 0, "d": 1, "a": 2, "e": 3},
    save_path=None,
    fps=5,
    **kwargs,
):
    """Create an animation by iterating over one dimension while keeping others fixed.

    Parameters:
        data (dict): Dictionary containing 4D volumes from which to plot 2D slices.
            Each value should have shape [frames(t), depth(d), azimuth(a), elevation(e)].
        select_dims (dict): Dictionary specifying which dimensions to select values from.
            Keys must be from ['t', 'd', 'a', 'e'].
            The animate_dim must have a list/range value, others must be integers.
        broadcast_dims (list): List of two dimension names to broadcast (use all values).
            Must be from ['t', 'd', 'a', 'e'] and cannot overlap with select_dims keys.
        animate_dim (str): Which dimension to animate (must be in select_dims).
        save_path (str, optional): Path to save the resulting GIF.
        fps (int, optional): Frames per second for the output GIF. Defaults to 5.
        **kwargs: Additional keyword arguments passed to plot_slices_from_3d_volumes().

    Returns:
        list: List of numpy arrays containing the frame images
    """
    # Validate inputs
    assert animate_dim in select_dims, (
        f"animate_dim must be in select_dims, got {animate_dim}"
    )
    animate_values = select_dims[animate_dim]
    assert isinstance(animate_values, (list, range)), (
        f"The animated dim {animate_dim} in select_dims[{animate_dim}] must be a list or range"
    )

    fig = None
    fig_contents = None

    # extract frustum kwargs if present
    if "frustum_kwargs" in kwargs:
        frustum_kwargs = kwargs.pop("frustum_kwargs")
    else:
        frustum_kwargs = None

    out_figures = []

    log.info("Creating animation...")
    progbar = keras.utils.Progbar(len(animate_values), unit_name="frame")

    for idx in animate_values:
        # Create new select_dims with current animation frame
        current_select_dims = select_dims.copy()
        current_select_dims[animate_dim] = idx

        fig, fig_contents, frustum_kwargs = plot_slices_from_3d_volumes(
            data,
            select_dims=current_select_dims,
            broadcast_dims=broadcast_dims,
            dim_name_to_idx=dim_name_to_idx,
            fig=fig,
            fig_contents=fig_contents,
            save_path=None,
            verbose=False,
            frustum_kwargs=frustum_kwargs,
            **kwargs,
        )

        # Convert matplotlib figure to numpy array
        out_figures.append(matplotlib_figure_to_numpy(fig))

        progbar.add(1)

    if save_path is not None:
        save_to_gif(out_figures, save_path, fps=fps)
        plt.close()

    return out_figures


def determine_subsampling_params(
    total_num_lines: int,
    subsampling_rate: float,
    line_width: int,
):
    """
    Determines the effective subsampling rate, line width, and number of lines to sample
    given the total number of lines, requested subsampling rate, and line width.

    Returns:
        {
            "line_width": int,
            "effective_subsampling_rate": float,
            "num_lines_to_sample": int,
            "effective_num_lines": int,
            "factors": List[int],
        }
    """

    total_lines = total_num_lines // line_width

    # Find the closest valid number of lines to sample (must be a factor of total_lines)
    requested_num_lines = int(np.round(total_num_lines * subsampling_rate / line_width))

    # Find all factors of total_lines
    factors = [i for i in range(1, total_lines + 1) if total_lines % i == 0]
    # Find the factor closest to requested_num_lines
    closest_num_lines = min(factors, key=lambda x: abs(x - requested_num_lines))

    effective_subsampling_rate = closest_num_lines / total_lines
    effective_subsampling_rate = float(np.round(effective_subsampling_rate, 2))

    effective_num_lines = int(np.round(effective_subsampling_rate * total_num_lines))

    return {
        "line_width": line_width,
        "effective_subsampling_rate": effective_subsampling_rate,
        "num_lines_to_sample": closest_num_lines,
        "effective_num_lines": effective_num_lines,
        "factors": factors,
    }


def path_with_respect_to_user_paths(base_path, path, split_folder="pretrained"):
    """Small helper function to offset path with respect to user paths.
    Will use "pretrained" folder as split point.
    """
    base_path = Path(base_path)
    path = str(Path(path))

    assert base_path.name == split_folder, (
        f"Base path last folder {base_path.name} should be {split_folder}"
    )

    split_folder_idx = path.find(split_folder)

    if split_folder_idx == -1:
        raise ValueError(f"Split folder {split_folder} not found in path {path}")

    return base_path.parent / path[split_folder_idx:]


def get_masks_for_volumes(
    image_shape,
    subsampling_rate,
    line_width,
    num_frames,
):
    # total_num_lines should be the elevation dimension (image_shape[1]), not image_shape[0]
    total_num_lines = image_shape[1]  # number of elevation planes
    subsampling_params = determine_subsampling_params(
        total_num_lines=total_num_lines,
        subsampling_rate=subsampling_rate,
        line_width=line_width,
    )

    effective_subsampling_rate = subsampling_params["effective_subsampling_rate"]
    effective_num_lines = subsampling_params["effective_num_lines"]

    if effective_subsampling_rate != subsampling_rate:
        log.warning(
            f"Initially requested factor {subsampling_rate} was not possible. "
            f"Using {effective_subsampling_rate} instead (closest valid factor: {subsampling_params['num_lines_to_sample']})."
        )
    subsampling_rate = float(effective_subsampling_rate)

    log.info(
        f"Subsampling rate: {log.green(subsampling_rate)}, with line width {log.green(line_width)}"
    )
    log.info(
        f"which equals {log.green(effective_num_lines)} lines out of {log.green(total_num_lines)}"
    )

    equispaced_sampler = EquispacedLines(
        img_height=image_shape[0],
        img_width=image_shape[1],
        n_actions=subsampling_params["num_lines_to_sample"],
        n_possible_actions=image_shape[1] // line_width,
        batch_size=1,
    )

    masks = []
    # loop over frames which offsets the mask
    lines = None
    for _ in range(num_frames):
        if lines is None:
            lines, mask = equispaced_sampler.initial_sample_stateless()
        else:
            lines, mask = equispaced_sampler.sample_stateless(lines)
        masks.append(mask)

    masks = ops.concatenate(masks)
    # frames, azimuth, depth, elevation
    masks = ops.expand_dims(masks, axis=1)

    assert int(ops.sum(masks[0, 0, 0])) == effective_num_lines, (
        f"Mask {int(ops.sum(masks[0, 0, 0]))} does not match effective_num_lines "
        f"{effective_num_lines}"
    )

    # [frames, azimuth, depth, elevation, 1]
    masks = ops.expand_dims(masks, axis=-1)

    return masks, subsampling_params, subsampling_rate, effective_num_lines
