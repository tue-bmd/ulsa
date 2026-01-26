"""Utility functions for plotting ULSA results with matplotlib."""

import re
from collections import OrderedDict
from collections.abc import Iterable
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from zea import log

METRIC_NAMES = {
    "dice": "DICE (→) [-]",
    "psnr": "PSNR (→) [dB]",
    "ssim": "SSIM (→) [-]",
    "lpips": "LPIPS (←) [-]",
    "mse": "MSE (←) [-]",  # on [0, 1] scale
    "rmse": "RMSE (←) [-]",  # on [0, 1] scale
    "nrmse": "NRMSE (←) [-]",
    "gcnr": "gCNR (→) [-]",
}

STRATEGY_COLORS = {
    "downstream_propagation_summed": "#d62728",  # Red
    "greedy_entropy": "#1f77b4",  # Blue
    "equispaced": "#2ca02c",  # Green
    "uniform_random": "#ff7f0e",  # Orange
}

STRATEGY_NAMES = {
    "downstream_propagation_summed": "Measurement Information Gain",
    "greedy_entropy": "Cognitive",
    "uniform_random": "Random",
    "equispaced": "Equispaced",
}


def get_axis_label(key, axis_label_map: dict):
    """Get friendly label for axis keys."""
    base_key = key.split(".")[-1]
    return axis_label_map.get(base_key, base_key.replace("_", " ").title())


def natural_sort(l: list):
    """Sort a list of strings or numbers in a natural order.
    This means that '10' comes after '2', and 'file1' comes before 'file10'."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", str(key))]
    return sorted(l, key=alphanum_key)


class ViolinPlotter:
    def __init__(
        self,
        group_names: dict = None,
        group_colors: dict = None,
        xlabel: str = "X",
        ylabel: str = "Metric",
        file_ext: str = "pdf",
        legend_loc: str = "best",
        legend_bbox=None,
        figsize: tuple = None,
        context: str = None,
        scatter_kwargs: dict = None,
        violin_kwargs: dict = None,
    ):
        """
        group_names: dict mapping group key to display name
        group_colors: dict mapping group key to color
        context: matplotlib style context (e.g., "styles/darkmode.mplstyle" or None)
        """
        self.group_names = group_names or {}
        self._group_colors = group_colors
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.file_ext = file_ext
        self.legend_loc = legend_loc
        self.legend_bbox = legend_bbox
        self.figsize = figsize
        self.context = context
        self.scatter_kwargs = scatter_kwargs
        self.violin_kwargs = violin_kwargs or {
            "showmeans": True,
            "showextrema": False,
        }

        # Given points
        x = np.array([1, 500, 1e12])  # 1e12 as a proxy for infinity
        y = np.array([0.9, 0.01, 1e-9])

        # Assume c = 0 (since at infinity, alpha = 0)
        # So: alpha = a / (n_points ** b)
        # Take logs to linearize: log(alpha) = log(a) - b*log(n_points)
        log_x = np.log(x)
        log_y = np.log(y)
        A = np.vstack([np.ones_like(log_x), -log_x]).T
        sol, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
        log_a, b = sol
        a = np.exp(log_a)

        def alpha_func(n):
            return a / (n**b)

        self._scale_alpha_func = alpha_func

    def _get_scatterer_kwargs(self, n_points):
        if self.scatter_kwargs is not None:
            return self.scatter_kwargs
        return {"alpha": self._scale_alpha_func(n_points), "s": 4}

    def get_group_colors(self, keys):
        """
        Generate a list of colors for the groups, cycling through group_colors if needed.
        """
        if not self._group_colors:
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            return {k: color_cycle[i % len(color_cycle)] for i, k in enumerate(keys)}
        else:
            # Use provided group colors, and fallback to gray
            return {k: self._group_colors.get(k, "#333333") for i, k in enumerate(keys)}

    def plot(
        self,
        data_dict,
        save_path,
        x_label_values=None,
        metric_name=None,
        groups_to_plot=None,
        context=None,
        legend_kwargs="default",
        ax=None,
        **kwargs,
    ):
        """
        data_dict: dict[group][x_value] = list of metric values
        context: matplotlib style context (overrides self.context if provided)
        """
        plot_context = context if context is not None else self.context
        with (
            plt.style.context(plot_context)
            if plot_context is not None
            else nullcontext()
        ):
            fig = None
            if ax is None:
                fig, ax = plt.subplots(figsize=self.figsize)
            self._plot_core(
                ax,
                data_dict,
                x_label_values,
                metric_name,
                groups_to_plot,
                **kwargs,
            )
            if legend_kwargs == "default":
                legend_kwargs = {
                    "loc": "outside upper center",
                    "ncol": 2,
                    "frameon": False,
                }
            if legend_kwargs is not None and fig is not None:
                fig.legend(**legend_kwargs)

            if save_path is not None:
                plt.savefig(save_path)
                plt.close()
                log.info(f"Saved violin plot to {log.yellow(save_path)}")

    def _order_groups_by_means(
        self, data_dict, groups_to_plot, all_x_values, reverse=True
    ):
        """
        Orders groups by their overall mean metric value across all x values.

        Parameters:
            data_dict (dict): Dictionary of the form data_dict[group][x_value] = list of metric values.
            groups_to_plot (list): List of group keys to consider.
            all_x_values (list): List of x values to aggregate over.
            reverse (bool): If True, sort groups in descending order of mean (highest first).

        Returns:
            list: Sorted list of group keys by mean metric value.
        """
        group_order = {}
        for group in groups_to_plot:
            all_values = []
            for x_val in all_x_values:
                if x_val in data_dict.get(group, {}):
                    values = data_dict[group][x_val]
                    if isinstance(values, (list, np.ndarray)):
                        try:
                            # check for list of lists
                            if isinstance(values[0], Iterable) and not isinstance(
                                values[0], (str, bytes)
                            ):
                                # in case values array is inhomogenous
                                flat_values = [
                                    item for sublist in values for item in sublist
                                ]
                            else:
                                flat_values = values
                            metric_values = np.array(flat_values, dtype=np.float64)
                            all_values.extend(metric_values)
                        except (ValueError, TypeError):
                            print("ViolinPlotter: Error parsing list valued results")
                            continue
            if all_values:
                group_order[group] = np.mean(all_values)

        sorted_groups = sorted(
            group_order.keys(), key=lambda x: group_order[x], reverse=reverse
        )
        return sorted_groups

    def _plot_core(
        self,
        ax: plt.Axes,
        data_dict,
        x_label_values=None,
        metric_name=None,
        groups_to_plot=None,
        ylim=None,
        order_by="mean",
        reverse_order=True,
        width=0.5,
    ):
        # Collect and sort x values
        if x_label_values is None:
            all_x_values = set()
            for group in data_dict:
                all_x_values.update(data_dict[group].keys())
        else:
            all_x_values = x_label_values
        all_x_values = natural_sort(all_x_values)

        # Create equally spaced positions
        plot_positions = np.arange(len(all_x_values))
        x_value_to_pos = dict(zip(all_x_values, plot_positions))

        if groups_to_plot is None:
            groups_to_plot = list(data_dict.keys())
        if len(groups_to_plot) == 2:
            # Bring violins closer together for 2 groups
            group_offset = np.linspace(-width / 4, width / 4, 2)
        else:
            group_offset = np.linspace(-width / 2, width / 2, len(groups_to_plot))

        # Calculate group means and order them (optional)
        if order_by == "mean":
            sorted_groups = self._order_groups_by_means(
                data_dict, groups_to_plot, all_x_values, reverse=reverse_order
            )
        elif order_by is None:
            sorted_groups = groups_to_plot
        elif not isinstance(order_by, str):
            sorted_groups = order_by
        else:
            raise ValueError(
                f"Invalid order_by value: {order_by}. Must be 'mean', None, or a list of groups."
            )

        # Plot violins in order
        for group_idx, group in enumerate(sorted_groups):
            violin_positions = []
            violin_data = []

            for x_val in all_x_values:
                try:
                    values = data_dict.get(group, {}).get(x_val, [])
                    if isinstance(values, (list, np.ndarray)):
                        if isinstance(values[0], Iterable) and not isinstance(
                            values[0], (str, bytes)
                        ):
                            # in case values array is inhomogenous
                            flat_values = [
                                item for sublist in values for item in sublist
                            ]
                        else:
                            flat_values = values
                        metric_values = np.array(flat_values, dtype=np.float64)
                        if metric_values.size > 0:
                            pos = x_value_to_pos[x_val] + group_offset[group_idx]
                            violin_positions.append(pos)
                            violin_data.append(metric_values)
                except (KeyError, ValueError, TypeError):
                    continue

            group_colors = self.get_group_colors(sorted_groups)
            if violin_data:
                parts = ax.violinplot(
                    violin_data,
                    positions=violin_positions,
                    widths=width / 4,
                    **self.violin_kwargs,
                )

                # Use consistent colors from group_colors
                group_color = group_colors[group]
                for pc in parts["bodies"]:
                    pc.set_facecolor(group_color)
                    pc.set_alpha(0.7)
                if "cbars" in parts:
                    parts["cbars"].set_color(group_color)
                if "cmins" in parts:
                    parts["cmins"].set_color(group_color)
                if "cmaxes" in parts:
                    parts["cmaxes"].set_color(group_color)
                if "cmeans" in parts:
                    parts["cmeans"].set_color("#000000")
                    parts["cmeans"].set_alpha(0.4)

                # Add scatter points with same color
                for pos, data in zip(violin_positions, violin_data):
                    ax.scatter(
                        [pos] * len(data),
                        data,
                        color=group_color,
                        **self._get_scatterer_kwargs(len(data)),
                    )

                # Add legend entry
                ax.scatter(
                    [],
                    [],
                    color=group_color,
                    label=f"{self.group_names.get(group, group)}",
                )

        # Customize plot
        if self.xlabel is not None:
            ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            ax.set_ylabel(self.ylabel if metric_name is None else metric_name)
        elif metric_name is not None:
            ax.set_ylabel(metric_name)
        ax.grid()
        ax.set_xticks(plot_positions, [str(x) for x in all_x_values])
        if ylim:
            ax.set_ylim(ylim)

        return ax


class OverlappingHistogramPlotter:
    def __init__(
        self,
        group_names=None,
        group_colors=None,
        xlabel="Metric",
        ylabel="x_value",
        file_ext="pdf",
        figsize=None,
        context=None,
        bins=30,
        alpha=0.4,
        kde=True,
        kde_lw=2,
        density=True,  # <--- NEW: default to density
    ):
        self.group_names = group_names or {}
        self._group_colors = group_colors
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.file_ext = file_ext
        self.figsize = figsize
        self.context = context
        self.bins = bins
        self.alpha = alpha
        self.kde = kde
        self.kde_lw = kde_lw
        self.density = density

    def get_group_colors(self, keys):
        if not self._group_colors:
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            return {k: color_cycle[i % len(color_cycle)] for i, k in enumerate(keys)}
        else:
            return {k: self._group_colors.get(k, "#333333") for i, k in enumerate(keys)}

    def plot_overlapping_histograms_by_xvalue(
        self,
        data_dict,
        save_path,
        metric_name=None,
        groups_to_plot=None,
        context=None,
        x_label_values=None,
        bins=None,
        alpha=None,
        kde=None,
        figsize=None,
        outer_y_label=None,
        inner_y_label=None,
        bin_edges=None,
        density=None,
    ):
        plot_context = context if context is not None else self.context
        bins = bins if bins is not None else self.bins
        alpha = alpha if alpha is not None else self.alpha
        kde = kde if kde is not None else self.kde
        density = self.density if density is None else density
        figsize = (
            figsize
            if figsize is not None
            else self.figsize or (8, 2.5 * len(x_label_values or []))
        )

        # Collect all x values
        if x_label_values is None:
            all_x_values = set()
            for group in data_dict:
                all_x_values.update(data_dict[group].keys())
        else:
            all_x_values = x_label_values
        all_x_values = natural_sort(all_x_values)
        K = len(all_x_values)

        if groups_to_plot is None:
            groups_to_plot = list(data_dict.keys())
        group_colors = self.get_group_colors(groups_to_plot)

        # Find global min/max for shared x-axis and collect all values for binning
        all_metric_values = []
        for group in groups_to_plot:
            for x_val in all_x_values:
                values = data_dict.get(group, {}).get(x_val, [])
                if isinstance(values, (list, np.ndarray)):
                    try:
                        flat_values = [item for sublist in values for item in sublist]
                        metric_values = np.array(flat_values, dtype=np.float64)
                        all_metric_values.extend(metric_values)
                    except Exception:
                        continue
        if not all_metric_values:
            raise ValueError("No data to plot.")
        global_min = np.min(all_metric_values)
        global_max = np.max(all_metric_values)
        if bin_edges is None:
            bin_edges = np.linspace(global_min, global_max, bins + 1)

        # --- First pass: compute global max density for all histograms and KDEs ---
        global_max_density = 0
        hist_densities = []
        kde_densities = []
        for i, x_val in enumerate(all_x_values):
            for group in groups_to_plot:
                values = data_dict.get(group, {}).get(x_val, [])
                if isinstance(values, (list, np.ndarray)):
                    try:
                        flat_values = [item for sublist in values for item in sublist]
                        metric_values = np.array(flat_values, dtype=np.float64)
                        if metric_values.size > 0:
                            hist_vals, _ = np.histogram(
                                metric_values, bins=bin_edges, density=density
                            )
                            hist_densities.append(
                                hist_vals.max() if len(hist_vals) else 0
                            )
                            if kde and metric_values.size > 1:
                                kde_est = gaussian_kde(metric_values)
                                x_grid = np.linspace(bin_edges[0], bin_edges[-1], 200)
                                kde_vals = kde_est(x_grid)
                                kde_densities.append(
                                    kde_vals.max() if len(kde_vals) else 0
                                )
                    except Exception:
                        continue
        if hist_densities or kde_densities:
            global_max_density = max(hist_densities + kde_densities)
        else:
            global_max_density = 1

        # --- Plotting ---
        with (
            plt.style.context(plot_context)
            if plot_context is not None
            else nullcontext()
        ):
            fig, axes = plt.subplots(K, 1, figsize=figsize, sharex=True)
            if K == 1:
                axes = [axes]
            for i, x_val in enumerate(all_x_values):
                ax = axes[i]
                for group in groups_to_plot:
                    values = data_dict.get(group, {}).get(x_val, [])
                    if isinstance(values, (list, np.ndarray)):
                        try:
                            flat_values = [
                                item for sublist in values for item in sublist
                            ]
                            metric_values = np.array(flat_values, dtype=np.float64)
                            if metric_values.size > 0:
                                ax.hist(
                                    metric_values,
                                    bins=bin_edges,
                                    color=group_colors[group],
                                    alpha=alpha,
                                    label=self.group_names.get(group, group),
                                    density=density,
                                    edgecolor="none",
                                )
                                if kde and metric_values.size > 1:
                                    kde_est = gaussian_kde(metric_values)
                                    x_grid = np.linspace(
                                        bin_edges[0], bin_edges[-1], 200
                                    )
                                    ax.plot(
                                        x_grid,
                                        kde_est(x_grid),
                                        color=group_colors[group],
                                        lw=self.kde_lw,
                                        alpha=0.7,
                                    )
                        except Exception:
                            continue
                ax.set_xlim(bin_edges[0], bin_edges[-1])
                ax.set_ylim(0, global_max_density * 1.05)
                # Set small repeated y-label (inner)
                ax.set_ylabel(
                    inner_y_label or ("Density" if density else "Count"),
                    fontsize=10,
                    labelpad=8,
                )
                # Annotate the x_value to the left of the subplot, vertically centered
                ax.annotate(
                    f"{x_val}",
                    xy=(0, 0.5),
                    xycoords=("axes fraction", "axes fraction"),
                    xytext=(-50, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                )
                if i == 0:
                    ax.legend()
                if i == K - 1:
                    ax.set_xlabel(
                        self.xlabel if metric_name is None else metric_name, fontsize=14
                    )
                    tick_step = max(1, len(bin_edges) // 10)
                    ax.set_xticks(bin_edges[::tick_step])
                    ax.set_xticklabels([f"{v:.1f}" for v in bin_edges[::tick_step]])
                else:
                    ax.set_xticklabels([])
                ax.grid(True, axis="x", alpha=0.3)
            # Add a single large y-axis label for all subplots (outer)
            if hasattr(fig, "supylabel"):
                fig.supylabel(outer_y_label or self.ylabel, fontsize=18)
            else:
                fig.text(
                    0.01,
                    0.5,
                    outer_y_label or self.ylabel,
                    va="center",
                    rotation="vertical",
                    fontsize=18,
                )
            plt.tight_layout(rect=(0.02, 0, 1, 1))
            plt.savefig(save_path)
            plt.close()
            log.info(f"Saved overlapping histogram stack to {log.yellow(save_path)}")


def get_inset(
    fig: plt.Figure,
    ax0: plt.Axes,
    ax1: plt.Axes,
    shape: tuple,
    height: float = 0.5,  # quantile of ax height
    y_offset: float = 0.1,  # quantile offset
) -> plt.Axes:
    """
    Add an inset image to a matplotlib ImageGrid plot.

    Example:
    ```python
    from mpl_toolkits.axes_grid1 import ImageGrid

    fig = plt.figure()
    axs = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.1)
    axs[0].imshow(targets, **kwargs)
    axs[1].imshow(reconstructions, **kwargs)
    inset_ax = get_inset(
        fig, axs[0], axs[1], measurements.shape, height=0.2, y_offset=0.05
    )
    inset_ax.imshow(measurements, **kwargs)
    ```
    """
    assert 0 < height <= 1, "height must be in (0, 1]"

    axpos0 = ax0.get_position()
    axpos1 = ax1.get_position()
    axheight = axpos0.ymax - axpos0.ymin

    absolute_height = height * axheight
    aspect_ratio = shape[1] / shape[0]
    w = absolute_height * aspect_ratio

    x_center = (axpos0.xmin + axpos0.xmax + axpos1.xmin + axpos1.xmax) / 4
    x = x_center - w / 2
    y = axpos0.ymax - absolute_height + y_offset * axheight

    inset_ax = fig.add_axes([x, y, w, absolute_height])
    inset_ax.axis("off")
    return inset_ax


def add_progress_bar(arr, bar_height=5, bar_value=255):
    """
    Adds a horizontal progress bar at the bottom of a (frames, h, w) array.
    The bar fills from left to right as frames progress.

    Parameters:
        arr (np.ndarray): Input array of shape (frames, h, w).
        bar_height (int): Height of the bar in pixels.
        bar_value (int or float): Value to fill the bar (e.g., 255 for uint8).

    Returns:
        np.ndarray: Array with the progress bar added.
    """
    frames, h, w = arr.shape
    out = arr.copy()
    for i in range(frames):
        fill_width = int((i + 1) / frames * w)
        out[i, h - bar_height : h, :fill_width] = bar_value
    return out


def write_roman(num):
    roman = OrderedDict()
    roman[1000] = "M"
    roman[900] = "CM"
    roman[500] = "D"
    roman[400] = "CD"
    roman[100] = "C"
    roman[90] = "XC"
    roman[50] = "L"
    roman[40] = "XL"
    roman[10] = "X"
    roman[9] = "IX"
    roman[5] = "V"
    roman[4] = "IV"
    roman[1] = "I"

    def roman_num(num):
        for r in roman.keys():
            x, y = divmod(num, r)
            yield roman[r] * x
            num -= r * x
            if num <= 0:
                break

    return "".join([a for a in roman_num(num)])
