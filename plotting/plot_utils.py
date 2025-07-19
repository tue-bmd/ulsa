import re
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from zea import log


def natural_sort(l: list):
    """Sort a list of strings or numbers in a natural order.
    This means that '10' comes after '2', and 'file1' comes before 'file10'."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", str(key))]
    return sorted(l, key=alphanum_key)


class ViolinPlotter:
    def __init__(
        self,
        group_names=None,
        group_colors=None,
        xlabel="X",
        ylabel="Metric",
        file_ext="pdf",
        legend_loc="best",
        legend_bbox=None,
        figsize=None,
        context=None,
        legend_position="top",  # "top" or "right"
        scatter_kwargs=None,
        violin_kwargs=None,
    ):
        """
        group_names: dict mapping group key to display name
        group_colors: dict mapping group key to color
        context: matplotlib style context (e.g., "styles/darkmode.mplstyle" or None)
        legend_position: "top" or "right"
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
        self.legend_position = legend_position
        if scatter_kwargs is None:
            scatter_kwargs = {
                "s": 20,
                "alpha": 0.2,
            }
        self.scatter_kwargs = scatter_kwargs
        self.violin_kwargs = violin_kwargs or {
            "showmeans": True,
            "showextrema": False,
        }

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
        legend_position=None,  # Allow override per plot
        **kwargs,
    ):
        """
        data_dict: dict[group][x_value] = list of metric values
        context: matplotlib style context (overrides self.context if provided)
        legend_position: "top" or "right" (overrides self.legend_position if provided)
        """
        plot_context = context if context is not None else self.context
        legend_position = (
            legend_position if legend_position is not None else self.legend_position
        )
        with (
            plt.style.context(plot_context)
            if plot_context is not None
            else nullcontext()
        ):
            self._plot_core(
                data_dict,
                save_path,
                x_label_values,
                metric_name,
                groups_to_plot,
                legend_position,
                **kwargs,
            )

    def _plot_core(
        self,
        data_dict,
        save_path,
        x_label_values=None,
        metric_name=None,
        groups_to_plot=None,
        legend_position="top",
        ylim=None,
    ):
        fig = plt.figure(figsize=self.figsize)

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

        width = 0.5
        if groups_to_plot is None:
            groups_to_plot = list(data_dict.keys())
        if len(groups_to_plot) == 2:
            # Bring violins closer together for 2 groups
            group_offset = np.linspace(-width / 4, width / 4, 2)
        else:
            group_offset = np.linspace(-width / 2, width / 2, len(groups_to_plot))

        # Calculate group means and order them
        group_order = {}
        for group in groups_to_plot:
            all_values = []
            for x_val in all_x_values:
                if x_val in data_dict.get(group, {}):
                    values = data_dict[group][x_val]
                    if isinstance(values, (list, np.ndarray)):
                        try:
                            # in case values array is inhomogenous
                            flat_values = [item for sublist in values for item in sublist]
                            # flat_values = [np.mean(l) for l in values]
                            metric_values = np.array(flat_values, dtype=np.float64)
                            all_values.extend(metric_values)
                        except (ValueError, TypeError):
                            continue
            if all_values:
                group_order[group] = np.mean(all_values)

        sorted_groups = sorted(
            group_order.keys(), key=lambda x: group_order[x], reverse=True
        )

        # Plot violins in order
        for group_idx, group in enumerate(sorted_groups):
            violin_positions = []
            violin_data = []

            for x_val in all_x_values:
                try:
                    values = data_dict.get(group, {}).get(x_val, [])
                    if isinstance(values, (list, np.ndarray)):
                        flat_values = [item for sublist in values for item in sublist]
                        # flat_values = [np.mean(l) for l in values]
                        metric_values = np.array(flat_values, dtype=np.float64)
                        if metric_values.size > 0:
                            pos = x_value_to_pos[x_val] + group_offset[group_idx]
                            violin_positions.append(pos)
                            violin_data.append(metric_values)
                except (KeyError, ValueError, TypeError):
                    continue

            group_colors = self.get_group_colors(sorted_groups)
            if violin_data:
                parts = plt.violinplot(
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
                    plt.scatter(
                        [pos] * len(data),
                        data,
                        color=group_color,
                        **self.scatter_kwargs,
                    )

                plt.scatter(
                    [],
                    [],
                    color=group_color,
                    label=f"{self.group_names.get(group, group)}",
                )

        # Customize plot
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel if metric_name is None else metric_name)
        if legend_position == "right":
            fig.legend(
                loc="outside center left",
                frameon=False,
            )
        else:  # "top" or default
            fig.legend(
                loc="outside upper center",
                ncol=2,
                frameon=False,
            )
        plt.grid(True, alpha=0.3)
        plt.xticks(plot_positions, [str(x) for x in all_x_values])
        if ylim:
            plt.ylim(ylim)

        plt.savefig(save_path)
        plt.close()
        log.info(f"Saved violin plot to {log.yellow(save_path)}")


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
        figsize = figsize if figsize is not None else self.figsize or (8, 2.5 * len(x_label_values or []))

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
                            hist_vals, _ = np.histogram(metric_values, bins=bin_edges, density=density)
                            hist_densities.append(hist_vals.max() if len(hist_vals) else 0)
                            if kde and metric_values.size > 1:
                                kde_est = gaussian_kde(metric_values)
                                x_grid = np.linspace(bin_edges[0], bin_edges[-1], 200)
                                kde_vals = kde_est(x_grid)
                                kde_densities.append(kde_vals.max() if len(kde_vals) else 0)
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
            fig, axes = plt.subplots(
                K, 1, figsize=figsize, sharex=True
            )
            if K == 1:
                axes = [axes]
            for i, x_val in enumerate(all_x_values):
                ax = axes[i]
                for group in groups_to_plot:
                    values = data_dict.get(group, {}).get(x_val, [])
                    if isinstance(values, (list, np.ndarray)):
                        try:
                            flat_values = [item for sublist in values for item in sublist]
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
                                    x_grid = np.linspace(bin_edges[0], bin_edges[-1], 200)
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
                ax.set_ylabel(inner_y_label or ("Density" if density else "Count"), fontsize=10, labelpad=8)
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
                    ax.set_xlabel(self.xlabel if metric_name is None else metric_name, fontsize=14)
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
                fig.text(0.01, 0.5, outer_y_label or self.ylabel, va='center', rotation='vertical', fontsize=18)
            plt.tight_layout(rect=(0.02, 0, 1, 1))
            plt.savefig(save_path)
            plt.close()
            log.info(f"Saved overlapping histogram stack to {log.yellow(save_path)}")