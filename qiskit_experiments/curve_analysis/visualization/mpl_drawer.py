# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Curve drawer for matplotlib backend."""

from typing import Sequence, Optional, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter, Formatter
from matplotlib.cm import tab10
from matplotlib.markers import MarkerStyle

from qiskit.utils import detach_prefix
from qiskit_experiments.framework.matplotlib import get_non_gui_ax

from .base_drawer import BaseCurveDrawer


class MplCurveDrawer(BaseCurveDrawer):
    """Curve drawer for MatplotLib backend."""

    DefaultMarkers = MarkerStyle().filled_markers
    DefaultColors = tab10.colors

    class PrefixFormatter(Formatter):
        """Matplotlib axis formatter to detach prefix.

        If a value is, e.g., x=1000.0 and the factor is 1000, then it will be shown
        as 1.0 in the ticks and its unit will be shown with the prefactor 'k'
        in the axis label.
        """

        def __init__(self, factor: float):
            self.factor = factor

        def __call__(self, x, pos=None):
            return self.fix_minus("{:.3g}".format(x * self.factor))

    def initialize_canvas(self):
        # Create axis if empty
        if not self.options.axis:
            axis = get_non_gui_ax()
            figure = axis.get_figure()
            figure.set_size_inches(*self.options.figsize)
        else:
            axis = self.options.axis

        n_rows, n_cols = self.options.subplots
        n_subplots = n_cols * n_rows
        if n_subplots > 1:
            # Add inset axis. User may provide a single axis object via the analysis option,
            # while this analysis tries to draw its result in multiple canvases,
            # especially when the analysis consists of multiple curves.
            # Inset axis is experimental implementation of matplotlib 3.0 so maybe unstable API.
            # This draws inset axes with shared x and y axis.
            inset_ax_h = 1 / n_rows
            inset_ax_w = 1 / n_cols
            for i in range(n_rows):
                for j in range(n_cols):
                    # x0, y0, width, height
                    bounds = [
                        inset_ax_w * j,
                        1 - inset_ax_h * (i + 1),
                        inset_ax_w,
                        inset_ax_h,
                    ]
                    sub_ax = axis.inset_axes(bounds, transform=axis.transAxes, zorder=1)
                    if j != 0:
                        # remove y axis except for most-left plot
                        sub_ax.set_yticklabels([])
                    else:
                        # this axis locates at left, write y-label
                        if self.options.ylabel:
                            label = self.options.ylabel
                            if isinstance(label, list):
                                # Y label can be given as a list for each sub axis
                                label = label[i]
                            sub_ax.set_ylabel(label, fontsize=self.options.axis_label_size)
                    if i != n_rows - 1:
                        # remove x axis except for most-bottom plot
                        sub_ax.set_xticklabels([])
                    else:
                        # this axis locates at bottom, write x-label
                        if self.options.xlabel:
                            label = self.options.xlabel
                            if isinstance(label, list):
                                # X label can be given as a list for each sub axis
                                label = label[j]
                            sub_ax.set_xlabel(label, fontsize=self.options.axis_label_size)
                    if j == 0 or i == n_rows - 1:
                        # Set label size for outer axes where labels are drawn
                        sub_ax.tick_params(labelsize=self.options.tick_label_size)
                    sub_ax.grid()

            # Remove original axis frames
            axis.axis("off")
        else:
            axis.set_xlabel(self.options.xlabel, fontsize=self.options.axis_label_size)
            axis.set_ylabel(self.options.ylabel, fontsize=self.options.axis_label_size)
            axis.tick_params(labelsize=self.options.tick_label_size)
            axis.grid()

        self._axis = axis

    def format_canvas(self):
        if self._axis.child_axes:
            # Multi canvas mode
            all_axes = self._axis.child_axes
        else:
            all_axes = [self._axis]

        # Add data labels if there are multiple labels registered per sub_ax.
        for sub_ax in all_axes:
            _, labels = sub_ax.get_legend_handles_labels()
            if len(labels) > 1:
                sub_ax.legend()

        # Format x and y axis
        for ax_type in ("x", "y"):
            # Get axis formatter from drawing options
            if ax_type == "x":
                lim = self.options.xlim
                unit = self.options.xval_unit
            else:
                lim = self.options.ylim
                unit = self.options.yval_unit

            # Compute data range from auto scale
            if not lim:
                v0 = np.nan
                v1 = np.nan
                for sub_ax in all_axes:
                    if ax_type == "x":
                        this_v0, this_v1 = sub_ax.get_xlim()
                    else:
                        this_v0, this_v1 = sub_ax.get_ylim()
                    v0 = np.nanmin([v0, this_v0])
                    v1 = np.nanmax([v1, this_v1])
                lim = (v0, v1)

            # Format axis number notation
            if unit:
                # If value is specified, automatically scale axis magnitude
                # and write prefix to axis label, i.e. 1e3 Hz -> 1 kHz
                maxv = max(np.abs(lim[0]), np.abs(lim[1]))
                try:
                    scaled_maxv, prefix = detach_prefix(maxv, decimal=3)
                    prefactor = scaled_maxv / maxv
                except ValueError:
                    prefix = ""
                    prefactor = 1

                formatter = MplCurveDrawer.PrefixFormatter(prefactor)
                units_str = f" [{prefix}{unit}]"
            else:
                # Use scientific notation with 3 digits, 1000 -> 1e3
                formatter = ScalarFormatter()
                formatter.set_scientific(True)
                formatter.set_powerlimits((-3, 3))

                units_str = ""

            for sub_ax in all_axes:
                if ax_type == "x":
                    ax = getattr(sub_ax, "xaxis")
                    tick_labels = sub_ax.get_xticklabels()
                else:
                    ax = getattr(sub_ax, "yaxis")
                    tick_labels = sub_ax.get_yticklabels()

                if tick_labels:
                    # Set formatter only when tick labels exist
                    ax.set_major_formatter(formatter)
                if units_str:
                    # Add units to label if both exist
                    label_txt_obj = ax.get_label()
                    label_str = label_txt_obj.get_text()
                    if label_str:
                        label_txt_obj.set_text(label_str + units_str)

            # Auto-scale all axes to the first sub axis
            if ax_type == "x":
                all_axes[0].get_shared_x_axes().join(*all_axes)
                all_axes[0].set_xlim(lim)
            else:
                all_axes[0].get_shared_y_axes().join(*all_axes)
                all_axes[0].set_ylim(lim)

        # Add title
        if self.options.figure_title is not None:
            self._axis.set_title(
                label=self.options.figure_title,
                fontsize=self.options.axis_label_size,
            )

    def _get_axis(self, index: Optional[int] = None) -> Axes:
        """A helper method to get inset axis.

        Args:
            index: Index of inset axis. If nothing is provided, it returns the entire axis.

        Returns:
            Corresponding axis object.

        Raises:
            IndexError: When axis index is specified but no inset axis is found.
        """
        if index is not None:
            try:
                return self._axis.child_axes[index]
            except IndexError as ex:
                raise IndexError(
                    f"Canvas index {index} is out of range. "
                    f"Only {len(self._axis.child_axes)} subplots are initialized."
                ) from ex
        else:
            return self._axis

    def _get_default_color(self, name: str) -> Tuple[float, ...]:
        """A helper method to get default color for the curve.

        Args:
            name: Name of the curve.

        Returns:
            Default color available in matplotlib.
        """
        if name not in self._curves:
            self._curves.append(name)

        ind = self._curves.index(name) % len(self.DefaultColors)
        return self.DefaultColors[ind]

    def _get_default_marker(self, name: str) -> str:
        """A helper method to get default marker for the scatter plot.

        Args:
            name: Name of the curve.

        Returns:
            Default marker available in matplotlib.
        """
        if name not in self._curves:
            self._curves.append(name)

        ind = self._curves.index(name) % len(self.DefaultMarkers)
        return self.DefaultMarkers[ind]

    def draw_raw_data(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        name: Optional[str] = None,
        **options,
    ):
        curve_opts = self.options.plot_options.get(name, {})
        marker = curve_opts.get("symbol", self._get_default_marker(name))
        axis = curve_opts.get("canvas", None)

        draw_options = {
            "color": "grey",
            "marker": marker,
            "alpha": 0.8,
            "zorder": 2,
        }
        draw_options.update(**options)
        self._get_axis(axis).scatter(x_data, y_data, **draw_options)

    def draw_formatted_data(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        y_err_data: Sequence[float],
        name: Optional[str] = None,
        **options,
    ):
        curve_opts = self.options.plot_options.get(name, {})
        axis = curve_opts.get("canvas", None)
        color = curve_opts.get("color", self._get_default_color(name))
        marker = curve_opts.get("symbol", self._get_default_marker(name))

        draw_ops = {
            "color": color,
            "marker": marker,
            "markersize": 9,
            "alpha": 0.8,
            "zorder": 4,
            "linestyle": "",
        }
        draw_ops.update(**options)
        if name:
            draw_ops["label"] = name

        if not np.all(np.isfinite(y_err_data)):
            y_err_data = None
        self._get_axis(axis).errorbar(x_data, y_data, yerr=y_err_data, **draw_ops)

    def draw_fit_line(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        name: Optional[str] = None,
        **options,
    ):
        curve_opts = self.options.plot_options.get(name, {})
        axis = curve_opts.get("canvas", None)
        color = curve_opts.get("color", self._get_default_color(name))

        draw_ops = {
            "color": color,
            "zorder": 5,
            "linestyle": "-",
            "linewidth": 2,
        }
        draw_ops.update(**options)
        self._get_axis(axis).plot(x_data, y_data, **draw_ops)

    def draw_confidence_interval(
        self,
        x_data: Sequence[float],
        y_ub: Sequence[float],
        y_lb: Sequence[float],
        name: Optional[str] = None,
        **options,
    ):
        curve_opts = self.options.plot_options.get(name, {})
        axis = curve_opts.get("canvas", None)
        color = curve_opts.get("color", self._get_default_color(name))

        draw_ops = {
            "zorder": 3,
            "alpha": 0.1,
            "color": color,
        }
        draw_ops.update(**options)
        self._get_axis(axis).fill_between(x_data, y1=y_lb, y2=y_ub, **draw_ops)

    def draw_fit_report(
        self,
        description: str,
        **options,
    ):
        bbox_props = {
            "boxstyle": "square, pad=0.3",
            "fc": "white",
            "ec": "black",
            "lw": 1,
            "alpha": 0.8,
        }
        bbox_props.update(**options)

        report_handler = self._axis.text(
            *self.options.fit_report_rpos,
            s=description,
            ha="center",
            va="top",
            size=self.options.fit_report_text_size,
            transform=self._axis.transAxes,
            zorder=6,
        )
        report_handler.set_bbox(bbox_props)

    @property
    def figure(self) -> Figure:
        """Return figure object handler to be saved in the database.

        In the MatplotLib the ``Figure`` and ``Axes`` are different object.
        User can pass a part of the figure (i.e. multi-axes) to the drawer option ``axis``.
        For example, a user wants to combine two different experiment results in the
        same figure, one can call ``pyplot.subplots`` with two rows and pass one of the
        generated two axes to each experiment drawer. Once all the experiments complete,
        the user will obtain the single figure collecting all experimental results.

        Note that this method returns the entire figure object, rather than a single axis.
        Thus, the experiment data saved in the database might have a figure
        collecting all child axes drawings.
        """
        return self._axis.get_figure()
