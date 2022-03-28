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

from typing import List, Dict, Sequence, Union, Optional, Callable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from qiskit.utils import detach_prefix
from uncertainties import unumpy as unp, UFloat

from qiskit_experiments.curve_analysis.curve_data import FitData
from qiskit_experiments.framework import AnalysisResultData
from qiskit_experiments.framework.matplotlib import get_non_gui_ax
from .base_drawer import BaseCurveDrawer


class MplCurveDrawer(BaseCurveDrawer):
    """Curve drawer for MatplotLib backend."""

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

                # pylint: disable=cell-var-from-loop
                formatter = FuncFormatter(lambda x, p: f"{x * prefactor: .3g}")

                # Add units to axis label if exist
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

    def draw_raw_data(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        ax_index: Optional[int] = None,
        **options,
    ):
        draw_options = {
            "color": "grey",
            "marker": "x",
            "alpha": 0.8,
            "zorder": 2,
        }
        draw_options.update(**options)
        self._get_axis(ax_index).scatter(x_data, y_data, **draw_options)

    def draw_formatted_data(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        y_err_data: Sequence[float],
        name: Optional[str] = None,
        ax_index: Optional[int] = None,
        **options,
    ):
        draw_ops = {
            "markersize": 9,
            "alpha": 0.8,
            "zorder": 4,
            "linestyle": "",
        }
        draw_ops.update(**options)
        if name:
            draw_ops["label"] = name
        self._get_axis(ax_index).errorbar(x_data, y_data, yerr=y_err_data, **draw_ops)

    def draw_fit_lines(
        self,
        fit_function: Callable,
        signature: List[str],
        fit_result: FitData,
        fixed_params: Dict[str, float],
        ax_index: Optional[int] = None,
        **options,
    ):
        draw_ops = {
            "markersize": 9,
            "zorder": 5,
            "linestyle": "-",
            "linewidth": 2,
        }
        draw_ops.update(**options)

        xmin, xmax = fit_result.x_range

        # This is ufloat parameters.
        parameters = {}
        for fitpar in signature:
            if fitpar in fixed_params:
                parameters[fitpar] = fixed_params[fitpar]
            else:
                parameters[fitpar] = fit_result.fitval(fitpar)

        # The confidence interval is automatically computed with uncertainty package.
        xs = np.linspace(xmin, xmax, 100)
        ys_with_sigma = fit_function(xs, **parameters)

        axis = self._get_axis(ax_index)

        # Plot fit line
        axis.plot(xs, unp.nominal_values(ys_with_sigma), **draw_ops)

        # Plot confidence interval
        sigmas = unp.std_devs(ys_with_sigma)
        if np.isfinite(sigmas).all():
            for n_sigma, alpha in self.options.plot_sigma:
                axis.fill_between(
                    xs,
                    y1=unp.nominal_values(ys_with_sigma) - n_sigma * sigmas,
                    y2=unp.nominal_values(ys_with_sigma) + n_sigma * sigmas,
                    alpha=alpha,
                    zorder=3,
                    color=draw_ops.get("color", "black"),
                )

    def draw_fit_report(
        self,
        analysis_results: List[AnalysisResultData],
        chisq: Union[float, Dict[str, float]],
    ):
        def _format_val(value, unit):
            # Return value with unit with prefix, i.e. 1000 Hz -> 1 kHz.
            if unit:
                try:
                    val, val_prefix = detach_prefix(value, decimal=3)
                except ValueError:
                    val = value
                    val_prefix = ""
                return f"{val: .3g}", f" {val_prefix}{unit}"
            if np.abs(value) < 1e-3 or np.abs(value) > 1e3:
                return f"{value: .4e}", ""
            return f"{value: .4g}", ""

        # Generate text inside the fit report
        report_str = ""
        for res in analysis_results:
            if isinstance(res.value, UFloat):
                unit = res.extra.get("unit", None)
                n_repr, n_unit = _format_val(res.value.nominal_value, unit)
                if res.value.std_dev is not None and np.isfinite(res.value.std_dev):
                    s_repr, s_unit = _format_val(res.value.std_dev, unit)
                    if n_unit == s_unit:
                        value_repr = f" {n_repr} \u00B1 {s_repr}{n_unit}"
                    else:
                        value_repr = f" {n_repr + n_unit} \u00B1 {s_repr + s_unit}"
                else:
                    value_repr = n_repr + n_unit
                report_str += f"{res.name} = {value_repr}\n"

        # Write reduced chi-squared values
        if isinstance(chisq, float):
            report_str += r"Fit $\chi^2$ = " + f"{chisq: .4g}"
        else:
            chisq_repr = [
                r"Fit $\chi^2$ = " + f"{val: .4g} ({name})" for name, val in chisq.items()
            ]
            report_str += "\n".join(chisq_repr)

        report_handler = self._axis.text(
            *self.options.fit_report_rpos,
            s=report_str,
            ha="center",
            va="top",
            size=self.options.fit_report_text_size,
            transform=self._axis.transAxes,
        )
        bbox_props = dict(boxstyle="square, pad=0.3", fc="white", ec="black", lw=1, alpha=0.8)
        report_handler.set_bbox(bbox_props)

    @property
    def figure(self) -> Figure:
        return self._axis.get_figure()
