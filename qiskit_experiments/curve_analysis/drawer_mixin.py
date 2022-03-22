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

"""Curve drawing functionality for analysis."""

import functools
from typing import Callable, Set, List, Dict, Sequence, Union, Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from qiskit.utils import detach_prefix
from uncertainties import unumpy as unp, UFloat

from qiskit_experiments.curve_analysis.curve_data import FitData
from qiskit_experiments.framework import AnalysisResultData, Options
from qiskit_experiments.framework.matplotlib import get_non_gui_ax


class CurveDrawerMixin:
    """Mixin to provide drawing functionalities for curve fit result."""

    # Internal state for drawing options
    _draw_options: Options
    _set_draw_options: Set

    def __init_subclass__(cls, **kwargs):
        """Initialize drawing options."""
        super().__init_subclass__(**kwargs)

        # This is a hack to let the mixin have internal state for drawing option.
        @functools.wraps(cls.__init__, assigned=("__annotations__",))
        def __new__(sub_cls, *args, **kwargs):
            instance = super(cls, sub_cls).__new__(sub_cls, *args, **kwargs)
            instance._draw_options = instance._default_draw_options()
            instance._set_draw_options = set()

            return instance

        cls.__new__ = __new__

    def _initialize_canvas(self) -> Axes:
        """Initialize matplotlib axis.

        Curve analysis drawer supports 2D sub-graph visualization.
        This first checks the drawing options if matplotlib axis is provided by users.
        Axis is initialized only when axis is not provided.
        Initialized axis is set to the drawing options.

        Returns:
            Handler to matplotlib axis object. This may be initialized with inset axes.
        """
        # Create axis if empty
        if not self.draw_options.axis:
            axis = get_non_gui_ax()

            # Set figure size if axis is not provided.
            # Otherwise user may set the figure size by oneself.
            # Then avoid applying draw defaults to avoid override user's preference.
            figure = axis.get_figure()
            figure.set_size_inches(*self.draw_options.figsize)
        else:
            axis = self.draw_options.axis

        n_rows, n_cols = self.draw_options.subplots
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
                        if self.draw_options.ylabel:
                            label = self.draw_options.ylabel
                            if isinstance(label, list):
                                # Y label can be given as a list for each sub axis
                                label = label[i]
                            sub_ax.set_ylabel(label, fontsize=self.draw_options.axis_label_size)
                    if i != n_rows - 1:
                        # remove x axis except for most-bottom plot
                        sub_ax.set_xticklabels([])
                    else:
                        # this axis locates at bottom, write x-label
                        if self.draw_options.xlabel:
                            label = self.draw_options.xlabel
                            if isinstance(label, list):
                                # X label can be given as a list for each sub axis
                                label = label[j]
                            sub_ax.set_xlabel(label, fontsize=self.draw_options.axis_label_size)
                    if j == 0 or i == n_rows - 1:
                        # Set label size for outer axes where labels are drawn
                        sub_ax.tick_params(labelsize=self.draw_options.tick_label_size)
                    sub_ax.grid()

            # Remove original axis frames
            axis.axis("off")
        else:
            axis.set_xlabel(self.draw_options.xlabel, fontsize=self.draw_options.axis_label_size)
            axis.set_ylabel(self.draw_options.ylabel, fontsize=self.draw_options.axis_label_size)
            axis.tick_params(labelsize=self.draw_options.tick_label_size)
            axis.grid()

        return axis

    @staticmethod
    def _get_axis(axis: Axes, index: Optional[int]) -> Axes:
        """Helper function to get axis from instance option.

        Args:
            axis: Matplotlib axis object to draw on.
            index: Index of canvas if multiple axes are initialized.

        Returns:
            Axis to visualize analysis data.

        Raises:
            IndexError: When axis index is specified but no inset axis is found.
        """
        if index is not None:
            try:
                return axis.child_axes[index]
            except IndexError as ex:
                raise IndexError(
                    f"Canvas index {index} is out of range. "
                    f"Only {len(axis.child_axes)} subplots are initialized."
                ) from ex
        else:
            return axis

    def _format_axis(self, axis: Axes):
        """Format axis labels.

        This method updates axis labels and tick labels.
        If the physical unit is specified for the axis values, it detaches prefix
        to scale axis values, for example, 1000.0 [Hz] will be shown as 1.0 [kHz].

        If more than two labels are specified in the single axis,
        it also creates a legend object to show labels.

        Args:
            axis: Matplotlib axis object to draw on.
        """
        if axis.child_axes:
            # Multi canvas mode
            all_axes = axis.child_axes
        else:
            all_axes = [axis]

        # Add data labels if there are multiple labels registered per sub_ax.
        for sub_ax in all_axes:
            _, labels = sub_ax.get_legend_handles_labels()
            if len(labels) > 1:
                sub_ax.legend()

        # Format x and y axis
        for ax_type in ("x", "y"):
            # Get axis formatter from drawing options
            if ax_type == "x":
                lim = self.draw_options.xlim
                unit = self.draw_options.xval_unit
            else:
                lim = self.draw_options.ylim
                unit = self.draw_options.yval_unit

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

    @staticmethod
    def _generate_fit_report(analysis_results: List[AnalysisResultData]) -> str:
        """A helper method that generates fit reports documentation, i.e. list of
        parameter name - value pair that is saved in the analysis result.

        Args:
            analysis_results: List of data entries.

        Returns:
            Documentation of fit reports.
        """

        def _format_val(value, unit):
            """Return value with unit with prefix, i.e. 1000 Hz -> 1 kHz."""
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

        analysis_description = ""
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
                analysis_description += f"{res.name} = {value_repr}\n"

        return analysis_description

    def _draw_fit_report(
        self,
        axis: Axes,
        analysis_results: List[AnalysisResultData],
        chisq: Union[float, Dict[str, float]],
    ):
        """Draw text box that shows fit reports.

        Args:
            axis: Matplotlib axis object to draw on.
            analysis_results: List of analysis result entries containing fit parameters.
            chisq: Chi-squared value from the fitting. If this is provided as a dictionary,
                the key is also shown with the chi-squared value.
        """
        # Write fit parameters
        report_str = self._generate_fit_report(analysis_results)

        # Write reduced chi-squared values
        if isinstance(chisq, float):
            report_str += r"Fit $\chi^2$ = " + f"{chisq: .4g}"
        else:
            chisq_repr = [
                r"Fit $\chi^2$ = " + f"{val: .4g} ({name})" for name, val in chisq.items()
            ]
            report_str += "\n".join(chisq_repr)

        report_handler = axis.text(
            *self.draw_options.fit_report_rpos,
            s=report_str,
            ha="center",
            va="top",
            size=self.draw_options.fit_report_text_size,
            transform=axis.transAxes,
        )
        bbox_props = dict(boxstyle="square, pad=0.3", fc="white", ec="black", lw=1, alpha=0.8)
        report_handler.set_bbox(bbox_props)

    def _draw_raw_data(
        self,
        axis: Axes,
        x_data: Sequence[float],
        y_data: Sequence[float],
        axis_ind: Optional[int] = None,
        **custom_opts,
    ):
        """Draw scatter plot for raw data.

        Args:
            axis: Matplotlib axis object to draw on.
            x_data: X values.
            y_data: Y values.
            axis_ind: Index of canvas if multi-axes mode is triggered.
            custom_opts: Matplotlib scatter plot options.
        """
        draw_options = {
            "color": "grey",
            "marker": "x",
            "alpha": 0.8,
            "zorder": 2,
        }
        draw_options.update(**custom_opts)
        self._get_axis(axis, axis_ind).scatter(x_data, y_data, **draw_options)

    def _draw_formatted_data(
        self,
        axis: Axes,
        x_data: Sequence[float],
        y_data: Sequence[float],
        y_err_data: Sequence[float],
        name: Optional[str] = None,
        axis_ind: Optional[int] = None,
        **custom_opts,
    ):
        """Draw error bar plot for formatted data that used for fitting.

        Args:
            axis: Matplotlib axis object to draw on.
            x_data: X values.
            y_data: Y values.
            y_err_data: Standard deviation of Y values.
            axis_ind: Index of canvas if multi-axes mode is triggered.
            custom_opts: Matplotlib scatter plot options.
        """
        draw_ops = {
            "markersize": 9,
            "alpha": 0.8,
            "zorder": 4,
            "linestyle": "",
        }
        draw_ops.update(**custom_opts)
        if name:
            draw_ops["label"] = name
        self._get_axis(axis, axis_ind).errorbar(x_data, y_data, yerr=y_err_data, **draw_ops)

    def _draw_fit_lines(
        self,
        axis: Axes,
        fit_function: Callable,
        signature: List[str],
        fit_result: FitData,
        fixed_params: Dict[str, float],
        axis_ind: Optional[int] = None,
        **custom_opts,
    ):
        """Draw lines and intervals for fit data.

        Args:
            axis: Matplotlib axis object to draw on.
            fit_function: The function defines a single curve.
            signature: The fit parameters associated with the function.
            fit_result: The result of fit.
            fixed_params: The parameter fixed during the fitting.
            axis_ind: Index of canvas if multi-axes mode is triggered.
            custom_opts: Matplotlib scatter plot options.
        """
        draw_ops = {
            "markersize": 9,
            "zorder": 5,
            "linestyle": "-",
            "linewidth": 2,
        }
        draw_ops.update(**custom_opts)

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

        axis = self._get_axis(axis, axis_ind)

        # Plot fit line
        axis.plot(xs, unp.nominal_values(ys_with_sigma), **draw_ops)

        # Plot confidence interval
        sigmas = unp.std_devs(ys_with_sigma)
        if np.isfinite(sigmas).all():
            for n_sigma, alpha in self.draw_options.plot_sigma:
                axis.fill_between(
                    xs,
                    y1=unp.nominal_values(ys_with_sigma) - n_sigma * sigmas,
                    y2=unp.nominal_values(ys_with_sigma) + n_sigma * sigmas,
                    alpha=alpha,
                    zorder=3,
                    color=draw_ops.get("color", "black"),
                )

    @property
    def draw_options(self) -> Options:
        """Return the analysis options for drawing method."""
        return self._draw_options

    @classmethod
    def _default_draw_options(cls) -> Options:
        """Return default draw options.
        Draw Options:
            axis (AxesSubplot): A matplotlib axis object to draw.
            subplots (Tuple[int, int]): Number of rows and columns if multi-canvas plot.
            xlabel (str): X label of fit result figure.
            ylabel (str): Y label of fit result figure.
            xlim (Tuple[float, float]): Min and max value of horizontal axis of the fit plot.
            ylim (Tuple[float, float]): Min and max value of vertical axis of the fit plot.
            xval_unit (str): SI unit of x values. No prefix is needed here.
                For example, when the x values represent time, this option will be just "s"
                rather than "ms". In the fit result plot, the prefix is automatically selected
                based on the maximum value. If your x values are in [1e-3, 1e-4], they
                are displayed as [1 ms, 10 ms]. This option is likely provided by the
                analysis class rather than end-users. However, users can still override
                if they need different unit notation. By default, this option is set to ``None``,
                and no scaling is applied. X axis will be displayed in the scientific notation.
            yval_unit (str): Unit of y values. Same as ``xval_unit``.
                This value is not provided in most experiments, because y value is usually
                population or expectation values.
            figsize (Tuple[int, int]): Size of figure (width, height).
            legend_loc (str): Vertical and horizontal location of the curve label window in
                strings separated by a space.
            tick_label_size (int): Text size of tick label.
            axis_label_size (int): Text size of label size.
            fit_report_rpos (Tuple[int, int]): Relative horizontal and vertical location
                of the fit report window.
            fit_report_text_size (int): Text size of fit report.
            plot_raw_data: Set ``True`` to visualize raw measurement result.
            plot_sigma (List[Tuple[float, float]]): Sigma values for fit confidence interval,
                which are the tuple of (n_sigma, alpha). The alpha indicates
                the transparency of the corresponding interval plot.
        """
        return Options(
            axis=None,
            subplots=(1, 1),
            xlabel=None,
            ylabel=None,
            xlim=None,
            ylim=None,
            xval_unit=None,
            yval_unit=None,
            figsize=(8, 5),
            legend_loc="center right",
            tick_label_size=14,
            axis_label_size=16,
            fit_report_rpos=(0.6, 0.95),
            fit_report_text_size=14,
            plot_raw_data=False,
            plot_sigma=[(1.0, 0.7), (3.0, 0.3)],
        )

    def set_draw_options(self, **fields):
        """Set the analysis options for drawing method.
        Args:
            fields: The fields to update the options
        """
        self._draw_options.update_options(**fields)
        self._set_draw_options = self._set_draw_options.union(fields)
