# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Curve drawer for matplotlib backend."""

import numbers
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import tab10
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import Formatter, ScalarFormatter
from qiskit.exceptions import QiskitError
from qiskit.utils import detach_prefix

from qiskit_experiments.framework.matplotlib import get_non_gui_ax

from ..utils import ExtentTuple
from .base_drawer import BaseDrawer, SeriesName


class MplDrawer(BaseDrawer):
    """Drawer for MatplotLib backend."""

    DefaultMarkers = MarkerStyle.filled_markers
    DefaultColors = tab10.colors

    class PrefixFormatter(Formatter):
        """Matplotlib axis formatter to detach prefix.

        If a value is, e.g., x=1000.0 and the factor is 1000, then it will be shown
        as 1.0 in the ticks and its unit will be shown with the prefactor 'k'
        in the axis label.
        """

        def __init__(self, factor: float):
            """Create a PrefixFormatter instance.

            Args:
                factor: factor by which to scale tick values.
            """
            self.factor = factor

        def __call__(self, x: Any, pos: int = None) -> str:
            """Returns the formatted string for tick position ``pos`` and value ``x``.

            Args:
                x: the tick value to format.
                pos: the tick label position.

            Returns:
                The formatted tick label.
            """
            return self.fix_minus(f"{x * self.factor:.3g}")

    def __init__(self):
        super().__init__()
        # Used to track which series have already been plotted. Needed for _get_default_marker and
        # _get_default_color.
        self._series = []

    def initialize_canvas(self):
        # Create axis if empty
        if not self.options.axis:
            axis = get_non_gui_ax()
            figure = axis.get_figure()
            figure.set_size_inches(*self.style["figsize"])
        else:
            axis = self.options.axis

        sharex = self.figure_options.sharex
        sharey = self.figure_options.sharey

        n_rows, n_cols = self.options.subplots
        n_subplots = n_cols * n_rows
        if n_subplots > 1:
            # Add inset axis. User may provide a single axis object via the analysis option,
            # while this analysis tries to draw its result in multiple canvases,
            # especially when the analysis consists of multiple curves.
            # Inset axis is experimental implementation of matplotlib 3.0 so maybe unstable API.
            # This draws inset axes with shared x and y axis.
            if (
                self.figure_options.get("custom_style", {}).get("style_name") == "residuals"
                and n_subplots != 2
            ):
                # raising an error for residual plotting that isn't on individual plot per figure.
                raise QiskitError(
                    "Residual plots and residual plotting style is supported for "
                    "figures with one sub-plot only."
                )

            inset_ax_h_list = self.figure_options.custom_style.get(
                "sub_plot_heights_list", [1 / n_rows] * n_rows
            )
            inset_ax_w_list = self.figure_options.custom_style.get(
                "sub_plot_widths_list", [1 / n_cols] * n_cols
            )

            # Check that the heights and widths are lists.
            if (not isinstance(inset_ax_h_list, List)) or (not isinstance(inset_ax_w_list, List)):
                raise QiskitError(
                    "Sub-plots heights and widths list need to be a list of floats that sum"
                    " up to 1"
                )

            # adding a check for correct sizes of subplots.
            if not np.isclose(sum(inset_ax_h_list), 1) or not np.isclose(sum(inset_ax_w_list), 1):
                raise QiskitError(
                    "The subplots aren't covering all the figure. "
                    "Check subplots heights and widths configurations."
                )

            # setting the row tracker.
            sum_heights = 0
            for i, inset_ax_h in enumerate(inset_ax_h_list):
                # updating row tracker.
                sum_heights += inset_ax_h

                # setting column tracker.
                sum_widths = 0

                for j, inset_ax_w in enumerate(inset_ax_w_list):
                    # x0, y0, width, height
                    bounds = [
                        sum_widths,
                        1 - sum_heights,
                        inset_ax_w,
                        inset_ax_h,
                    ]

                    sub_ax = axis.inset_axes(bounds, transform=axis.transAxes, zorder=1)
                    if j != 0 and sharey:
                        # remove y axis except for most-left plot
                        sub_ax.yaxis.set_tick_params(labelleft=False)
                    else:
                        # this axis locates at left, write y-label
                        if self.figure_options.ylabel:
                            label = self.figure_options.ylabel
                            if isinstance(label, list):
                                # Y label can be given as a list for each sub axis
                                label = label[i]
                            sub_ax.set_ylabel(label, fontsize=self.style["axis_label_size"])
                    if i != n_rows - 1 and sharex:
                        # remove x axis except for most-bottom plot
                        sub_ax.xaxis.set_tick_params(labelleft=False)
                    else:
                        # this axis locates at bottom, write x-label
                        if self.figure_options.xlabel:
                            label = self.figure_options.xlabel
                            if isinstance(label, list):
                                # X label can be given as a list for each sub axis
                                label = label[j]
                            sub_ax.set_xlabel(label, fontsize=self.style["axis_label_size"])
                    if j == 0 or i == n_rows - 1:
                        # Set label size for outer axes where labels are drawn
                        sub_ax.tick_params(labelsize=self.style["tick_label_size"])
                    sub_ax.grid()

                    # updating where we are on the grid.
                    sum_widths += inset_ax_w

            # Remove original axis frames
            axis.axis("off")
        else:
            axis.set_xlabel(self.figure_options.xlabel, fontsize=self.style["axis_label_size"])
            axis.set_ylabel(self.figure_options.ylabel, fontsize=self.style["axis_label_size"])
            axis.tick_params(labelsize=self.style["tick_label_size"])
            axis.grid()

        self._axis = axis

    def format_canvas(self):
        if self._axis.child_axes:
            # Multi canvas mode
            all_axes = self._axis.child_axes
        else:
            all_axes = [self._axis]

        # Set axes scale. This needs to be done before anything tries to work with
        # the axis limits because if no limits or data are set explicitly the
        # default limits depend on the scale method (for example, the minimum
        # value is 0 for linear scaling but not for log scaling).
        def signed_sqrt(x):
            return np.sign(x) * np.sqrt(abs(x))

        def signed_square(x):
            return np.sign(x) * x**2

        for ax_type in ("x", "y"):
            for sub_ax in all_axes:
                scale = self.figure_options.get(f"{ax_type}scale")
                if ax_type == "x":
                    mpl_setscale = sub_ax.set_xscale
                else:
                    mpl_setscale = sub_ax.set_yscale

                # Apply non linear axis spacing
                if scale is not None:
                    if scale == "quadratic":
                        mpl_setscale("function", functions=(signed_square, signed_sqrt))
                    else:
                        mpl_setscale(scale)

        # Get axis formatter from drawing options
        formatter_opts = {}
        for ax_type in ("x", "y"):
            limit = self.figure_options.get(f"{ax_type}lim")
            unit = self.figure_options.get(f"{ax_type}val_unit")
            unit_scale = self.figure_options.get(f"{ax_type}val_unit_scale")

            # Format options to a list for each axis
            if limit is None or isinstance(limit[0], numbers.Number):
                limit = [limit] * len(all_axes)
            if unit is None or isinstance(unit, str):
                unit = [unit] * len(all_axes)
            if isinstance(unit_scale, bool):
                unit_scale = [unit_scale] * len(all_axes)

            # Compute min-max value for auto scaling
            min_vals = []
            max_vals = []
            for sub_ax in all_axes:
                if ax_type == "x":
                    min_v, max_v = sub_ax.get_xlim()
                else:
                    min_v, max_v = sub_ax.get_ylim()
                min_vals.append(min_v)
                max_vals.append(max_v)

            formatter_opts[ax_type] = {
                "limit": limit,
                "unit": unit,
                "unit_scale": unit_scale,
                "min_ax_vals": min_vals,
                "max_ax_vals": max_vals,
            }

        for i, sub_ax in enumerate(all_axes):
            # Add data labels if there are multiple labels registered per sub_ax.
            _, labels = sub_ax.get_legend_handles_labels()
            if len(labels) > 1:
                sub_ax.legend(loc=self.style["legend_loc"])

            for ax_type in ("x", "y"):
                limit = formatter_opts[ax_type]["limit"][i]
                unit = formatter_opts[ax_type]["unit"][i]
                unit_scale = formatter_opts[ax_type]["unit_scale"][i]
                min_ax_vals = formatter_opts[ax_type]["min_ax_vals"]
                max_ax_vals = formatter_opts[ax_type]["max_ax_vals"]
                share_axis = self.figure_options.get(f"share{ax_type}")

                if ax_type == "x":
                    mpl_axis_obj = getattr(sub_ax, "xaxis")
                    mpl_setlimit = sub_ax.set_xlim
                    mpl_share = sub_ax.sharex
                else:
                    mpl_axis_obj = getattr(sub_ax, "yaxis")
                    mpl_setlimit = sub_ax.set_ylim
                    mpl_share = sub_ax.sharey

                if limit is None:
                    if share_axis:
                        limit = min(min_ax_vals), max(max_ax_vals)
                    else:
                        limit = min_ax_vals[i], max_ax_vals[i]

                # Create formatter for axis tick label notation
                if unit and unit_scale:
                    # If value is specified, automatically scale axis magnitude
                    # and write prefix to axis label, i.e. 1e3 Hz -> 1 kHz
                    maxv = max(np.abs(limit[0]), np.abs(limit[1]))
                    try:
                        scaled_maxv, prefix = detach_prefix(maxv, decimal=3)
                        prefactor = scaled_maxv / maxv
                    except ValueError:
                        prefix = ""
                        prefactor = 1
                    formatter = MplDrawer.PrefixFormatter(prefactor)
                    units_str = f" [{prefix}{unit}]"
                else:
                    # Use scientific notation with 3 digits, 1000 -> 1e3
                    formatter = ScalarFormatter()
                    formatter.set_scientific(True)
                    formatter.set_powerlimits((-3, 3))
                    units_str = f" [{unit}]" if unit else ""
                mpl_axis_obj.set_major_formatter(formatter)

                # Add units to axis label if both exist
                if units_str:
                    label_txt_obj = mpl_axis_obj.get_label()
                    label_str = label_txt_obj.get_text()
                    if label_str:
                        label_txt_obj.set_text(label_str + units_str)

                # Consider axis sharing among subplots
                if share_axis:
                    if i == 0:
                        # Limit is set to the first axis only.
                        mpl_setlimit(limit)
                    else:
                        # get_shared_*_axes() is immutable from matplotlib>=3.6.0.
                        # Must use Axis.share*() instead, but this can only be called once per axis.
                        # Here we call share* on all axes in a chain, which should have the same effect.
                        mpl_share(all_axes[i - 1])
                else:
                    mpl_setlimit(limit)

        # Add title
        if self.figure_options.figure_title is not None:
            self._axis.set_title(
                label=self.figure_options.figure_title,
                fontsize=self.style["axis_label_size"],
            )

    def _get_axis(self, index: Optional[int] = None) -> Axes:
        """A helper method to get inset axis.

        Args:
            index: Index of inset axis. If nothing is provided, it returns the entire
                axis.

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

    def _get_default_color(self, name: SeriesName) -> Tuple[float, ...]:
        """A helper method to get default color for the series.

        Args:
            name: Name of the series.

        Returns:
            Default color available in matplotlib.
        """
        if self.figure_options.get("custom_style", {}).get("style_name") == "residuals":
            if name[: -len("_residuals")] in self._series:
                name = name[: -len("_residuals")]

        if name not in self._series:
            self._series.append(name)

        ind = self._series.index(name) % len(self.DefaultColors)
        return self.DefaultColors[ind]

    def _get_default_marker(self, name: SeriesName) -> str:
        """A helper method to get default marker for the scatter plot.

        Args:
            name: Name of the series.

        Returns:
            Default marker available in matplotlib.
        """
        if name not in self._series:
            self._series.append(name)

        ind = self._series.index(name) % len(self.DefaultMarkers)
        return self.DefaultMarkers[ind]

    def _update_label_in_options(
        self,
        options: Dict[str, any],
        name: Optional[SeriesName],
        label: Optional[str] = None,
        legend: bool = False,
    ):
        """Helper function to set the label entry in ``options`` based on given
        arguments.

        This method uses :meth:`label_for` to get the label for the series identified by
        ``name``. If :meth:`label_for` returns ``None``, then
        ``_update_label_in_options`` doesn't add a `"label"` entry into ``options``.
        I.e., a label entry is added to ``options`` only if it is not ``None``.

        Args:
            options: The options dictionary being modified.
            name: The name of the series being labelled. Used as a fall-back label if
                ``label`` is None and no label exists in ``series_params`` for this
                series.
            label: Optional legend label to override ``name`` and ``series_params``.
            legend: Whether a label entry should be added to ``options``. Used as an
                easy toggle to disable adding a label entry. Defaults to False.
        """
        if legend:
            _label = self.label_for(name, label)
            if _label:
                options["label"] = _label

    def scatter(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        x_err: Optional[Sequence[float]] = None,
        y_err: Optional[Sequence[float]] = None,
        name: Optional[SeriesName] = None,
        label: Optional[str] = None,
        legend: bool = False,
        **options,
    ):

        series_params = self.figure_options.series_params.get(name, {})
        marker = series_params.get("symbol", self._get_default_marker(name))
        color = series_params.get("color", self._get_default_color(name))
        axis = series_params.get("canvas", None)

        draw_options = {
            "color": color,
            "marker": marker,
            "alpha": 0.8,
            "zorder": 2,
        }
        self._update_label_in_options(draw_options, name, label, legend)
        draw_options.update(**options)

        if x_err is None and y_err is None:
            # Size of symbols is defined by the `s` kwarg for scatter(). Check if `s` exists in
            # `draw_options`, if not set to the default style. Square the `symbol_size` as `s` for MPL
            # scatter is proportional to the width and not the area of the marker, but `symbol_size` is
            # proportional to the area.
            if "s" not in draw_options:
                draw_options["s"] = self.style["symbol_size"] ** 2
            self._get_axis(axis).scatter(x_data, y_data, **draw_options)
        else:
            # Check for invalid error values.
            if y_err is not None and not np.all(np.isfinite(y_err)):
                y_err = None
            if x_err is not None and not np.all(np.isfinite(x_err)):
                x_err = None

            # `errorbar` has extra default draw_options to set, but we want to accept any overrides from
            # `options`, and thus draw_options.
            errorbar_options = {
                "linestyle": "",
                # `markersize` is equivalent to `symbol_size`.
                "markersize": self.style["symbol_size"],
                "capsize": self.style["errorbar_capsize"],
            }
            errorbar_options.update(draw_options)

            self._get_axis(axis).errorbar(
                x_data, y_data, yerr=y_err, xerr=x_err, **errorbar_options
            )

    def line(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        name: Optional[SeriesName] = None,
        label: Optional[str] = None,
        legend: bool = False,
        **options,
    ):
        series_params = self.figure_options.series_params.get(name, {})
        axis = series_params.get("canvas", None)
        color = series_params.get("color", self._get_default_color(name))

        draw_ops = {
            "color": color,
            "linestyle": series_params.get("linestyle", "-"),
            "linewidth": series_params.get("linewidth", 2),
        }
        self._update_label_in_options(draw_ops, name, label, legend)
        draw_ops.update(**options)
        self._get_axis(axis).plot(x_data, y_data, **draw_ops)

    def hline(
        self,
        y_value: float,
        name: Optional[SeriesName] = None,
        label: Optional[str] = None,
        legend: bool = False,
        **options,
    ):
        series_params = self.figure_options.series_params.get(name, {})
        axis = series_params.get("canvas", None)
        color = series_params.get("color", self._get_default_color(name))

        draw_ops = {
            "color": color,
            "linestyle": series_params.get("linestyle", "-"),
            "linewidth": series_params.get("linewidth", 2),
        }
        self._update_label_in_options(draw_ops, name, label, legend)
        draw_ops.update(**options)
        self._get_axis(axis).axhline(y_value, **draw_ops)

    def filled_y_area(
        self,
        x_data: Sequence[float],
        y_ub: Sequence[float],
        y_lb: Sequence[float],
        name: Optional[SeriesName] = None,
        label: Optional[str] = None,
        legend: bool = False,
        **options,
    ):
        series_params = self.figure_options.series_params.get(name, {})
        axis = series_params.get("canvas", None)
        color = series_params.get("color", self._get_default_color(name))

        draw_ops = {
            "alpha": 0.1,
            "color": color,
        }
        self._update_label_in_options(draw_ops, name, label, legend)
        draw_ops.update(**options)
        self._get_axis(axis).fill_between(x_data, y1=y_lb, y2=y_ub, **draw_ops)

    def filled_x_area(
        self,
        x_ub: Sequence[float],
        x_lb: Sequence[float],
        y_data: Sequence[float],
        name: Optional[SeriesName] = None,
        label: Optional[str] = None,
        legend: bool = False,
        **options,
    ):
        series_params = self.figure_options.series_params.get(name, {})
        axis = series_params.get("canvas", None)
        color = series_params.get("color", self._get_default_color(name))

        draw_ops = {
            "alpha": 0.1,
            "color": color,
        }
        self._update_label_in_options(draw_ops, name, label, legend)
        draw_ops.update(**options)
        self._get_axis(axis).fill_betweenx(y_data, x1=x_lb, x2=x_ub, **draw_ops)

    def textbox(
        self,
        description: str,
        rel_pos: Optional[Tuple[float, float]] = None,
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

        if rel_pos is None:
            rel_pos = self.style["textbox_rel_pos"]

        text_box_handler = self._axis.text(
            *rel_pos,
            s=description,
            ha="center",
            va="top",
            size=self.style["textbox_text_size"],
            transform=self._axis.transAxes,
            zorder=1000,  # Very large zorder to draw over other graphics.
        )
        text_box_handler.set_bbox(bbox_props)

    def _series_names_to_cmap(
        self, series_names: List[SeriesName]
    ) -> Tuple[Colormap, Dict[str, float]]:
        """Create a :class:`Colormap` instance of series colours.

        This method creates a :class:`Colormap` instance that can be used to plot an
        image of series classifications: i.e., a 2D array of series names. The returned
        Colormap positions the series colours, from :meth:`_get_default_color`, along
        the range :math:`0` to :math:`1`. The returned dictionary contains mappings from
        series names (``Union[str, int, float]``) to floats which are used to "sample"
        from the Colormap.

        Example:
            .. code-block:: python

                # 2D array of classification strings, where each value is a series name.
                data_classification = np.array(..., dtype=str)

                # Get Colormap and its key.
                series_names = [...]
                cmap,cmap_map = self._series_names_to_cmap(series_names)

                # Convert classified data into float data.
                data_float = np.vectorize(
                    lambda x: cmap(cmap_map[x])
                )(data_classification)

                # Plot float data with Colormap.
                plt.imshow(data_float, cmap=cmap, ...)

        Args:
            series_names: List of series names.

        Returns:
            A tuple ``(cmap, map)`` where ``cmap`` is a Matplotlib Colormap instance and
            ``map`` is a dictionary that maps series names (dictionary keys) to floats
            (dictionary values) that identify the series names' colours in ``cmap``.
        """
        # Remove duplicates from series_names, just in-case. Use dict.fromkeys to preserve order and
        # remove duplicates.
        unique_series_names = list(dict.fromkeys(series_names))

        # Generate list of colours by calling querying series_params and self._get_default_color(name).
        colours = []
        for series_name in unique_series_names:
            series_params = self.figure_options.series_params.get(series_name, {})
            colour = series_params.get("color", self._get_default_color(series_name))
            colours.append(colour)

        # Create CMap.
        cmap = LinearSegmentedColormap.from_list(
            "SeriesMap",
            colours,
        )

        # Create a dictionary to lookup the floating-point value for each series name.
        series_cmap_idx = dict(
            zip(unique_series_names, np.linspace(0, 1, len(unique_series_names)))
        )

        return cmap, series_cmap_idx

    def image(
        self,
        data: np.ndarray,
        extent: Optional[ExtentTuple] = None,
        name: Optional[SeriesName] = None,
        label: Optional[str] = None,
        cmap: Optional[Union[str, Any]] = None,
        cmap_use_series_colors: bool = False,
        colorbar: bool = False,
        **options,
    ):
        series_params = self.figure_options.series_params.get(name, {})
        axis = series_params.get("canvas", None)

        if len(data.shape) == 3:
            if data.shape[-1] != 3 and data.shape[-1] != 4:
                raise QiskitError("Image data is three-dimensional but is not RGB/A data.")

        # Register extent in ops.
        image_ops = {
            "extent": extent,
        }

        # Apply cmap and get image data to be plotted.
        if cmap_use_series_colors and len(data.shape) == 2:
            series_names = np.unique(data).tolist()
            _cmap, cmap_series_map = self._series_names_to_cmap(series_names)
            img = np.vectorize(lambda x: cmap_series_map[x])(data)
            image_ops["cmap"] = _cmap
        else:
            img = data
            if cmap:
                image_ops["cmap"] = cmap

        # Apply kwargs passed in options.
        image_ops.update(**options)

        mapping = self._get_axis(axis).imshow(img, **image_ops)

        # Create colorbar if requested.
        if colorbar:
            colorbar_label = series_params.get("label", label if label is not None else str(name))
            self._get_axis(axis).figure.colorbar(mapping, label=colorbar_label)

    @property
    def figure(self) -> Figure:
        """Return figure object handler to be saved in the database.

        In the MatplotLib the ``Figure`` and ``Axes`` are different object. User can
        pass a part of the figure (i.e. multi-axes) to the drawer option ``axis``. For
        example, a user wants to combine two different experiment results in the same
        figure, one can call ``pyplot.subplots`` with two rows and pass one of the
        generated two axes to each experiment drawer. Once all the experiments complete,
        the user will obtain the single figure collecting all experimental results.

        Note that this method returns the entire figure object, rather than a single
        axis. Thus, the experiment data saved in the database might have a figure
        collecting all child axes drawings.
        """
        return self._axis.get_figure()
