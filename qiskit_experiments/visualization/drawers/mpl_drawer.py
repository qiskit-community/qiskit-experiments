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
                        if self.figure_options.ylabel:
                            label = self.figure_options.ylabel
                            if isinstance(label, list):
                                # Y label can be given as a list for each sub axis
                                label = label[i]
                            sub_ax.set_ylabel(label, fontsize=self.style["axis_label_size"])
                    if i != n_rows - 1:
                        # remove x axis except for most-bottom plot
                        sub_ax.set_xticklabels([])
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

        # Add data labels if there are multiple labels registered per sub_ax.
        for sub_ax in all_axes:
            _, labels = sub_ax.get_legend_handles_labels()
            if len(labels) > 1:
                sub_ax.legend(loc=self.style["legend_loc"])

        # Format x and y axis
        for ax_type in ("x", "y"):
            # Get axis formatter from drawing options
            if ax_type == "x":
                lim = self.figure_options.xlim
                unit = self.figure_options.xval_unit
                unit_scale = self.figure_options.xval_unit_scale
            else:
                lim = self.figure_options.ylim
                unit = self.figure_options.yval_unit
                unit_scale = self.figure_options.yval_unit_scale

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
            if unit and unit_scale:
                # If value is specified, automatically scale axis magnitude
                # and write prefix to axis label, i.e. 1e3 Hz -> 1 kHz
                maxv = max(np.abs(lim[0]), np.abs(lim[1]))
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
                # get_shared_y_axes() is immutable from matplotlib>=3.6.0. Must use Axis.sharey()
                # instead, but this can only be called once per axis. Here we call sharey  on all axes in
                # a chain, which should have the same effect.
                if len(all_axes) > 1:
                    for ax1, ax2 in zip(all_axes[1:], all_axes[0:-1]):
                        ax1.sharex(ax2)
                all_axes[0].set_xlim(lim)
            else:
                # get_shared_y_axes() is immutable from matplotlib>=3.6.0. Must use Axis.sharey()
                # instead, but this can only be called once per axis. Here we call sharey  on all axes in
                # a chain, which should have the same effect.
                if len(all_axes) > 1:
                    for ax1, ax2 in zip(all_axes[1:], all_axes[0:-1]):
                        ax1.sharey(ax2)
                all_axes[0].set_ylim(lim)
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
            "linestyle": "-",
            "linewidth": 2,
        }
        self._update_label_in_options(draw_ops, name, label, legend)
        draw_ops.update(**options)
        self._get_axis(axis).plot(x_data, y_data, **draw_ops)

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
