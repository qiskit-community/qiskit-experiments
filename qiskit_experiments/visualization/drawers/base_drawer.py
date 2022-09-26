# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Drawer abstract class."""

from abc import ABC, abstractmethod
from typing import Dict, Sequence, Optional

from qiskit_experiments.framework import Options
from qiskit_experiments.visualization import PlotStyle


class BaseDrawer(ABC):
    """Abstract class for the serializable Qiskit Experiments figure drawer.

    A drawer may be implemented by different drawer backends such as matplotlib or Plotly. Sub-classes
    that wrap these backends by subclassing `BaseDrawer` must implement the following abstract methods.

    initialize_canvas

        This method should implement a protocol to initialize a drawer canvas with user input ``axis``
        object. Note that ``drawer`` supports visualization of experiment results in multiple canvases
        tiled into N (row) x M (column) inset grids, which is specified in the option ``subplots``. By
        default, this is N=1, M=1 and thus no inset grid will be initialized. The data points to draw
        might be provided with a canvas number defined in :attr:`SeriesDef.canvas` which defaults to
        ``None``, i.e. no-inset grids.

        This method should first check the drawer options (:attr:`options`) for the axis object and
        initialize the axis only when it is not provided by the options. Once axis is initialized, this
        is set to the instance member ``self._axis``.

    format_canvas

        This method formats the appearance of the canvas. Typically, it updates
        axis and tick labels. Note that the axis SI unit may be specified in the drawer options. In this
        case, axis numbers should be auto-scaled with the unit prefix.

    draw_raw_data

        This method draws raw experiment data points on the canvas, like a scatter-plot.

    draw_formatted_data

        This method plots data with error-bars for the y-values. The formatted data might be averaged
        over the same x values, or smoothed by a filtering algorithm, depending on how analysis class is
        implemented. This method is called with error bars of y values and the name of the series.

    draw_line

        This method plots a line from provided X and Y values. This method is typically called with
        interpolated x and y values from a curve-fit.

    draw_confidence_interval

        This method plots a shaped region bounded by upper and lower Y-values. This method is typically
        called with interpolated x and a pair of y values that represent the upper and lower bound within
        certain confidence interval. This might be called multiple times with different interval sizes.
        It is normally good to set some transparency for a confidence interval so the figure has enough
        contrast between points, lines, and the confidence-interval shape.

    draw_report

        This method draws a report on the canvas, which is a rectangular region containing some text.
        This method is typically called with a list of analysis results and reduced chi-squared values
        from a curve-fit.

    """

    def __init__(self):
        """Create a BaseDrawer instance."""
        # Normal options. Which includes the drawer axis, subplots, and default style.
        self._options = self._default_options()
        # A set of changed options for serialization.
        self._set_options = set()

        # Plot options which are typically updated by a plotter instance. Plot-options include the axis
        # labels, figure title, and a custom style instance.
        self._plot_options = self._default_plot_options()
        # A set of changed plot-options for serialization.
        self._set_plot_options = set()

        # The initialized axis/axes, set by `initialize_canvas`.
        self._axis = None

    @property
    def options(self) -> Options:
        """Return the drawer options."""
        return self._options

    @property
    def plot_options(self) -> Options:
        """Return the plot options.

        These are typically updated by a plotter instance, and thus may change. It is recommended to set
        plot options in a parent :class:`BasePlotter` instance that contains the :class:`BaseDrawer`
        instance.
        """
        return self._plot_options

    @classmethod
    def _default_options(cls) -> Options:
        """Return default drawer options.

        Drawer Options:
            axis (Any): Arbitrary object that can be used as a canvas.
            subplots (Tuple[int, int]): Number of rows and columns when the experimental
                result is drawn in the multiple windows.
            default_style (PlotStyle): The default style for drawer.
                This must contain all required style parameters for :class:`drawer`, as is defined in
                :meth:`PlotStyle.default_style()`. Subclasses can add extra required style parameters by
                overriding :meth:`_default_style`.
        """
        return Options(
            axis=None,
            subplots=(1, 1),
            default_style=cls._default_style(),
        )

    @classmethod
    def _default_style(cls) -> PlotStyle:
        return PlotStyle.default_style()

    @classmethod
    def _default_plot_options(cls) -> Options:
        """Return default plot options.

        Plot Options:
            xlabel (Union[str, List[str]]): X-axis label string of the output figure.
                If there are multiple columns in the canvas, this could be a list of labels.
            ylabel (Union[str, List[str]]): Y-axis label string of the output figure.
                If there are multiple rows in the canvas, this could be a list of labels.
            xlim (Tuple[float, float]): Min and max value of the horizontal axis.
                If not provided, it is automatically scaled based on the input data points.
            ylim (Tuple[float, float]): Min and max value of the vertical axis.
                If not provided, it is automatically scaled based on the input data points.
            xval_unit (str): SI unit of x values. No prefix is needed here.
                For example, when the x values represent time, this option will be just "s"
                rather than "ms". In the output figure, the prefix is automatically selected
                based on the maximum value in this axis. If your x values are in [1e-3, 1e-4],
                they are displayed as [1 ms, 10 ms]. This option is likely provided by the
                analysis class rather than end-users. However, users can still override
                if they need different unit notation. By default, this option is set to ``None``,
                and no scaling is applied. If nothing is provided, the axis numbers will be
                displayed in the scientific notation.
            yval_unit (str): Unit of y values. See ``xval_unit`` for details.
            figure_title (str): Title of the figure. Defaults to None, i.e. nothing is shown.
            series_params (Dict[str, Dict[str, Any]]): A dictionary of plot parameters for each series.
                This is keyed on the name for each series. Sub-dictionary is expected to have following
                three configurations, "canvas", "color", and "symbol"; "canvas" is the integer index of
                axis (when multi-canvas plot is set), "color" is the color of the series, and "symbol" is
                the marker style of the series. Defaults to an empty dictionary.
            custom_style (PlotStyle): The style definition to use when drawing. This overwrites style
                parameters in ``default_style`` in :attr:`options`. Defaults to an empty PlotStyle
                instance (i.e., :code-block:`PlotStyle()`).
        """
        return Options(
            xlabel=None,
            ylabel=None,
            xlim=None,
            ylim=None,
            xval_unit=None,
            yval_unit=None,
            figure_title=None,
            series_params={},
            custom_style=PlotStyle(),
        )

    def set_options(self, **fields):
        """Set the drawer options.
        Args:
            fields: The fields to update the options
        """
        self._options.update_options(**fields)
        self._set_options = self._set_options.union(fields)

    def set_plot_options(self, **fields):
        """Set the plot options.
        Args:
            fields: The fields to update the plot options
        """
        self._plot_options.update_options(**fields)
        self._set_plot_options = self._set_plot_options.union(fields)

    @property
    def style(self) -> PlotStyle:
        """The combined plot style for this drawer.

        The returned style instance is a combination of :attr:`options.default_style` and
        :attr:`plot_options.custom_style`. Style parameters set in ``custom_style`` override those set in
        ``default_style``. If ``custom_style`` is not an instance of :class:`PlotStyle`, the returned
        style is equivalent to ``default_style``.

        Returns:
            PlotStyle: The plot style for this drawer.
        """
        if isinstance(self.plot_options.custom_style, PlotStyle):
            return PlotStyle.merge(self.options.default_style, self.plot_options.custom_style)
        return self.options.default_style

    @abstractmethod
    def initialize_canvas(self):
        """Initialize the drawer canvas."""

    @abstractmethod
    def format_canvas(self):
        """Final cleanup for the canvas appearance."""

    @abstractmethod
    def draw_raw_data(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        name: Optional[str] = None,
        **options,
    ):
        """Draw raw data.

        Args:
            x_data: X values.
            y_data: Y values.
            name: Name of this series.
            options: Valid options for the drawer backend API.
        """

    @abstractmethod
    def draw_formatted_data(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        y_err_data: Sequence[float],
        name: Optional[str] = None,
        **options,
    ):
        """Draw the formatted data that is used for fitting.

        Args:
            x_data: X values.
            y_data: Y values.
            y_err_data: Standard deviation of Y values.
            name: Name of this series.
            options: Valid options for the drawer backend API.
        """

    @abstractmethod
    def draw_line(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        name: Optional[str] = None,
        **options,
    ):
        """Draw fit line.

        Args:
            x_data: X values.
            y_data: Fit Y values.
            name: Name of this series.
            options: Valid options for the drawer backend API.
        """

    @abstractmethod
    def draw_confidence_interval(
        self,
        x_data: Sequence[float],
        y_ub: Sequence[float],
        y_lb: Sequence[float],
        name: Optional[str] = None,
        **options,
    ):
        """Draw confidence interval.

        Args:
            x_data: X values.
            y_ub: The upper boundary of Y values.
            y_lb: The lower boundary of Y values.
            name: Name of this series.
            options: Valid options for the drawer backend API.
        """

    @abstractmethod
    def draw_report(
        self,
        description: str,
        **options,
    ):
        """Draw text box that shows reports, such as fit results.

        Args:
            description: A string to be drawn inside a report box.
            options: Valid options for the drawer backend API.
        """

    @property
    @abstractmethod
    def figure(self):
        """Return figure object handler to be saved in the database."""

    def config(self) -> Dict:
        """Return the config dictionary for this drawer."""
        options = dict((key, getattr(self._options, key)) for key in self._set_options)
        plot_options = dict(
            (key, getattr(self._plot_options, key)) for key in self._set_plot_options
        )

        return {
            "cls": type(self),
            "options": options,
            "plot_options": plot_options,
        }

    def __json_encode__(self):
        return self.config()

    @classmethod
    def __json_decode__(cls, value):
        instance = cls()
        if "options" in value:
            instance.set_options(**value["options"])
        if "plot_options" in value:
            instance.set_plot_options(**value["plot_options"])
        return instance
