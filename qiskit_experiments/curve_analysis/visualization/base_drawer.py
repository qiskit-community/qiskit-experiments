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

"""Curve drawer abstract class."""

from abc import ABC, abstractmethod
from typing import Dict, Sequence, Optional

from qiskit_experiments.framework import Options


class BaseCurveDrawer(ABC):
    """Abstract class for the serializable Qiskit Experiments curve drawer.

    A curve drawer may be implemented by different drawing backends such as matplotlib
    or plotly. Sub-classes that wrap these backends by subclassing `BaseCurveDrawer` must
    implement the following abstract methods.

    initialize_canvas

        This method should implement a protocol to initialize a drawing canvas
        with user input ``axis`` object. Note that curve analysis drawer
        supports visualization of experiment results in multiple canvases
        tiled into N (row) x M (column) inset grids, which is specified in the option ``subplots``.
        By default, this is N=1, M=1 and thus no inset grid will be initialized.
        The data points to draw might be provided with a canvas number defined in
        :attr:`SeriesDef.canvas` which defaults to ``None``, i.e. no-inset grids.

        This method should first check the drawing options for the axis object
        and initialize the axis only when it is not provided by the options.
        Once axis is initialized, this is set to the instance member ``self._axis``.

    format_canvas

        This method should implement a protocol to format the appearance of canvas.
        Typically, it updates axis and tick labels. Note that the axis SI unit
        may be specified in the drawing options. In this case, axis numbers should be
        auto-scaled with the unit prefix.

    draw_raw_data

        This method is called after data processing is completed.
        This method draws raw experiment data points on the canvas.

    draw_formatted_data

        This method is called after data formatting is completed.
        The formatted data might be averaged over the same x values,
        or smoothed by a filtering algorithm, depending on how analysis class is implemented.
        This method is called with error bars of y values and the name of the curve.

    draw_fit_line

        This method is called after fitting is completed and when there is valid fit outcome.
        This method is called with the interpolated x and y values.

    draw_confidence_interval

        This method is called after fitting is completed and when there is valid fit outcome.
        This method is called with the interpolated x and a pair of y values
        that represent the upper and lower bound within certain confidence interval.
        This might be called multiple times with different interval sizes.

    draw_fit_report

        This method is called after fitting is completed and when there is valid fit outcome.
        This method is called with the list of analysis results and the reduced chi-squared values.
        The fit report should be generated to show this information on the canvas.

    """

    def __init__(self):
        self._options = self._default_options()
        self._set_options = set()
        self._axis = None
        self._curves = list()

    @property
    def options(self) -> Options:
        """Return the drawing options."""
        return self._options

    @classmethod
    def _default_options(cls) -> Options:
        """Return default draw options.

        Draw Options:
            axis (Any): Arbitrary object that can be used as a drawing canvas.
            subplots (Tuple[int, int]): Number of rows and columns when the experimental
                result is drawn in the multiple windows.
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
            figsize (Tuple[int, int]): A tuple of two numbers representing the size of
                the output figure (width, height). Note that this is applicable
                only when ``axis`` object is not provided. If any canvas object is provided,
                the figure size associated with the axis is preferentially applied.
            legend_loc (str): Vertical and horizontal location of the curve legend window in
                a single string separated by a space. This defaults to ``center right``.
                Vertical position can be ``upper``, ``center``, ``lower``.
                Horizontal position can be ``right``, ``center``, ``left``.
            tick_label_size (int): Size of text representing the axis tick numbers.
            axis_label_size (int): Size of text representing the axis label.
            fit_report_rpos (Tuple[int, int]): A tuple of numbers showing the location of
                the fit report window. These numbers are horizontal and vertical position
                of the top left corner of the window in the relative coordinate
                on the output figure, i.e. ``[0, 1]``.
                The fit report window shows the selected fit parameters and the reduced
                chi-squared value.
            fit_report_text_size (int): Size of text in the fit report window.
            plot_sigma (List[Tuple[float, float]]): A list of two number tuples
                showing the configuration to write confidence intervals for the fit curve.
                The first argument is the relative sigma (n_sigma), and the second argument is
                the transparency of the interval plot in ``[0, 1]``.
                Multiple n_sigma intervals can be drawn for the single curve.
            plot_options (Dict[str, Dict[str, Any]]): A dictionary of plot options for each curve.
                This is keyed on the model name for each curve. Sub-dictionary is expected to have
                following three configurations, "canvas", "color", and "symbol"; "canvas" is the
                integer index of axis (when multi-canvas plot is set), "color" is the
                color of the curve, and "symbol" is the marker style of the curve for scatter plots.
            figure_title (str): Title of the figure. Defaults to None, i.e. nothing is shown.
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
            plot_sigma=[(1.0, 0.7), (3.0, 0.3)],
            plot_options={},
            figure_title=None,
        )

    def set_options(self, **fields):
        """Set the drawing options.
        Args:
            fields: The fields to update the options
        """
        self._options.update_options(**fields)
        self._set_options = self._set_options.union(fields)

    @abstractmethod
    def initialize_canvas(self):
        """Initialize the drawing canvas."""

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
            name: Name of this curve.
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
            name: Name of this curve.
            options: Valid options for the drawer backend API.
        """

    @abstractmethod
    def draw_fit_line(
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
            name: Name of this curve.
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
        """Draw cofidence interval.

        Args:
            x_data: X values.
            y_ub: The upper boundary of Y values.
            y_lb: The lower boundary of Y values.
            name: Name of this curve.
            options: Valid options for the drawer backend API.
        """

    @abstractmethod
    def draw_fit_report(
        self,
        description: str,
        **options,
    ):
        """Draw text box that shows fit reports.

        Args:
            description: A string to describe the fiting outcome.
            options: Valid options for the drawer backend API.
        """

    @property
    @abstractmethod
    def figure(self):
        """Return figure object handler to be saved in the database."""

    def config(self) -> Dict:
        """Return the config dictionary for this drawing."""
        options = dict((key, getattr(self._options, key)) for key in self._set_options)

        return {"cls": type(self), "options": options}

    def __json_encode__(self):
        return self.config()

    @classmethod
    def __json_decode__(cls, value):
        instance = cls()
        if "options" in value:
            instance.set_options(**value["options"])
        return instance
