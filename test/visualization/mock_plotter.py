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
"""
Mock plotter for testing.
"""

from typing import Dict, List, Union

from qiskit_experiments.visualization import BaseDrawer, BasePlotter
from qiskit_experiments.visualization.drawers import SeriesName


class MockPlotter(BasePlotter):
    """Mock plotter for visualization tests.

    If :attr:`plotting_enabled` is true, :class:`.MockPlotter` will plot formatted data.
    :attr:`plotting_enabled` defaults to false as most test usage of the class uses :class:`.MockDrawer`,
    which doesn't generate a useful figure.
    """

    def __init__(self, drawer: BaseDrawer, plotting_enabled: bool = False):
        """Construct a mock plotter instance for testing.

        Args:
            drawer: The drawer to use for plotting
            plotting_enabled: Whether to actually plot using :attr:`drawer` or not. Defaults to False.
        """
        super().__init__(drawer)
        self._plotting_enabled = plotting_enabled
        self._plotted_data_log = {}
        self._enabled_legends = {}

    def enable_legend_for(self, series_name: str, plot_type: str):
        """Enables legend for the given series name and plot-type.

        Plot types can be identified by the series data-key prefixes. They are also listed here,
        alongside associated data-keys.

        Plot Types:
            scatter: For ``scatter_x``, ``scatter_y`` data-keys.
            errorbar: For ``errorbar_x``, ``errorbar_y``, ``errorbar_x_err``, ``errorbar_y_err``
                data-keys.
            line: For ``line_x`` and ``line_y`` data-keys.
            filled_y_area: For ``filled_y_area_x``, ``filled_y_area_y_ub``, ``filled_y_area_y_lb``
                data-keys.
            filled_x_area: For ``filled_x_area_y``, ``filled_x_area_x_ub``, ``filled_x_area_x_lb``
                data-keys.

        Args:
            series_name: Series name for which a legend should be enabled.
            plot_type: The plot type for which a legend should be enabled.
        """
        if series_name not in self._enabled_legends:
            self._enabled_legends[series_name] = set()
        self._enabled_legends[series_name].add(plot_type)

    def legend_enabled_for(self, series_name: str, plot_type: str) -> bool:
        """Returns whether legends for plot ``plot_type`` has been enabled for the given series.

        Args:
            series_name: The series-name identifying the series.
            plot_type: The plot type as a string. See :meth:`enable_legend_for` for accepted values.

        Returns:
            bool: Whether legend has been enabled for the given series and plot type.
        """
        return plot_type in self._enabled_legends.get(series_name, set())

    @property
    def plotting_enabled(self):
        """Whether :class:`MockPlotter` should plot data.

        Defaults to False during construction.
        """
        return self._plotting_enabled

    @property
    def plotted_data_log(self) -> Dict[SeriesName, Dict[str, int]]:
        """Returns a dictionary of counters representing the number of times a given series and data-key
        were plotted.

        Returns:
            Dict[SeriesName,Dict[str, int]]: A dictionary of counters.
        """
        return self._plotted_data_log

    def plotted_data_counter(self, series_name: SeriesName, data_key: str) -> int:
        """Returns the plotted data counter from :attr:`plotted_data_log`, or zero if no counter exists.

        Args:
            series_name (_type_): The series name for the counter.
            data_key (_type_): The data-key for the counter.

        Returns:
            int: The plotted data log counter, or zero if no counter exists.
        """
        return self.plotted_data_log.get(series_name, {}).get(data_key, 0)

    def _log_plotted_data(self, series_name: SeriesName, data_keys: Union[str, List[str]]):
        """Logs that the following data-keys, for the given series, were plotted.

        This method helps track the data that is plotted by :class:`MockPlotter`. When called, an
        internal counter, keyed on both ``series_name`` and entries from ``data_keys`` is incremented.
        The internal counters can be accessed using :attr:`plotted_data_log` and
        :meth:`plotted_data_counter`. Individual series-name and data-key counters are accessed as
        follows:

        .. code-block:: python

            # Figure is generated at some point, drawing graphics on a canvas.
            plotter.figure()

            ## Counters are checked in a unit-test.
            # Here we expect that the "scatter_x" data for series "seriesA" was plotted once.
            self.assertEqual(plotter.plotted_data_counter("seriesA", "scatter_x"), 1)

        Args:
            series_name: The series-name for which the data-keys were plotted.
            data_keys: A list of data-keys that were plotted. If a single data-key is provided as a
                string, it will be added to a list automatically.
        """
        if series_name not in self._plotted_data_log:
            self._plotted_data_log[series_name] = {}
        if not isinstance(data_keys, list):
            data_keys = [data_keys]
        for data_key in data_keys:
            if data_key not in self._plotted_data_log[series_name]:
                self._plotted_data_log[series_name][data_key] = 0
            self._plotted_data_log[series_name][data_key] += 1

    def _plot_figure(self):
        """Plots a figure if :attr:`plotting_enabled` is True.

        If :attr:`plotting_enabled` is True, :class:`MockPlotter` calls
        :meth:`~BaseDrawer.scatter` for a series titled ``seriesA`` with ``x``, ``y``, and
        ``z`` data-keys assigned to the x and y values and the y-error/standard deviation respectively.
        :class:`MockPlotter` will also plot arbitrary data-keys for series-data identified by data-keys
        given by :meth:`expected_series_data_keys`. If :attr:`drawer` generates a figure and ``x``,
        ``y``, and ``z`` data is provided, then :meth:`figure` should return a scatterplot figure with
        error-bars. The mappings between other data-keys and the graphics drawn are listed below.

        Data Keys:
            scatter_{x,y}: Draws scatter points.
            errorbar_{x,y} and errorbar_{x,y}_err: Draws scatter points with error-bars.
                ``errorbar_{x,y}_err`` can be None, in which case error-bars are not drawn along that
                axis.
            line_{x,y}: Draws a line.
            filled_y_area_x and filled_y_area_y_{ub,lb}: Draws a filled region as a function of X values.
            filled_x_area_y and filled_x_area_x_{ub,lb}: Draws a filled region as a function of Y values.
        """
        if self.plotting_enabled:
            for ser in self.series:
                if self.data_exists_for(ser, ["x", "y", "z"]):
                    self.drawer.scatter(
                        *self.data_for(ser, ["x", "y", "z"]),
                        name=ser,
                    )
                    self._log_plotted_data(ser, ["x", "y", "z"])

                if self.data_exists_for(ser, ["scatter_x", "scatter_y"]):
                    self.drawer.scatter(
                        *self.data_for(ser, ["scatter_x", "scatter_y"]),
                        name=ser,
                        legend=self.legend_enabled_for(ser, "scatter"),
                    )
                    self._log_plotted_data(ser, ["scatter_x", "scatter_y"])

                if self.data_exists_for(ser, ["errorbar_x", "errorbar_y"]):
                    self.drawer.scatter(
                        *self.data_for(
                            ser, ["errorbar_x", "errorbar_y", "errorbar_x_err", "errorbar_y_err"]
                        ),
                        name=ser,
                        legend=self.legend_enabled_for(ser, "errorbar"),
                    )
                    self._log_plotted_data(ser, ["errorbar_x", "errorbar_y"])
                    for data_key in ["errorbar_x_err", "errorbar_y_err"]:
                        if self.data_exists_for(ser, data_key):
                            self._log_plotted_data(ser, data_key)

                if self.data_exists_for(ser, ["line_x", "line_y"]):
                    self.drawer.line(
                        *self.data_for(ser, ["line_x", "line_y"]),
                        name=ser,
                        legend=self.legend_enabled_for(ser, "line"),
                    )
                    self._log_plotted_data(ser, ["line_x", "line_y"])

                if self.data_exists_for(
                    ser, ["filled_y_area_x", "filled_y_area_y_ub", "filled_y_area_y_lb"]
                ):
                    self.drawer.filled_y_area(
                        *self.data_for(
                            ser,
                            ["filled_y_area_x", "filled_y_area_y_ub", "filled_y_area_y_lb"],
                        ),
                        name=ser,
                        legend=self.legend_enabled_for(ser, "filled_y_area"),
                    )
                    self._log_plotted_data(
                        ser,
                        ["filled_y_area_x", "filled_y_area_y_ub", "filled_y_area_y_lb"],
                    )

                if self.data_exists_for(
                    ser, ["filled_x_area_y", "filled_x_area_x_ub", "filled_x_area_x_lb"]
                ):
                    self.drawer.filled_x_area(
                        *self.data_for(
                            ser,
                            ["filled_x_area_y", "filled_x_area_x_ub", "filled_x_area_x_lb"],
                        ),
                        name=ser,
                        legend=self.legend_enabled_for(ser, "filled_x_area"),
                    )
                    self._log_plotted_data(
                        ser,
                        ["filled_x_area_y", "filled_x_area_x_ub", "filled_x_area_x_lb"],
                    )

    @classmethod
    def expected_series_data_keys(cls) -> List[str]:
        """Dummy data-keys.

        Data Keys:
            x: Dummy value.
            y: Dummy value.
            z: Dummy value.
            scatter_x: X values for scatter plot.
            scatter_y: Y values for scatter plot.
            errorbar_x: X values for scatter with error.
            errorbar_y: Y values for scatter with error.
            errorbar_x_err: X value errors for scatter with error.
            errorbar_y_err: Y value errors for scatter with error.
            line_x: X values for line plot.
            line_y: Y values for line plot.
            filled_y_area_x: X values for filled Y area plot.
            filled_y_area_y_ub: Y upperbound values for filled Y area plot.
            filled_y_area_y_lb: Y lowerbound values for filled Y area plot.
            filled_x_area_y: Y values for filled X area plot.
            filled_x_area_x_ub: X upperbound values for filled X area plot.
            filled_x_area_x_lb: X lowerbound values for filled X area plot.
        """
        return [
            "x",
            "y",
            "z",
            "scatter_x",
            "scatter_y",
            "errorbar_x",
            "errorbar_y",
            "errorbar_x_err",
            "errorbar_y_err",
            "line_x",
            "line_y",
            "filled_y_area_x",
            "filled_y_area_y_ub",
            "filled_y_area_y_lb",
            "filled_x_area_y",
            "filled_x_area_x_ub",
            "filled_x_area_x_lb",
        ]

    @classmethod
    def expected_supplementary_data_keys(cls) -> List[str]:
        """Dummy supplementary data-keys.

        Data Keys:
            textbox_text: Text to draw in a textbox.
        """
        return [
            "report_text",
            "supplementary_data_key",
            "textbox_text",
        ]
