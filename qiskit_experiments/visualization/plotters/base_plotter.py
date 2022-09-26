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
"""Base plotter abstract class"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from qiskit_experiments.framework import Options
from qiskit_experiments.visualization.drawers import BaseDrawer
from qiskit_experiments.visualization.style import PlotStyle


class BasePlotter(ABC):
    """An abstract class for the serializable figure plotters.

    A plotter takes data from an experiment analysis class or experiment and plots a given figure using a
    drawing backend. Sub-classes define the kind of figure created and the expected data.

    Data is split into series and figure data. Series data is grouped by series name (str). For
    :class:`CurveAnalysis`, this is the model name for a curve fit. For series data associated with a
    single series name and figure data, data-values are identified by a data-key (str). Different data
    per series and figure must have a different data-key to avoid overwriting values. Experiment and
    analysis results can be passed to the plotter so appropriate graphics can be drawn on the figure
    canvas. Series data is added to the plotter using :meth:`set_series_data` whereas figure data is
    added using :meth:`set_figure_data`. Series and figure data are retrieved using :meth:`data_for` and
    :attr:`figure_data` respectively.

    Series data contains values to be plotted on a canvas, such that the data can be grouped into subsets
    identified by their series name. Series names can be thought of as legend labels for the plotted
    data, and as curve names for a curve-fit. Figure data is not associated with a series or curve and is
    instead only associated with the figure. Examples include analysis reports or other text that is
    drawn onto the figure canvas.

    # TODO: Add example usage and description of options and plot-options.
    """

    def __init__(self, drawer: BaseDrawer):
        """Create a new plotter instance.

        Args:
            drawer: The drawer to use when creating the figure.
        """
        # Data to be plotted, such as scatter points, interpolated fits, and confidence intervals
        self._series_data: Dict[str, Dict[str, Any]] = {}
        # Data for figure-wide drawing, unrelated to series data, such as text or fit reports.
        self._figure_data: Dict[str, Any] = {}

        # Options for the plotter
        self._options = self._default_options()
        # Plotter options that have changed, for serialization.
        self._set_options = set()

        # Plot options that are updated in the drawer when `plotter.figure()` is called
        self._plot_options = self._default_plot_options()
        # Plot options that have changed, for serialization.
        self._set_plot_options = set()

        # The drawer backend to use for plotting.
        self._drawer = drawer

    @property
    def drawer(self) -> BaseDrawer:
        """The drawer being used by the plotter."""
        return self._drawer

    @drawer.setter
    def drawer(self, new_drawer: BaseDrawer):
        """Set the drawer to be used by the plotter."""
        self._drawer = new_drawer

    @property
    def figure_data(self) -> Dict[str, Any]:
        """Data for the figure being plotted.

        Figure data includes text, fit reports, or other data that is associated with the figure as a
        whole and not an individual series.
        """
        return self._figure_data

    @property
    def series_data(self) -> Dict[str, Dict[str, Any]]:
        """Data for series being plotted.

        Series data includes data such as scatter points, interpolated fit values, and
        standard-deviations. Series data is grouped by series-name and then by a data-key, both strings.
        Though series data can be accessed through :meth:`series_data`, it is recommended to use
        :meth:`data_for` and :meth:`data_exists_for`.

        Returns:
            dict: A dictionary containing series data.
        """
        return self._series_data

    @property
    def series(self) -> List[str]:
        """Series names that have been added to this plotter."""
        return list(self._series_data.keys())

    def data_keys_for(self, series_name: str) -> List[str]:
        """Returns a list of data-keys for the given series.

        Args:
            series_name: The series name for the given series.

        Returns:
            list: The list of data-keys for data in the plotter associated with the given series. If the
                series has not been added to the plotter, an empty list is returned.
        """
        if series_name not in self._series_data:
            return []
        return list(self._series_data[series_name])

    def data_for(self, series_name: str, data_keys: Union[str, List[str]]) -> Tuple[Optional[Any]]:
        """Returns data associated with the given series.

        The returned tuple contains the data, associated with ``data_keys``, in the same orders as they
        are provided. For example,

        .. code-example::python
            plotter.set_series_data("seriesA", x=data.x, y=data.y, yerr=data.yerr)

            # The following calls are equivalent.
            x, y, yerr = plotter.series_data_for("seriesA", ["x", "y", "yerr"])
            x, y, yerr = data.x, data.y, data.yerr

        :meth:`data_for` is intended to be used by sub-classes of :class:`BasePlotter` when
        plotting in :meth:`_plot_figure`.

        Args:
            series_name: The series name for the given series.
            data_keys: List of data-keys for the data to be returned. If a single data-key is given as a
                string, it is wrapped in a list.

        Returns:
            tuple: A tuple of data associated with the given series, identified by ``data_keys``. If no
                data has been set for a data-key, None is returned for the associated tuple entry.
        """

        # We may be given a single data-key, but we need a list for the rest of the function.
        if not isinstance(data_keys, list):
            data_keys = [data_keys]

        # The series doesn't exist in the plotter data, return None for each data-key in the output.
        if series_name not in self._series_data:
            return (None,) * len(data_keys)

        return tuple(self._series_data[series_name].get(key, None) for key in data_keys)

    def set_series_data(self, series_name: str, **data_kwargs):
        """Sets data for the given series.

        Note that if data has already been assigned for the given series and data-key, it will be
        overwritten with the new values. ``set_series_data`` will warn if the data-key is unexpected;
        i.e., not within those returned by :meth:`expected_series_data_keys`.

        Args:
            series_name: The name of the given series.
            data_kwargs: The data to be added, where the keyword is the data-key.
        """
        # Warn if the data-keys are not expected.
        unknown_data_keys = [
            data_key for data_key in data_kwargs if data_key not in self.expected_series_data_keys()
        ]
        for unknown_data_key in unknown_data_keys:
            warnings.warn(
                f"{self.__class__.__name__} encountered an unknown data-key {unknown_data_key}. It may "
                "not be used by the plotter class."
            )

        # Set data
        if series_name not in self._series_data:
            self._series_data[series_name] = {}
        self._series_data[series_name].update(**data_kwargs)

    def clear_series_data(self, series_name: Optional[str] = None):
        """Clear series data for this plotter.

        Args:
            series_name: The series name identifying which data should be cleared. If None, all series
                data is cleared. Defaults to None.
        """
        if series_name is None:
            self._series_data = {}
        elif series_name in self._series_data:
            self._series_data.pop(series_name)

    def set_figure_data(self, **data_kwargs):
        """Sets data for the entire figure.

        Figure data differs from series data in that it is not associate with a series name. Fit reports
        are examples of figure data as they are drawn on figures to report on analysis results and the
        "goodness" of a curve-fit, not a specific line, point, or shape drawn on the figure canvas.

        Note that if data has already been assigned for the given data-key, it will be overwritten with
        the new values. ``set_figure_data`` will warn if the data-key is unexpected; i.e., not within
        those returned by :meth:`expected_figure_data_keys`.

        """

        # Warn if any data-keys are not expected.
        unknown_data_keys = [
            data_key for data_key in data_kwargs if data_key not in self.expected_figure_data_keys()
        ]
        for unknown_data_key in unknown_data_keys:
            warnings.warn(
                f"{self.__class__.__name__} encountered an unknown data-key {unknown_data_key}. It may "
                "not be used by the plotter class."
            )

        self._figure_data.update(**data_kwargs)

    def clear_figure_data(self):
        """Clears figure data."""
        self._figure_data = {}

    def data_exists_for(self, series_name: str, data_keys: Union[str, List[str]]) -> bool:
        """Returns whether the given data-keys exist for the given series.

        Args:
            series_name: The name of the given series.
            data_keys: The data-keys to be checked.

        Returns:
            bool: True if all data-keys have values assigned for the given series. False if at least one
                does not have a value assigned.
        """
        if not isinstance(data_keys, list):
            data_keys = [data_keys]

        # Handle non-existent series name
        if series_name not in self._series_data:
            return False

        return all(key in self._series_data[series_name] for key in data_keys)

    @abstractmethod
    def _plot_figure(self):
        """Generates a figure using :attr:`drawer` and :meth:`data`.

        Sub-classes must override this function to plot data using the drawer. This function is called by
        :meth:`figure` when :attr:`drawer` can be used to draw on the canvas.
        """

    def figure(self) -> Any:
        """Generates and returns a figure for the already provided series and figure data.

        :meth:`figure` calls :meth:`_plot_figure`, which is overridden by sub-classes. Before and after
        calling :meth:`_plot_figure`; :func:`_initialize_drawer`, :func:`initialize_canvas` and
        :func:`format_canvas` are called on the drawer respectively.

        Returns:
            Any: A figure generated by :attr:`drawer`, of the same type as ``drawer.figure``.
        """
        # Initialize drawer, to copy axis, subplots, and plot-options across.
        self._initialize_drawer()

        # Initialize canvas, which creates subplots, assigns axis labels, etc.
        self.drawer.initialize_canvas()

        # Plot figure for given subclass. This is the core of BasePlotter subclasses.
        self._plot_figure()

        # Final formatting of canvas, which sets axis limits etc.
        self.drawer.format_canvas()

        # Return whatever figure is created by the drawer.
        return self.drawer.figure

    @classmethod
    @abstractmethod
    def expected_series_data_keys(cls) -> List[str]:
        """Returns the expected series data-keys supported by this plotter."""

    @classmethod
    @abstractmethod
    def expected_figure_data_keys(cls) -> List[str]:
        """Returns the expected figures data-keys supported by this plotter."""

    @property
    def options(self) -> Options:
        """Options for the plotter.

        Options for a plotter modify how the class generates a figure. This includes an optional axis
        object, being the drawer canvas. Make sure verify whether the option you want to set is in
        :attr:`options` or :attr:`plot_options`.
        """
        return self._options

    @property
    def plot_options(self) -> Options:
        """Plot options for the plotter and its drawer.

        Plot options differ from normal options (:attr:`options`) in that the plotter passes plot options
        on to the drawer when creating a figure (when :meth:`figure` is called). This way :attr:`drawer`
        can draw an appropriate figure. An example of a plot option is the x-axis label.
        """
        return self._plot_options

    @classmethod
    def _default_options(cls) -> Options:
        """Return default plotter options.

        Options:
            axis (Any): Arbitrary object that can be used as a drawing canvas.
            subplots (Tuple[int, int]): Number of rows and columns when the experimental
                result is drawn in the multiple windows.
            style (PlotStyle): The style definition to use when plotting.
                This overwrites plotting options set in :attr:`drawer`. The default is an empty style
                object, and such the default drawer plotting style will be used.
        """
        return Options(
            axis=None,
            subplots=(1, 1),
            style=PlotStyle(),
        )

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
                For example, when the x values represent time, this option will be just "s" rather than
                "ms". In the output figure, the prefix is automatically selected based on the maximum
                value in this axis. If your x values are in [1e-3, 1e-4], they are displayed as [1 ms, 10
                ms]. This option is likely provided by the analysis class rather than end-users. However,
                users can still override if they need different unit notation. By default, this option is
                set to ``None``, and no scaling is applied. If nothing is provided, the axis numbers will
                be displayed in the scientific notation.
            yval_unit (str): Unit of y values. See ``xval_unit`` for details.
            figure_title (str): Title of the figure. Defaults to None, i.e. nothing is shown.
            series_params (Dict[str, Dict[str, Any]]): A dictionary of plot parameters for each series.
                This is keyed on the name for each series. Sub-dictionary is expected to have following
                three configurations, "canvas", "color", and "symbol"; "canvas" is the integer index of
                axis (when multi-canvas plot is set), "color" is the color of the curve, and "symbol" is
                the marker style of the curve for scatter plots.
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
        )

    def set_options(self, **fields):
        """Set the plotter options.

        Args:
            fields: The fields to update in options.
        """
        for field in fields:
            if not hasattr(self._options, field):
                raise AttributeError(
                    f"Options field {field} is not valid for {type(self).__name__}"
                )
        self._options.update_options(**fields)
        self._set_options = self._set_options.union(fields)

    def set_plot_options(self, **fields):
        """Set the plot options.

        Args:
            fields: The fields to update in plot options.
        """
        # Don't check if any option in fields already exists (like with `set_options`), as plot options
        # are passed to `.drawer` which may have other plot-options. Any plot-option that isn't set in
        # `.drawer.plot_options` won't be set anyway. Setting `.drawer.plot_options` only occurs in
        # `.figure()`, so we can't compare to `.drawer.plot_options` now as `.drawer` may be changed
        # between now and the call to `.figure()`.
        self._plot_options.update_options(**fields)
        self._set_plot_options = self._set_plot_options.union(fields)

    def _initialize_drawer(self):
        """Configures :attr:`drawer` before plotting.

        The following actions are taken:
            1. ``axis``, ``subplots``, and ``style`` are passed to :attr:`drawer`.
            2. ``plot_options`` in :attr:`drawer` are updated based on values set in plotter
                :attr:`plot_options`

        These steps are different as any plot-option can be passed to :attr:`drawer` if the drawer has a
        plot-option with the same name. However, ``axis``, ``subplots``, and ``style`` are the only
        plotter options passed to :attr:`drawer`. This is done because :class:`BasePlotter` distinguishes
        between plotter options and plot-options.
        """
        ## Axis, subplots, and style
        if self.options.axis:
            self.drawer.set_options(axis=self.options.axis)
        if self.options.subplots:
            self.drawer.set_options(subplots=self.options.subplots)
        self.drawer.set_plot_options(custom_style=self.options.style)

        ## Plot Options
        # HACK: We need to accesses internal variables of the Options class, which is not good practice.
        # Options._fields are dictionaries. However, given we are accessing an internal variable, this
        # may change in the future.
        _drawer_plot_options = self.drawer.plot_options._fields
        _plotter_plot_options = self.plot_options._fields

        # If an option exists in drawer.plot_options AND in self.plot_options, set the drawers
        # plot-option value to that from the plotter.
        for opt_key in _drawer_plot_options:
            if opt_key in _plotter_plot_options:
                _drawer_plot_options[opt_key] = _plotter_plot_options[opt_key]

        # Use drawer.set_plot_options so plot-options are serialized.
        self.drawer.set_plot_options(**_drawer_plot_options)

    def config(self) -> Dict:
        """Return the config dictionary for this drawing."""
        options = dict((key, getattr(self._options, key)) for key in self._set_options)
        plot_options = dict(
            (key, getattr(self._plot_options, key)) for key in self._set_plot_options
        )
        drawer = self.drawer.__json_encode__()

        return {
            "cls": type(self),
            "options": options,
            "plot_options": plot_options,
            "drawer": drawer,
        }

    def __json_encode__(self):
        return self.config()

    @classmethod
    def __json_decode__(cls, value):
        ## Process drawer as it's needed to create a plotter
        drawer_values = value["drawer"]
        # We expect a subclass of BaseDrawer
        drawer_cls: BaseDrawer = drawer_values["cls"]
        drawer = drawer_cls.__json_decode__(drawer_values)

        # Create plotter instance
        instance = cls(drawer)
        if "options" in value:
            instance.set_options(**value["options"])
        if "plot_options" in value:
            instance.set_plot_options(**value["plot_options"])
        return instance
