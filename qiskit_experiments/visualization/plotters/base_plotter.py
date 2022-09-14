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

from abc import ABC, abstractmethod
from typing import Any, Iterable, Union, Optional, List, Tuple, Dict
from qiskit_experiments.framework import Options
from qiskit_experiments.visualization.drawers import BaseDrawer


class BasePlotter(ABC):
    """An abstract class for the serializable figure plotters.

    A plotter takes data from an experiment analysis class and plots a given figure using a drawing
    backend. Sub-classes define the kind of figure created.

    Data is grouped into series, identified by a series name (str). There can be multiple different sets
    of data associated with a given series name, which are identified by a data key (str). Experimental
    and analysis results can be passed to the plotter so appropriate graphics can be drawn on the figure
    canvas. Adding data is done through :meth:`set_series_data` and :meth:`set_figure_data`, with
    querying done through the :meth:`data_for` and :meth:`data_exists_for` methods.

    There are two types of data associated with a plotter: series and figure data. The former is a
    dataset of values to be plotted on a canvas, such that the data can be grouped into subsets
    identified by their series name. Series names can be thought of as legend labels for the plotted
    data. Figure data is not associated with a series and is instead only associated with the figure.
    Examples include analysis reports or other text that is drawn onto the figure canvas.
    """

    def __init__(self, drawer: BaseDrawer):
        """Create a new plotter instance.

        Args:
            drawer: The drawer to use when creating the figure.
        """
        self._series_data: Dict[str, Dict[str, Any]] = {}
        self._figure_data: Dict[str, Any] = {}
        self._options = self._default_options()
        self._set_options = set()
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
        return self._figure_data

    @property
    def series_data(self) -> Dict[str, Dict[str, Any]]:
        return self._series_data

    @property
    def series(self) -> List[str]:
        """The series names for this plotter."""
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

        The returned tuple contains the data, associated with ``data_keys``, in the same orders as they are provided. For example,

        .. code-example::python
            plotter.set_series_data("seriesA", x=data.x, y=data.y, yerr=data.yerr)

            # The following calls are equivalent.
            x, y, yerr = plotter.series_data_for("seriesA", ["x", "y", "yerr"])
            x, y, yerr = data.x, data.y, data.yerr

        :meth:`series_data_for` is intended to be used by sub-classes of :class:`BasePlotter` when
        plotting in :meth:`_plot_figure`.

        Args:
            series_name: The series name for the given series.
            data_keys: List of data-keys for the data to be returned.

        Returns:
            tuple: A tuple of data associated with the given series, identified by ``data_keys``.
        """

        # We may be given a single data-key, but we need an iterable for the rest of the function.
        if not isinstance(data_keys, list):
            data_keys = [data_keys]

        # The series doesn't exist in the plotter data, return None for each data-key in the output.
        if series_name not in self._series_data:
            return (None,) * len(data_keys)

        return (self._series_data[series_name].get(key, None) for key in data_keys)

    def set_series_data(self, series_name: str, **data_kwargs):
        """Sets data for the given series.

        Note that if data has already been assigned for the given series and data-key, it will be
        overridden by the new values.

        Args:
            series_name: The name of the given series.
            data_kwargs: The data to be added, where the keyword is the data-key.
        """
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
        "goodness" of a curve-fit, not on the specific of a given line, point, or shape drawn on the
        figure canvas.
        """
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

        return all([key in self._series_data[series_name] for key in data_keys])

    @abstractmethod
    def _plot_figure(self):
        """Generates a figure using :attr:`drawer` and :meth:`data`.

        Sub-classes must override this function to plot data using the drawer. This function is called by
        :meth:`figure`.
        """

    def figure(self) -> Any:
        """Generates and returns a figure for the already provided data.

        :meth:`figure` calls :meth:`_plot_figure`, which is overridden by sub-classes. Before and after calling :meth:`_plot_figure`, :func:`initialize_canvas` and :func:`format_canvas` are called on the drawer respectively.

        Returns:
            Any: A figure generated by :attr:`drawer`.
        """
        self.drawer.initialize_canvas()
        self._plot_figure()
        self.drawer.format_canvas()
        return self.drawer.figure

    @property
    def options(self) -> Options:
        return self._options

    @classmethod
    @abstractmethod
    def _default_series_data_keys(cls) -> List[str]:
        """Returns the default series data-keys supported by this plotter.

        Returns:
            list: List of data-keys.
        """
        # TODO: This function is meant to be similar to _default_options, so that data-keys are defined somewhere. Not sure if this is the best way of doing it.

    @classmethod
    def _default_options(cls) -> Options:
        """Return default plotting options."""
        return Options()

    def set_options(self, **fields):
        """Set the plotter options.

        Args:
            fields: The fields to update the options.
        """
        self._options.update_options(**fields)
        self._set_options = self._set_options.union(fields)

    def config(self) -> Dict:
        """Return the config dictionary for this drawing."""
        # TODO: Figure out how self._drawer:BaseDrawer be serialized?
        options = dict((key, getattr(self._options, key)) for key in self._set_options)

        return {
            "cls": type(self),
            "options": options,
        }

    def __json_encode__(self):
        return self.config()

    @classmethod
    def __json_decode__(cls, value):
        # TODO: Figure out how self._drawer:BaseDrawer be serialized?
        instance = cls()
        if "options" in value:
            instance.set_options(**value["options"])
        return instance
