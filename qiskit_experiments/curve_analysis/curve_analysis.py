# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Analysis class for curve fitting.
"""
# pylint: disable=invalid-name

import dataclasses
import inspect
from abc import ABC
import functools
from typing import Any, Dict, List, Tuple, Callable, Union, Optional

import numpy as np
from qiskit.providers.options import Options

from qiskit_experiments.framework import BaseAnalysis, ExperimentData
from qiskit_experiments.data_processing import DataProcessor
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.matplotlib import pyplot, requires_matplotlib, HAS_MATPLOTLIB
from qiskit_experiments.data_processing.processor_library import get_processor

from qiskit_experiments.curve_analysis.curve_data import CurveData, SeriesDef
from qiskit_experiments.curve_analysis.curve_analysis_result_data import CurveAnalysisResultData
from qiskit_experiments.curve_analysis.curve_fit import multi_curve_fit
from qiskit_experiments.curve_analysis.visualization import (
    plot_scatter,
    plot_errorbar,
    plot_curve_fit,
)
from qiskit_experiments.curve_analysis.utils import get_opt_value, get_opt_error


class CurveAnalysis(BaseAnalysis, ABC):
    """A base class for curve fit type analysis.

    The subclasses can override class attributes to define the behavior of
    data extraction and fitting. This docstring describes how code developers can
    create a new curve fit analysis subclass inheriting from this base class.

    Class Attributes:
        - ``__series__``: A set of data points that will be fit to the same parameters
          in the fit function. If this analysis contains multiple curves,
          the same number of series definitions should be listed. Each series definition
          is a :class:`SeriesDef` element, that may be initialized with

            - ``fit_func``: The function to which the data will be fit.
            - ``filter_kwargs``: Circuit metadata key and value associated with this curve.
              The data points of the curve are extracted from ExperimentData based on
              this information.
            - ``name``: Name of the curve. This is arbitrary data field, but should be unique.
            - ``plot_color``: String color representation of this series in the plot.
            - ``plot_symbol``: String formatter of the scatter of this series in the plot.
            - ``plot_fit_uncertainty``: A Boolean signaling whether to plot fit uncertainty
              for this series in the plot.

        See the Examples below for more details.


    Examples:

        **A fitting for single exponential decay curve**

        In this type of experiment, the analysis deals with a single curve.
        Thus filter_kwargs and series name are not necessary defined.

        .. code-block::

            class AnalysisExample(CurveAnalysis):

                __series__ = [
                    SeriesDef(
                        fit_func=lambda x, p0, p1, p2:
                            exponential_decay(x, amp=p0, lamb=p1, baseline=p2),
                    ),
                ]


        **A fitting for two exponential decay curve with partly shared parameter**

        In this type of experiment, the analysis deals with two curves.
        We need a __series__ definition for each curve, and filter_kwargs should be
        properly defined to separate each curve series.

        .. code-block::

            class AnalysisExample(CurveAnalysis):

                __series__ = [
                    SeriesDef(
                        name="my_experiment1",
                        fit_func=lambda x, p0, p1, p2, p3:
                            exponential_decay(x, amp=p0, lamb=p1, baseline=p3),
                        filter_kwargs={"experiment": 1},
                        plot_color="red",
                        plot_symbol="^",
                    ),
                    SeriesDef(
                        name="my_experiment2",
                        fit_func=lambda x, p0, p1, p2, p3:
                            exponential_decay(x, amp=p0, lamb=p2, baseline=p3),
                        filter_kwargs={"experiment": 2},
                        plot_color="blue",
                        plot_symbol="o",
                    ),
                ]

        In this fit model, we have 4 parameters `p0, p1, p2, p3` and both series share
        `p0` and `p3` as `amp` and `baseline` of the `exponential_decay` fit function.
        Parameter `p1` (`p2`) is only used by `my_experiment1` (`my_experiment2`).
        Both series have same fit function in this example.


        **A fitting for two trigonometric curves with the same parameter**

        In this type of experiment, the analysis deals with two different curves.
        However the parameters are shared with both functions.

        .. code-block::

            class AnalysisExample(CurveAnalysis):

                __series__ = [
                    SeriesDef(
                        name="my_experiment1",
                        fit_func=lambda x, p0, p1, p2, p3:
                            cos(x, amp=p0, freq=p1, phase=p2, baseline=p3),
                        filter_kwargs={"experiment": 1},
                        plot_color="red",
                        plot_symbol="^",
                    ),
                    SeriesDef(
                        name="my_experiment2",
                        fit_func=lambda x, p0, p1, p2, p3:
                            sin(x, amp=p0, freq=p1, phase=p2, baseline=p3),
                        filter_kwargs={"experiment": 2},
                        plot_color="blue",
                        plot_symbol="o",
                    ),
                ]

        In this fit model, we have 4 parameters `p0, p1, p2, p3` and both series share
        all parameters. However, these series have different fit curves, i.e.
        `my_experiment1` (`my_experiment2`) uses the `cos` (`sin`) fit function.


        **A fitting with fixed parameter**

        In this type of experiment, we can provide fixed fit function parameter.
        This parameter should be assigned via analysis options
        and not passed to the fitter function.

        .. code-block::

            class AnalysisExample(CurveAnalysis):

                __series__ = [
                    SeriesDef(
                        fit_func=lambda x, p0, p1, p2:
                            exponential_decay(x, amp=p0, lamb=p1, baseline=p2),
                    ),
                ]

                __fixed_parameters__ = ["p1"]

        You can add arbitrary number of parameters to the class variable
        ``__fixed_parameters__`` from the fit function arguments.
        This parameter should be defined with the fit functions otherwise the analysis
        instance cannot be created. In above example, parameter ``p1`` should be also
        defined in the analysis options. This parameter will be excluded from the fit parameters
        and thus will not appear in the analysis result.

    Notes:
        This CurveAnalysis class provides several private methods that subclasses can override.

        - Customize figure generation:
            Override :meth:`~self._create_figures`. For example, here you can create
            arbitrary number of new figures or upgrade the default figure appearance.

        - Customize pre-data processing:
            Override :meth:`~self._format_data`. For example, here you can apply smoothing
            to y values, remove outlier, or apply filter function to the data.

        - Customize post-analysis data processing:
            Override :meth:`~self._post_analysis`. For example, here you can
            calculate new entity from fit values, such as EPC of RB experiment.

        - Customize fitting options:
            Override :meth:`~self._setup_fitting`. For example, here you can
            calculate initial guess from experiment data and setup fitter options.

        See docstring of each method for more details.

        Note that other private methods are not expected to be overridden.
        If you forcibly override these methods, the behavior of analysis logic is not well tested
        and we cannot guarantee it works as expected (you may suffer from bugs).
        Instead, you can open an issue in qiskit-experiment github to upgrade this class
        with proper unittest framework.

        https://github.com/Qiskit/qiskit-experiments/issues
    """

    #: List[SeriesDef]: List of mapping representing a data series
    __series__ = None

    #: List[str]: Fixed parameter in fit function. Value should be set to the analysis options.
    __fixed_parameters__ = None

    def __new__(cls) -> "CurveAnalysis":
        """Parse series data if all fit functions have the same argument.

        Raises:
            AnalysisError:
                - When fit functions have different argument.

        Returns:
            CurveAnalysis instance with validated series definitions.
        """
        obj = object.__new__(cls)

        fsigs = set()
        for series_def in obj.__series__:
            fsigs.add(inspect.signature(series_def.fit_func))
        if len(fsigs) > 1:
            raise AnalysisError(
                "Fit functions specified in the series definition have "
                "different function signature. They should receive "
                "the same parameter set for multi-objective function fit."
            )

        # remove the first function argument. this is usually x, i.e. not a fit parameter.
        fit_params = list(list(fsigs)[0].parameters.keys())[1:]

        # remove fixed parameters
        if obj.__fixed_parameters__ is not None:
            for fixed_param in obj.__fixed_parameters__:
                try:
                    fit_params.remove(fixed_param)
                except ValueError as ex:
                    raise AnalysisError(
                        f"Defined fixed parameter {fixed_param} is not a fit function argument."
                        "Update series definition to ensure the parameter name is defined with "
                        f"fit functions. Currently available parameters are {fit_params}."
                    ) from ex
        obj.__fit_params = fit_params

        return obj

    def __init__(self):
        """Initialize data fields that are privately accessed by methods."""

        #: Dict[str, Any]: Experiment metadata
        self.__experiment_metadata = None

        #: List[CurveData]: Processed experiment data set.
        self.__processed_data_set = list()

        # Add expected options to instance variable so that every method can access to.
        for key in self._default_options().__dict__:
            setattr(self, f"__{key}", None)

    @classmethod
    def _default_options(cls) -> Options:
        """Return default analysis options.

        Analysis Options:
            curve_fitter (Callable): A callback function to perform fitting with formatted data.
                See :func:`~qiskit_experiments.analysis.multi_curve_fit` for example.
            data_processor (Callable): A callback function to format experiment data.
                This can be a :class:`~qiskit_experiments.data_processing.DataProcessor`
                instance that defines the `self.__call__` method.
            normalization (bool) : Set ``True`` to normalize y values within range [-1, 1].
            p0 (Dict[str, float]): Array-like or dictionary
                of initial parameters.
            bounds (Dict[str, Tuple[float, float]]): Array-like or dictionary
                of (min, max) tuple of fit parameter boundaries.
            x_key (str): Circuit metadata key representing a scanned value.
            plot (bool): Set ``True`` to create figure for fit result.
            axis (AxesSubplot): Optional. A matplotlib axis object to draw.
            xlabel (str): X label of fit result figure.
            ylabel (str): Y label of fit result figure.
            ylim (Tuple[float, float]): Min and max height limit of fit plot.
            fit_reports (Dict[str, str]): Mapping of fit parameters and representation
                in the fit report.
            return_data_points (bool): Set ``True`` to return formatted XY data.
        """
        return Options(
            curve_fitter=multi_curve_fit,
            data_processor=None,
            normalization=False,
            p0=None,
            bounds=None,
            x_key="xval",
            plot=True,
            axis=None,
            xlabel=None,
            ylabel=None,
            ylim=None,
            fit_reports=None,
            return_data_points=False,
        )

    @requires_matplotlib
    def _create_figures(self, result_data: CurveAnalysisResultData) -> List["Figure"]:
        """Create new figures with the fit result and raw data.

        Subclass can override this method to create different type of figures, but
        the ``requires_matplotlib`` decorator is needed to ensure this method
        works with ``DbExperimentData``.

        Args:
            result_data: Result data containing fit parameters.

        Returns:
            List of figures.
        """
        fit_available = all(key in result_data for key in ("popt", "popt_err", "xrange"))

        axis = self._get_option("axis")
        if axis is None:
            figure = pyplot.figure(figsize=(8, 5))
            axis = figure.subplots(nrows=1, ncols=1)
        else:
            figure = axis.get_figure()

        ymin, ymax = np.inf, -np.inf
        for series_def in self.__series__:

            # plot raw data

            curve_data_raw = self._data(series_name=series_def.name, label="raw_data")
            ymin = min(ymin, *curve_data_raw.y)
            ymax = max(ymax, *curve_data_raw.y)
            plot_scatter(xdata=curve_data_raw.x, ydata=curve_data_raw.y, ax=axis, zorder=0)

            # plot formatted data

            curve_data_fit = self._data(series_name=series_def.name, label="fit_ready")
            if np.all(np.isnan(curve_data_fit.y_err)):
                sigma = None
            else:
                sigma = np.nan_to_num(curve_data_fit.y_err)

            plot_errorbar(
                xdata=curve_data_fit.x,
                ydata=curve_data_fit.y,
                sigma=sigma,
                ax=axis,
                label=series_def.name,
                marker=series_def.plot_symbol,
                color=series_def.plot_color,
                zorder=1,
                linestyle="",
            )

            # plot fit curve

            if fit_available:
                plot_curve_fit(
                    func=series_def.fit_func,
                    result=result_data,
                    ax=axis,
                    color=series_def.plot_color,
                    zorder=2,
                    fit_uncertainty=series_def.plot_fit_uncertainty,
                )

        # format axis

        if len(self.__series__) > 1:
            axis.legend(loc="center right")
        axis.set_xlabel(self._get_option("xlabel"), fontsize=16)
        axis.set_ylabel(self._get_option("ylabel"), fontsize=16)
        axis.tick_params(labelsize=14)
        axis.grid(True)

        # automatic scaling y axis by actual data point.
        # note that y axis will be scaled by confidence interval by default.
        # sometimes we cannot see any data point if variance of parameters is too large.

        height = ymax - ymin
        axis.set_ylim(ymin - 0.1 * height, ymax + 0.1 * height)

        # write analysis report

        fit_reports = self._get_option("fit_reports")
        if fit_reports and fit_available:
            # write fit status in the plot
            analysis_description = ""
            for par_name, label in fit_reports.items():
                try:
                    # fit value
                    pval = get_opt_value(result_data, par_name)
                    perr = get_opt_error(result_data, par_name)
                except ValueError:
                    # maybe post processed value
                    pval = result_data[par_name]
                    perr = result_data[f"{par_name}_err"]
                analysis_description += f"{label} = {pval: .3e}\u00B1{perr: .3e}\n"
            chisq = result_data["reduced_chisq"]
            analysis_description += f"Fit \u03C7-squared = {chisq: .4f}"

            report_handler = axis.text(
                0.60,
                0.95,
                analysis_description,
                ha="center",
                va="top",
                size=14,
                transform=axis.transAxes,
            )

            bbox_props = dict(boxstyle="square, pad=0.3", fc="white", ec="black", lw=1, alpha=0.8)
            report_handler.set_bbox(bbox_props)

        return [figure]

    def _setup_fitting(self, **options) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """An analysis subroutine that is called to set fitter options.

        Subclasses can override this method to provide their own fitter options
        such as initial guesses.

        Create initial guesses from the experimental data provided by the ``self._data()`` method.
        If there are multiple series, you can get a specific series
        by specifying its ``series_name``.
        This function returns a ``CurveData`` instance, which is the `dataclass`
        containing x values `.x`, y values `.y`, and  sigma values `.y_err`.

        You can also access the defined analysis options with the ``self._get_option``.
        For example:

        .. code-block::

            curve_data = self._data(series_name="my_experiment1")

            if self._get_option("my_option1") == "abc":
                p0 = my_guess_function(curve_data.x, curve_data.y, ...)
                bounds = ...
            else:
                p0 = ...
                bounds = ...

            return {"p0": p0, "bounds": bounds}

        Note that this subroutine can generate multiple fit options.
        If multiple options are provided, fitter runs multiple times for each fit option,
        and find the best result measured by the reduced chi-squared value.

        .. code-block::

            fit_1 = {"p0": p0_1, "bounds": bounds, "extra_fit_parameter": "option1"}
            fit_2 = {"p0": p0_2, "bounds": bounds, "extra_fit_parameter": "option2"}

            return [fit_1, fit_2]

        Note that you can also change fitter options (not only initial guesses) in each
        fit condition. This might be convenient to fit parameter with multiple fit algorithms
        or different fitting options. By default, this class uses `scipy.curve_fit`
        as the fitter function. See Scipy API docs for more fitting option details.

        Args:
            options: User provided extra options that are not defined in default options.

        Returns:
            List of FitOptions that are passed to fitter function.
        """
        fit_options = {"p0": self._get_option("p0"), "bounds": self._get_option("bounds")}
        fit_options.update(options)

        return fit_options

    def _format_data(self, data: CurveData) -> CurveData:
        """An optional subroutine to perform data pre-processing.

        Subclasses can override this method to apply pre-precessing to data values to fit.
        Otherwise the analysis uses extracted data values as-is.

        For example,

        - Apply smoothing to y values to deal with noisy observed values
        - Remove redundant data points (outlier)
        - Apply frequency filter function

        etc...

        .. note::

            The data returned by this method should have the label "fit_ready".

        Returns:
            Formatted CurveData instance.
        """
        return CurveData(
            label="fit_ready",
            x=data.x,
            y=data.y,
            y_err=data.y_err,
            data_index=data.data_index,
            metadata=data.metadata,
        )

    def _post_analysis(self, result_data: CurveAnalysisResultData) -> CurveAnalysisResultData:
        """Calculate new quantity from the fit result.

        Subclasses can override this method to do post analysis.

        Args:
            result_data: Result containing fit result.

        Returns:
            Updated result data containing the result of post analysis.
        """
        return result_data

    def _extract_curves(
        self, experiment_data: ExperimentData, data_processor: Union[Callable, DataProcessor]
    ):
        """Extract curve data from experiment data.

        This method internally populate two type of curve data.

        - raw_data:

            This is the data directly obtained from the experiment data.
            You can access to this data with ``self._data(label="raw_data")``.

        - fit_ready:

            This is the formatted data created  pre-processing defined by
            `self._format_data()` method. This method is implemented by subclasses.
            You can access to this data with ``self._data(label="fit_ready")``.

        If multiple series exists, you can optionally specify ``series_name`` in
        ``self._data`` method to filter data in the target series.

        .. notes::
            The target metadata properties to define each curve entry is described by
            the class attribute __series__ (see `filter_kwargs`).

        Args:
            experiment_data: ExperimentData object to fit parameters.
            data_processor: A callable or DataProcessor instance to format data into numpy array.
                This should take list of dictionary and returns two tuple of float values
                that represent a y value and an error of it.
        Raises:
            DataProcessorError: When `x_key` specified in the analysis option is not
                defined in the circuit metadata.
            AnalysisError: When formatted data has the label other than fit_ready.
        """
        self.__processed_data_set = list()

        def _is_target_series(datum, **filters):
            try:
                return all(datum["metadata"][key] == val for key, val in filters.items())
            except KeyError:
                return False

        # Extract X, Y, Y_sigma data
        data = experiment_data.data()

        x_key = self._get_option("x_key")
        try:
            x_values = [datum["metadata"][x_key] for datum in data]
        except KeyError as ex:
            raise DataProcessorError(
                f"X value key {x_key} is not defined in circuit metadata."
            ) from ex

        if isinstance(data_processor, DataProcessor):
            y_values, y_sigmas = data_processor(data)
            if y_sigmas is None:
                y_sigmas = np.full(y_values.shape, np.nan)
        else:
            y_values, y_sigmas = zip(*map(data_processor, data))

        # Store metadata
        metadata = np.asarray([datum["metadata"] for datum in data], dtype=object)

        # Format data
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        y_sigmas = np.asarray(y_sigmas, dtype=float)

        # Find series (invalid data is labeled as -1)
        data_index = np.full(x_values.size, -1, dtype=int)
        for idx, series_def in enumerate(self.__series__):
            data_matched = np.asarray(
                [_is_target_series(datum, **series_def.filter_kwargs) for datum in data], dtype=bool
            )
            data_index[data_matched] = idx

        # Store raw data
        raw_data = CurveData(
            label="raw_data",
            x=x_values,
            y=y_values,
            y_err=y_sigmas,
            data_index=data_index,
            metadata=metadata,
        )
        self.__processed_data_set.append(raw_data)

        # Format raw data
        formatted_data = self._format_data(raw_data)
        if formatted_data.label != "fit_ready":
            raise AnalysisError(f"Not expected data label {formatted_data.label} != fit_ready.")
        self.__processed_data_set.append(formatted_data)

    def _format_fit_options(self, **fitter_options) -> Dict[str, Any]:
        """Format fitting option args to dictionary of parameter names.

        Args:
            fitter_options: Fit options generated by `self._setup_fitting`.

        Returns:
            Formatted fit options.

        Raises:
            AnalysisError:
                - When fit functions have different signature.
                - When fit option is dictionary but key doesn't match with parameter names.
                - When initial guesses are not provided.
                - When fit option is array but length doesn't match with parameter number.
        """
        # Remove any fixed parameter so as not to give them to the fitter.
        if self.__fixed_parameters__ is not None and len(self.__fixed_parameters__) > 0:
            for pname in self.__fixed_parameters__:
                fitter_options.pop(pname, None)

        # Validate dictionary keys
        def _check_keys(parameter_name):
            named_values = fitter_options[parameter_name]
            if not named_values.keys() == set(self.__fit_params):
                raise AnalysisError(
                    f"Fitting option `{parameter_name}` doesn't have the "
                    f"expected parameter names {','.join(self.__fit_params)}."
                )

        # Convert array into dictionary
        def _dictionarize(parameter_name):
            parameter_array = fitter_options[parameter_name]
            if len(parameter_array) != len(self.__fit_params):
                raise AnalysisError(
                    f"Value length of fitting option `{parameter_name}` doesn't "
                    "match with the length of expected parameters. "
                    f"{len(parameter_array)} != {len(self.__fit_params)}."
                )
            return dict(zip(self.__fit_params, parameter_array))

        if fitter_options.get("p0", None):
            if isinstance(fitter_options["p0"], dict):
                _check_keys("p0")
            else:
                fitter_options["p0"] = _dictionarize("p0")
        else:
            # p0 should be defined
            raise AnalysisError("Initial guess p0 is not provided to the fitting options.")

        if fitter_options.get("bounds", None):
            if isinstance(fitter_options["bounds"], dict):
                _check_keys("bounds")
            else:
                fitter_options["bounds"] = _dictionarize("bounds")
        else:
            # bounds are optional
            fitter_options["bounds"] = {par: (-np.inf, np.inf) for par in self.__fit_params}

        return fitter_options

    @property
    def _experiment_type(self) -> str:
        """Return type of experiment."""
        try:
            return self.__experiment_metadata["experiment_type"]
        except (TypeError, KeyError):
            # Ignore experiment metadata is not set or key is not found
            return None

    @property
    def _num_qubits(self) -> int:
        """Getter for qubit number."""
        try:
            return self.__experiment_metadata["num_qubits"]
        except (TypeError, KeyError):
            # Ignore experiment metadata is not set or key is not found
            return None

    @property
    def _physical_qubits(self) -> List[int]:
        """Getter for physical qubit indices."""
        try:
            return list(self.__experiment_metadata["physical_qubits"])
        except (TypeError, KeyError):
            # Ignore experiment metadata is not set or key is not found
            return None

    def _experiment_options(self, index: int = -1) -> Dict[str, Any]:
        """Return the experiment options of given job index.

        Args:
            index: Index of job metadata to extract. Default to -1 (latest).

        Returns:
            Experiment options. This option is used for circuit generation.
        """
        try:
            return self.__experiment_metadata["job_metadata"][index]["experiment_options"]
        except (TypeError, KeyError, IndexError):
            # Ignore experiment metadata or job metadata is not set or key is not found
            return None

    def _analysis_options(self, index: int = -1) -> Dict[str, Any]:
        """Returns the analysis options of given job index.

        Args:
            index: Index of job metadata to extract. Default to -1 (latest).

        Returns:
            Analysis options. This option is used for analysis.
        """
        try:
            return self.__experiment_metadata["job_metadata"][index]["analysis_options"]
        except (TypeError, KeyError, IndexError):
            # Ignore experiment metadata or job metadata is not set or key is not found
            return None

    def _run_options(self, index: int = -1) -> Dict[str, Any]:
        """Returns the run options of given job index.

        Args:
            index: Index of job metadata to extract. Default to -1 (latest).

        Returns:
            Run options. This option is used for backend execution.
        """
        try:
            return self.__experiment_metadata["job_metadata"][index]["run_options"]
        except (TypeError, KeyError, IndexError):
            # Ignore experiment metadata or job metadata is not set or key is not found
            return None

    def _transpile_options(self, index: int = -1) -> Dict[str, Any]:
        """Returns the transpile options of given job index.

        Args:
            index: Index of job metadata to extract. Default to -1 (latest).

        Returns:
            Transpile options. This option is used for circuit optimization.
        """
        try:
            return self.__experiment_metadata["job_metadata"][index]["transpile_options"]
        except (TypeError, KeyError, IndexError):
            # Ignore experiment metadata or job metadata is not set or key is not found
            return None

    def _data(
        self,
        series_name: Optional[str] = None,
        label: Optional[str] = "fit_ready",
    ) -> CurveData:
        """Getter for experiment data set.

        Args:
            series_name: Series name to search for.
            label: Label attached to data set. By default it returns "fit_ready" data.

        Returns:
            Filtered curve data set.

        Raises:
            AnalysisError: When requested series or label are not defined.
        """
        # pylint: disable = undefined-loop-variable
        for data in self.__processed_data_set:
            if data.label == label:
                break
        else:
            raise AnalysisError(f"Requested data with label {label} does not exist.")

        if series_name is None:
            return data

        for idx, series_def in enumerate(self.__series__):
            if series_def.name == series_name:
                locs = data.data_index == idx
                return CurveData(
                    label=label,
                    x=data.x[locs],
                    y=data.y[locs],
                    y_err=data.y_err[locs],
                    data_index=idx,
                    metadata=data.metadata[locs] if data.metadata is not None else None,
                )

        raise AnalysisError(f"Specified series {series_name} is not defined in this analysis.")

    def _arg_parse(self, **options) -> Dict[str, Any]:
        """Parse input kwargs with predicted input.

        Args:
            options: User-input keyword argument options.

        Returns:
            Keyword arguments that not specified in the default options.
        """
        extra_options = dict()
        for key, value in options.items():
            private_key = f"__{key}"
            if hasattr(self, private_key):
                setattr(self, private_key, value)
            else:
                extra_options[key] = value

        return extra_options

    def _get_option(self, arg_name: str) -> Any:
        """A helper function to get specified field from the input analysis options.

        Args:
            arg_name: Name of option.

        Return:
            Arbitrary object specified by the option name.

        Raises:
            AnalysisError:
                - When `arg_name` is not found in the analysis options.
        """
        try:
            return getattr(self, f"__{arg_name}")
        except AttributeError as ex:
            raise AnalysisError(
                f"The argument {arg_name} is selected but not defined. "
                "This key-value pair should be defined in the analysis option."
            ) from ex

    def _run_analysis(
        self, experiment_data: ExperimentData, **options
    ) -> Tuple[List["AnalysisResultData"], List["pyplot.Figure"]]:
        """Run analysis on circuit data.

        Args:
            experiment_data: the experiment data to analyze.
            options: kwarg options for analysis function.

        Returns:
            tuple: A pair ``(analysis_results, figures)`` where
                   ``analysis_results`` may be a single or list of
                   CurveAnalysisResultData objects, and ``figures`` is a list of any
                   figures for the experiment.

        Raises:
            AnalysisError: if the analysis fails.
        """
        result_data = CurveAnalysisResultData()
        result_data["analysis_type"] = self.__class__.__name__
        figures = list()

        #
        # 1. Parse arguments
        #
        if self.__fixed_parameters__ is not None and len(self.__fixed_parameters__) > 0:
            assigned_params = dict()
            # Extract fixed parameter value from analysis options
            for pname in self.__fixed_parameters__:
                try:
                    assigned_params[pname] = options[pname]
                except KeyError as ex:
                    raise AnalysisError(
                        f"The value of the fixed-value parameter {pname} for the fit function "
                        f"of {self.__class__.__name__} was not found. "
                        "This value must be provided by the analysis options to run this analysis."
                    ) from ex
            # Override series definition with assigned fit functions.
            assigned_series = []
            for series_def in self.__series__:
                dict_def = dataclasses.asdict(series_def)
                dict_def["fit_func"] = functools.partial(series_def.fit_func, **assigned_params)
                assigned_series.append(SeriesDef(**dict_def))
            self.__series__ = assigned_series

        # pop arguments that are not given to fitter
        extra_options = self._arg_parse(**options)

        # get experiment metadata
        try:
            self.__experiment_metadata = experiment_data.metadata()

        except AttributeError:
            pass

        #
        # 2. Setup data processor
        #

        # No data processor has been provided at run-time we infer one from the job
        # metadata and default to the data processor for averaged classified data.
        data_processor = self._get_option("data_processor")

        if not data_processor:
            run_options = self._run_options() or dict()

            try:
                meas_level = run_options["meas_level"]
            except KeyError as ex:
                raise AnalysisError(
                    f"Cannot process data without knowing the measurement level: {str(ex)}."
                ) from ex

            meas_return = run_options.get("meas_return", None)
            normalization = self._get_option("normalization")

            data_processor = get_processor(meas_level, meas_return, normalization)

        if isinstance(data_processor, DataProcessor) and not data_processor.is_trained:
            # Qiskit DataProcessor instance. May need calibration.
            try:
                data_processor.train(data=experiment_data.data())
            except DataProcessorError as ex:
                raise AnalysisError(
                    f"DataProcessor calibration failed with error message: {str(ex)}."
                ) from ex

        #
        # 3. Extract curve entries from experiment data
        #
        try:
            self._extract_curves(experiment_data=experiment_data, data_processor=data_processor)
        except DataProcessorError as ex:
            raise AnalysisError(
                f"Data extraction and formatting failed with error message: {str(ex)}."
            ) from ex

        #
        # 4. Run fitting
        #
        try:
            curve_fitter = self._get_option("curve_fitter")
            formatted_data = self._data(label="fit_ready")

            # Generate fit options
            fit_candidates = self._setup_fitting(**extra_options)

            # Fit for each fit parameter combination
            if isinstance(fit_candidates, dict):
                # Only single initial guess
                fit_options = self._format_fit_options(**fit_candidates)
                fit_result = curve_fitter(
                    funcs=[series_def.fit_func for series_def in self.__series__],
                    series=formatted_data.data_index,
                    xdata=formatted_data.x,
                    ydata=formatted_data.y,
                    sigma=formatted_data.y_err,
                    **fit_options,
                )
                result_data.update(**fit_result)
            else:
                # Multiple initial guesses
                fit_options_candidates = [
                    self._format_fit_options(**fit_options) for fit_options in fit_candidates
                ]
                fit_results = []
                for fit_options in fit_options_candidates:
                    try:
                        fit_result = curve_fitter(
                            funcs=[series_def.fit_func for series_def in self.__series__],
                            series=formatted_data.data_index,
                            xdata=formatted_data.x,
                            ydata=formatted_data.y,
                            sigma=formatted_data.y_err,
                            **fit_options,
                        )
                        fit_results.append(fit_result)
                    except AnalysisError:
                        pass
                if len(fit_results) == 0:
                    raise AnalysisError(
                        "All initial guesses and parameter boundaries failed to fit the data. "
                        "Please provide better initial guesses or fit parameter boundaries."
                    )
                # Sort by chi squared value
                fit_results = sorted(fit_results, key=lambda r: r["reduced_chisq"])
                result_data.update(**fit_results[0])

            result_data["success"] = True

        except AnalysisError as ex:
            result_data["error_message"] = str(ex)
            result_data["success"] = False

        else:
            #
            # 5. Post-process analysis data
            #
            result_data = self._post_analysis(result_data=result_data)

        finally:
            #
            # 6. Create figures
            #
            if self._get_option("plot") and HAS_MATPLOTLIB:
                figures.extend(self._create_figures(result_data=result_data))

            #
            # 7. Optionally store raw data points
            #
            if self._get_option("return_data_points"):
                raw_data_dict = dict()
                for series_def in self.__series__:
                    series_data = self._data(series_name=series_def.name, label="raw_data")
                    raw_data_dict[series_def.name] = {
                        "xdata": series_data.x,
                        "ydata": series_data.y,
                        "sigma": series_data.y_err,
                    }
                result_data["raw_data"] = raw_data_dict

        return [result_data], figures
