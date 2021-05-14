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
from typing import Any, Dict, List, Tuple, Callable, Union

import numpy as np
from qiskit.exceptions import QiskitError

from qiskit_experiments.analysis.curve_fitting import multi_curve_fit
from qiskit_experiments.analysis.data_processing import level2_probability
from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.data_processing import DataProcessor
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.experiment_data import AnalysisResult, ExperimentData


@dataclasses.dataclass
class SeriesDef:
    """Description of curve."""

    name: str
    fit_func: Callable
    filter_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)


class FitOptions(dict):
    """Fit options passed to the fitter function."""


class CurveAnalysis(BaseAnalysis):
    """A base class for curve fit type analysis.

    The subclasses can override class attributes to define the behavior of
    data extraction and fitting. This docstring describes how code developers can
    create a new curve fit analysis subclass inheriting from this base class.

    Class Attributes:

        __x_key__: Key in the circuit metadata under which to find the value for
            the horizontal axis.
        __processing_options__: Circuit metadata keys that are passed to the data processor.
            The key should conform to the data processor API.
        __series__: A set of data points that will be fit to a the same parameters
            in the fit function. If this analysis contains multiple curves,
            the same number of series definitions should be listed.
            Each series definition is SeriesDef element, that may be initialized with::

                name: Name of the curve. This is arbitrary data field, but should be unique.
                fit_func: Callback function to perform fit.
                filter_kwargs: Circuit metadata key and value associated with this curve.
                    The data points of the curve is extracted from ExperimentData based on
                    this information.

            See the Examples below for more details.
        __base_fitter__: A callable to perform curve fitting.
        __default_data_processor__: A callable to format y, y error data.

    Examples:

        A fitting for single exponential decay curve
        ============================================

        In this type of experiment, the analysis deals with a single curve.
        Thus filter_kwargs is not necessary defined.

        .. code-block::

            class AnalysisExample(CurveAnalysis):

                __x_key__ = "scan_val"

                __series__ = [
                    SeriesDef(
                        name="my_experiment1",
                        fit_func=lambda x, p0, p1, p2:
                            exponential_decay(x, amp=p0, lamb=p1, baseline=p2),
                    ),
                ]


        A fitting for two exponential decay curve with partly shared parameter
        ======================================================================

        In this type of experiment, the analysis deals with two curves.
        We need a __series__ definition for each curve, and filter_kwargs should be
        properly defined to separate each curve series.

        .. code-block::

            class AnalysisExample(CurveAnalysis):

                __x_key__ = "scan_val"

                __series__ = [
                    SeriesDef(
                        name="my_experiment1",
                        fit_func=lambda x, p0, p1, p2, p3:
                            exponential_decay(x, amp=p0, lamb=p1, baseline=p3),
                        filter_kwargs={"experiment": 1},
                    ),
                    SeriesDef(
                        name="my_experiment2",
                        fit_func=lambda x, p0, p1, p2, p3:
                            exponential_decay(x, amp=p0, lamb=p2, baseline=p3),
                        filter_kwargs={"experiment": 2},
                    ),
                ]


        A fitting for two trigonometric curves with the same parameter
        =============================================================

        In this type of experiment, the analysis deals with two different curves.
        However the parameters are shared with both functions.

        .. code-block::

            class AnalysisExample(CurveAnalysis):

                __x_key__ = "scan_val"

                __series__ = [
                    SeriesDef(
                        name="my_experiment1",
                        fit_func=lambda x, p0, p1, p2, p3:
                            cos(x, amp=p0, freq=p1, phase=p2, baseline=p3),
                        filter_kwargs={"experiment": 1},
                    ),
                    SeriesDef(
                        name="my_experiment2",
                        fit_func=lambda x, p0, p1, p2, p3:
                            sin(x, amp=p0, freq=p1, phase=p2, baseline=p3),
                        filter_kwargs={"experiment": 2},
                    )
                ]


    Notes:
        This CurveAnalysis class provides several private methods that subclasses can override.

        - Customize figure generation:
            Override :meth:`~self._create_figures`. For example, here you can create
            arbitrary number of new figures or upgrade the default figure appearance.

        - Customize pre-data processing:
            Override :meth:`~self._data_pre_processing`. For example, here you can
            take a mean over y values for the same x value, or apply smoothing to y values.

        - Customize post-analysis data processing:
            Override :meth:`~self._post_processing`. For example, here you can
            calculate new entity from fit values. Such as EPC of RB experiment.

        - Customize fitting options:
            Override :meth:`~self._setup_fitting`. For example, here you can
            calculate initial guess from experiment data and setup fitter options.

        - Customize data processor calibration:
            Override :meth:`~Self._calibrate_data_processor`. This is special subroutine
            that is only called when a DataProcessor instance is used as the data processor.
            You can take arbitrary data from experiment result and setup your processor.

        Note that other private methods are not expected to be overridden.
        If you forcibly override these methods, the behavior of analysis logic is not well tested
        and we cannot guarantee it works as expected (you may suffer from bugs).
        Instead, you can open an issue in qiskit-experiment github to upgrade this class
        with proper unittest framework.

        https://github.com/Qiskit/qiskit-experiments/issues
    """

    #: str: Metadata key representing a scanned value.
    __x_key__ = "xval"

    #: str: Metadata keys specifying data processing options.
    __processing_options__ = ["outcome"]

    #: List[SeriesDef]: List of mapping representing a data series
    __series__ = None

    # Callable: Default curve fitter. This can be overwritten.
    __base_fitter__ = multi_curve_fit

    # Union[Callable, DataProcessor]: Data processor to format experiment data.
    __default_data_processor__ = level2_probability

    # pylint: disable = unused-argument, missing-return-type-doc
    def _create_figures(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        y_sigmas: np.ndarray,
        series: np.ndarray,
        fit_data: AnalysisResult,
    ):
        """Create new figures with the fit result and raw data.

        Subclass can override this method to return figures.

        Args:
            x_values: Full data set of x values.
            y_values: Full data set of y values.
            y_sigmas: Full data set of y sigmas.
            series: An integer array representing a mapping of data location to series index.
            fit_data: Analysis result containing fit parameters.

        Returns:
            List of figures (format TBD).
        """
        # TODO implement default figure. Will wait for Qiskit-terra #5499
        return list()

    # pylint: disable = unused-argument
    def _setup_fitting(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        y_sigmas: np.ndarray,
        series: np.ndarray,
        **options,
    ) -> List[FitOptions]:
        """An analysis subroutine that is called to set fitter options.

        This subroutine takes full data array and user-input fit options.
        Subclasses can override this method to provide own fitter options
        such as initial guesses.

        Note that this subroutine can generate multiple fit options.
        If multiple options are provided, fitter runs multiple times for each fit option,
        and find the best result measured by the reduced chi-squared value.

        Args:
            x_values: Full data set of x values.
            y_values: Full data set of y values.
            y_sigmas: Full data set of y sigmas.
            series: An integer array representing a mapping of data location to series index.
            options: User provided fit options.

        Returns:
            List of FitOptions that are passed to fitter function.
        """
        fit_options = FitOptions(**options)

        return [fit_options]

    # pylint: disable = unused-argument
    @staticmethod
    def _calibrate_data_processor(
        data_processor: DataProcessor, experiment_data: ExperimentData
    ) -> DataProcessor:
        """An optional subroutine to perform data processor calibration.

        Subclass can override this method to calibrate data processor instance.
        This routine is called only when a DataProcessor instance is specified in the
        class attribute __default_data_processor__.

        Args:
            data_processor: Data processor instance to calibrate.
            experiment_data: Unfiltered experiment data set.

        Returns:
            Calibrated data processor instance.
        """
        return data_processor

    @staticmethod
    def _data_pre_processing(
        x_values: np.ndarray, y_values: np.ndarray, y_sigmas: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """An optional subroutine to perform data pre-processing.

        Subclasses can override this method to apply pre-precessing to data values to fit.
        Otherwise the analysis uses extracted data values as-is.

        For example,

        - Take mean over all y data values with the same x data value
        - Apply smoothing to y values to deal with noisy observed values

        Args:
            x_values: Numpy float array to represent X values.
            y_values: Numpy float array to represent Y values.
            y_sigmas: Numpy float array to represent Y errors.

        Returns:
            Numpy array tuple of pre-processed (x_values, y_values, y_sigmas).
        """
        return x_values, y_values, y_sigmas

    @staticmethod
    def _post_processing(analysis_result: AnalysisResult) -> AnalysisResult:
        """Calculate new quantity from the fit result.

        Subclasses can override this method to do post analysis.

        Args:
            analysis_result: Analysis result containing fit result.

        Returns:
            New AnalysisResult instance containing the result of post analysis.
        """
        return analysis_result

    def _extract_curves(
        self,
        experiment_data: ExperimentData,
        data_processor: Union[Callable, DataProcessor],
    ) -> Tuple[np.ndarray, ...]:
        """Extract curve data from experiment data.

        .. notes::
            The target metadata properties to define each curve entry is described by
            the class attribute __series__. This method returns the same numbers
            of curve data entries as one defined in this attribute.
            The returned CurveData entry contains circuit metadata fields that are
            common to the entire curve scan, i.e. series-level metadata.

        Args:
            experiment_data: ExperimentData object to fit parameters.
            data_processor: A callable or DataProcessor instance to format data into numpy array.
                This should take list of dictionary and returns two tuple of float values
                that represent a y value and an error of it.

        Returns:
            List of ``CurveEntry`` containing x-values, y-values, and y values sigma.

        Raises:
            QiskitError:
                - When __x_key__ is not defined in the circuit metadata.
            KeyError:
                - When circuit metadata doesn't provide required data processor options.
        """

        def _is_target_series(datum, **filters):
            try:
                return all(datum["metadata"][key] == val for key, val in filters.items())
            except KeyError:
                return False

        # Extract X, Y, Y_sigma data
        data = experiment_data.data()

        try:
            x_values = [datum["metadata"][self.__x_key__] for datum in data]
        except KeyError as ex:
            raise QiskitError(
                f"X value key {self.__x_key__} is not defined in circuit metadata."
            ) from ex

        def _data_processing(datum):
            # A helper function to receive data processor runtime option from metadata
            try:
                # Extract data processor options
                dp_options = {key: datum["metadata"][key] for key in self.__processing_options__}
            except KeyError as ex:
                raise KeyError(
                    "Required data processor options are not provided by circuit metadata."
                ) from ex
            return data_processor(datum, **dp_options)

        y_values, y_sigmas = zip(*map(_data_processing, data))

        # Format data
        x_values, y_values, y_sigmas = self._data_pre_processing(
            x_values=np.asarray(x_values, dtype=float),
            y_values=np.asarray(y_values, dtype=float),
            y_sigmas=np.asarray(y_sigmas, dtype=float),
        )

        # Find series (invalid data is labeled as -1)
        series = -1 * np.ones(x_values.size, dtype=int)
        for idx, series_def in enumerate(self.__series__):
            data_index = np.asarray(
                [_is_target_series(datum, **series_def.filter_kwargs) for datum in data], dtype=bool
            )
            series[data_index] = idx

        return x_values, y_values, y_sigmas, series

    def _format_fit_options(self, options: FitOptions) -> FitOptions:
        """Format fitting option args to dictionary of parameter names.

        Args:
            options: Generated fit options without tested.

        Returns:
            Formatted fit options.

        Raises:
            QiskitError:
                - When fit functions have different signature.
            KeyError:
                - When fit option is dictionary but key doesn't match with parameter names.
                - When initial guesses are not provided.
            ValueError:
                - When fit option is array but length doesn't match with parameter number.
        """
        # check fit function signatures
        fsigs = set()
        for series_def in self.__series__:
            fsigs.add(inspect.signature(series_def.fit_func))
        if len(fsigs) > 1:
            raise QiskitError(
                "Fit functions specified in the series definition have "
                "different function signature. They should receive "
                "the same parameter set for multi-objective function fit."
            )
        fit_params = list(list(fsigs)[0].parameters.keys())[1:]

        # Validate dictionaly keys
        def _check_keys(parameter_name):
            named_values = options[parameter_name]
            if not named_values.keys() == set(fit_params):
                raise KeyError(
                    f"Fitting option {parameter_name} doesn't have the "
                    f"expected parameter names {','.join(fit_params)}."
                )

        # Convert array into dictionary
        def _dictionarize(parameter_name):
            parameter_array = options[parameter_name]
            if len(parameter_array) != len(fit_params):
                raise ValueError(
                    f"Value length of fitting option {parameter_name} doesn't "
                    "match with the length of expected parameters. "
                    f"{len(parameter_array)} != {len(fit_params)}."
                )
            return dict(zip(fit_params, parameter_array))

        if "p0" in options:
            if isinstance(options["p0"], dict):
                _check_keys("p0")
            else:
                options["p0"] = _dictionarize("p0")
        else:
            raise KeyError("Initial guess p0 is not provided to the fitting options.")

        if "bounds" in options:
            if isinstance(options["bounds"], dict):
                _check_keys("bounds")
            else:
                options["bounds"] = _dictionarize("bounds")
        else:
            options["bounds"] = dict(zip(fit_params, [(-np.inf, np.inf)] * len(fit_params)))

        return options

    def _subset_data(
        self,
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        y_sigmas: np.ndarray,
        series: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        """A helper method to extract reduced set of data.

        Args:
            name: Series name to search for.
            x_values: Full data set of x values.
            y_values: Full data set of y values.
            y_sigmas: Full data set of y sigmas.
            series: An integer array representing a mapping of data location to series index.

        Returns:
            Tuple of x values, y values, y sigmas for the specific series.

        Raises:
            QiskitError:
                - When name is not defined in the __series__ definition.
        """
        for idx, series_def in enumerate(self.__series__):
            if series_def.name == name:
                data_index = series == idx
                return x_values[data_index], y_values[data_index], y_sigmas[data_index]
        raise QiskitError(f"Specified series {name} is not defined in this analysis.")

    def _run_analysis(
        self, data: ExperimentData, **options
    ) -> Tuple[List[AnalysisResult], List["Figure"]]:
        """Run analysis on circuit data.

        Args:
            experiment_data: the experiment data to analyze.
            options: kwarg options for analysis function.

        Returns:
            tuple: A pair ``(analysis_results, figures)`` where
                   ``analysis_results`` may be a single or list of
                   AnalysisResult objects, and ``figures`` is a list of any
                   figures for the experiment.

        Raises:

        """
        analysis_result = AnalysisResult()

        #
        # 1. Setup data processor
        #
        data_processor = options.pop("data_processor", self.__default_data_processor__)

        # TODO add ` and not data_processor.trained:`
        if isinstance(data_processor, DataProcessor):
            # Qiskit DataProcessor instance. May need calibration.
            try:
                data_processor = self._calibrate_data_processor(
                    data_processor=data_processor,
                    experiment_data=data,
                )
            except DataProcessorError as ex:
                analysis_result["error_message"] = str(ex)
                analysis_result["success"] = False
                return [analysis_result], list()
        else:
            # Callback function
            data_processor = data_processor.__func__

        #
        # 2. Extract curve entries from experiment data
        #
        # pylint: disable=broad-except
        try:
            xdata, ydata, sigma, series = self._extract_curves(data, data_processor)
        except Exception as ex:
            analysis_result["error_message"] = str(ex)
            analysis_result["success"] = False
            return [analysis_result], list()

        #
        # 3. Run fitting
        #
        # pylint: disable=broad-except
        try:
            # Generate fit options
            fit_options_set = [
                self._format_fit_options(fit_options)
                for fit_options in self._setup_fitting(xdata, ydata, sigma, series, **options)
            ]
            fit_results = [
                self.__base_fitter__.__func__(
                    funcs=[series_def.fit_func for series_def in self.__series__],
                    series=series,
                    xdata=xdata,
                    ydata=ydata,
                    sigma=sigma,
                    **fit_options,
                )
                for fit_options in fit_options_set
            ]
            # Sort by chi squared value
            fit_results = sorted(fit_results, key=lambda r: r["reduced_chisq"])

            # Returns best fit result
            analysis_result = fit_results[0]
            analysis_result["success"] = True
        except Exception as ex:
            analysis_result["error_message"] = str(ex)
            analysis_result["success"] = False

        #
        # 4. Post-process analysis data
        #
        analysis_result = self._post_processing(analysis_result)

        #
        # 5. Create figures
        #
        figures = self._create_figures(
            x_values=xdata, y_values=ydata, y_sigmas=sigma, series=series, fit_data=analysis_result
        )

        #
        # 6. Save raw data
        #
        raw_data_dict = dict()
        for series_def in self.__series__:
            sub_xdata, sub_ydata, sub_sigma = self._subset_data(
                name=series_def.name, x_values=xdata, y_values=ydata, y_sigmas=sigma, series=series
            )
            raw_data_dict[series_def.name] = {
                "xdata": sub_xdata,
                "ydata": sub_ydata,
                "sigma": sub_sigma,
            }
        analysis_result["raw_data"] = raw_data_dict

        return [analysis_result], figures
