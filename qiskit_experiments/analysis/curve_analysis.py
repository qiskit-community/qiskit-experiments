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

import functools
from abc import abstractmethod
from collections import defaultdict
from typing import Any, NamedTuple, Dict, List, Optional, Tuple, Callable

import numpy as np
import scipy.optimize as opt
from qiskit.exceptions import QiskitError

from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.experiment_data import AnalysisResult, ExperimentData
from qiskit_experiments.analysis.data_processing import level2_probability

# Description of data properties for single curve entry
SeriesDef = NamedTuple(
    "SeriesDef",
    [
        ("name", str),
        ("param_names", List[str]),
        ("fit_func_index", int),
        ("filter_kwargs", Optional[Dict[str, Any]]),
    ],
)

# Human readable data set for single curve entry
CurveEntry = NamedTuple(
    "CurveEntry",
    [
        ("curve_name", str),
        ("x_values", np.ndarray),
        ("y_values", np.ndarray),
        ("y_sigmas", np.ndarray),
        ("metadata", dict),
    ],
)


def scipy_curve_fit_wrapper(
    func: Callable,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    **kwargs,
) -> AnalysisResult:
    r"""A helper function to perform a non-linear least squares to fit

    This solves the optimization problem

    .. math::
        \Theta_{\mbox{opt}} = \arg\min_\Theta \sum_i \sigma_i^{-2} (f(x_i, \Theta) -  y_i)^2

    using ``scipy.optimize.curve_fit``.

    Args:
        func: a fit function `f(x, *params)`.
        xdata: a 1D float array of x-data.
        ydata: a 1D float array of y-data.
        p0: initial guess for optimization parameters.
        sigma: Optional, a 1D array of standard deviations in ydata in absolute units.
        bounds: Optional, lower and upper bounds for optimization parameters.
        kwargs: additional kwargs for scipy.optimize.curve_fit.

    Returns:
        result containing ``popt`` the optimal fit parameters,
        ``popt_err`` the standard error estimates popt,
        ``pcov`` the covariance matrix for the fit,
        ``reduced_chisq`` the reduced chi-squared parameter of fit,
        ``dof`` the degrees of freedom of the fit,
        ``xrange`` the range of xdata values used for fit.

    Raises:
        QiskitError: if the number of degrees of freedom of the fit is
                     less than 1.

    .. note::
        ``sigma`` is assumed to be specified in the same units as ``ydata``
        (absolute units). If sigma is instead specified in relative units
        the `absolute_sigma=False` kwarg of scipy curve_fit must be used.
        This affects the returned covariance ``pcov`` and error ``popt_err``
        parameters via ``pcov(absolute_sigma=False) = pcov * reduced_chisq``
        ``popt_err(absolute_sigma=False) = popt_err * sqrt(reduced_chisq)``.
    """
    # Check the degrees of freedom is greater than 0
    dof = len(ydata) - len(p0)
    if dof < 1:
        raise QiskitError(
            "The number of degrees of freedom of the fit data and model "
            " (len(ydata) - len(p0)) is less than 1"
        )

    # Override scipy.curve_fit default for absolute_sigma=True
    # if sigma is specified.
    if sigma is not None and "absolute_sigma" not in kwargs:
        kwargs["absolute_sigma"] = True

    # Run curve fit
    # pylint: disable = unbalanced-tuple-unpacking
    popt, pcov = opt.curve_fit(
        f=func, xdata=xdata, ydata=ydata, sigma=sigma, p0=p0, bounds=bounds, **kwargs
    )
    popt_err = np.sqrt(np.diag(pcov))

    # Calculate the reduced chi-squared for fit
    yfits = func(xdata, *popt)
    residues = (yfits - ydata) ** 2
    if sigma is not None:
        residues = residues / (sigma ** 2)
    reduced_chisq = np.sum(residues) / dof

    # Compute xdata range for fit
    xdata_range = [min(xdata), max(xdata)]

    result = {
        "popt": popt,
        "popt_err": popt_err,
        "pcov": pcov,
        "reduced_chisq": reduced_chisq,
        "dof": dof,
        "xrange": xdata_range,
    }

    return AnalysisResult(result)


class CurveAnalysis(BaseAnalysis):
    """A base class for curve fit type analysis.

    The subclass can override class attributes to define the behavior of
    data extraction and fitting. This docstring describes how code developers can
    create a new curve fit analysis subclass inheriting from this base class.

    Class Attributes:

        __x_key__: String representation of horizontal axis.
            This should be defined in the circuit metadata for data extraction.
        __series__: List of curve property definitions. Each element should be
            defined as SeriesDef entry. This field can be left as None if the
            analysis is performed for only single line.
        __data_processor__: A callable to define the data processing procedure
            to extract curve series. This function should return x-values, y-values, and
            y-errors in numpy array format.
        __fit_funcs__: List of callable to fit parameters. This is order sensitive.
            The list index corresponds to the function index specified by __series__ definition.
        __param_names__: Name of parameters to fit. This is order sensitive.
        __base_fitter__: A callable to perform single curve fitting.
            The function API should conform to the scipy curve fit module.

    Examples:

        T1 experiment
        =============

        In this type of experiment, the analysis deals with single curve.
        Thus __series__ is not necessary be assigned.

        .. code-block::

            class AnalysisExample(CurveAnalysis):

                __x_key__ = "delay"

                __fit_funcs__ = [library.exponential]

                __param_names__ = ["a", "tau", "b"]

        IRB experiment
        ==============

        In this type of experiment, the analysis deals with two curves.
        We need __series__ definition for each curve.
        Both curves can be represented by the same exponential function,
        but with different parameter set. Note that parameters will be partly shared.

        .. code-block::

            class AnalysisExample(CurveAnalysis):

                __x_key__ = "ncliffs"

                __series__ = [
                    SeriesDef(
                        name="standard_rb",
                        param_names=["a", "alpha_std", "b"],
                        fit_func_index=0,
                        filter_kwargs={"interleaved": False}
                    ),
                    SeriesDef(
                        name="interleaved_rb",
                        param_names=["a", "alpha_int", "b"],
                        fit_func_index=0,
                        filter_kwargs={"interleaved": True}
                    )
                ]

                __fit_funcs__ = [library.exponential]

                __param_names__ = ["a", "alpha_std", "alpha_int", "b"]

        Note that the subclass can optionally override :meth:``_post_processing``.
        This method takes fit analysis result and calculate new entity with it.
        EPC calculation can be performed here.

        Ramsey XY experiment
        ====================

        In this type of experiment, the analysis deals with two curves.
        We need __series__ definition for each curve.
        In contrast to the IRB example, this experiment may have two fit functions
        to represent cosinusoidal (real part) and sinusoidal (imaginary part) oscillation,
        however the parameters are shared with both functions.

        .. code-block::

            class AnalysisExample(CurveAnalysis):

                __x_key__ = "delays"

                __series__ = [
                    SeriesDef(
                        name="x",
                        param_names=["a", "freq", "phase", "b"],
                        fit_func_index=0,
                        filter_kwargs={"pulse": "x"}
                    ),
                    SeriesDef(
                        name="y",
                        param_names=["a", "freq", "phase", "b"],
                        fit_func_index=1,
                        filter_kwargs={"pulse": "y"}
                    )
                ]

                __fit_funcs__ = [library.cos, library.sin]

                __param_names__ = ["a", "freq", "phase", "b"]

    .. notes::
        This class provides several private methods that subclasses can override.

        - _run_fitting: Central method to perform fitting with the provided list of curve data.
            Subclasses can create initial guess of parameters and override
            fitter analysis options here.
            Note that each curve data provides circuit metadata that may be useful to
            calculate initial guess or apply some coefficient to values.

        - _create_figure: A method to create figures. Subclasses can override this method
            to create figures. Both raw data and fit analysis is provided.

        - _post_processing: A method to calculate new entity from fit result.
            This returns fit result as-is by default.
    """

    #: str: Metadata key representing a scanned value.
    __x_key__ = ""

    #: List[SeriesDef]: List of mapping representing a data series
    __series__ = None

    #: Callable: Data processor. This should return x-values, y-values, y-sigmas.
    __data_processor__ = level2_probability

    # TODO this should be replaced with preset DataProcessor node.

    # TODO data processor may be initialized with some variables.
    # For example, if it contains front end discriminator, it may be initialized with
    # some discrimination line parameters. These parameters cannot be hard-coded here.

    #: List[Callable]: A callback function to define the expected curve
    __fit_funcs__ = None

    #: List[str]: Parameter name list
    __param_names__ = list()

    # Callable: Default curve fitter. This can be overwritten.
    __base_fitter__ = scipy_curve_fit_wrapper

    def _run_fitting(self, curve_data: List[CurveEntry], **options) -> AnalysisResult:
        """Fit series of curve data.

        Subclass can override this method to return figures.
        For example, initial guess is not automatically provided by this base class.

        Args:
            curve_data: List of raw curve data points to fit.
            **options: Fitting options.

        Returns:
            Analysis result populated by fit parameters.
        """
        return self._series_curve_fit(curve_data=curve_data, **options)

    @abstractmethod
    def _create_figure(self, curve_data: List[CurveEntry], fit_data: AnalysisResult):
        """Create new figure with the fit result and raw data.

        Subclass can override this method to return figures.

        Args:
            curve_data: List of raw curve data points with metadata.
            fit_data: Analysis result containing fit parameters.

        Returns:
            List of figures (format TBD).
        """
        pass

    @staticmethod
    def _post_processing(analysis_result: AnalysisResult) -> AnalysisResult:
        """Calculate new quantity from the fit result.

        Subclass can override this method to do post analysis.

        Args:
            analysis_result: Analysis result containing fit result.

        Returns:
            New AnalysisResult instance containing the result of post analysis.
        """
        return analysis_result

    def _data_processing(self, experiment_data: ExperimentData) -> List[CurveEntry]:
        """Extract curve data from experiment data.

        .. notes::
            The target metadata properties to define each curve entry is described by
            the class attribute __series__. This method returns the same numbers
            of curve data entries as one defined in this attribute.
            The returned CurveData entry contains circuit metadata fields that are
            common to the entire curve scan, i.e. series-level metadata.

        Args:
            experiment_data: ExperimentData object to fit parameters.

        Returns:
            List of ``CurveEntry`` containing x-values, y-values, and y values sigma.
        """
        if self.__series__:
            series = self.__series__
        else:
            series = [
                SeriesDef(
                    name="fit-curve-0",
                    param_names=self.__param_names__,
                    fit_func_index=0,
                    filter_kwargs=None,
                )
            ]

        def _is_target_series(datum, **filters):
            try:
                return all(datum["metadata"][key] == val for key, val in filters.items())
            except KeyError:
                return False

        curve_data = list()
        for curve_properties in series:
            if curve_properties.filter_kwargs:
                # filter data
                series_data = [
                    datum
                    for datum in experiment_data.data
                    if _is_target_series(datum, **curve_properties.filter_kwargs)
                ]
            else:
                # use data as-is
                series_data = experiment_data
            xvals, yvals, sigmas = self.__data_processor__(series_data)
            # TODO data processor may need calibration.
            # If we use the level1 data, it may be necessary to calculate principal component
            # with entire scan data. Otherwise we need to use real or imaginary part.

            # Get common metadata fields except for xval and filter args.
            # These properties are obvious.
            common_keys = list(
                functools.reduce(
                    lambda k1, k2: k1 & k2,
                    map(lambda d: d.keys(), [datum["metadata"] for datum in series_data]),
                )
            )
            common_keys.remove(self.__x_key__)
            if curve_properties.filter_kwargs:
                for key in curve_properties.filter_kwargs:
                    common_keys.remove(key)

            # Extract common metadata for the curve
            curve_metadata = defaultdict(set)
            for datum in series_data:
                for key in common_keys:
                    curve_metadata[key].add(datum["metadata"][key])

            curve_data.append(
                CurveEntry(
                    curve_name=curve_properties.name,
                    x_values=xvals,
                    y_values=yvals,
                    y_sigmas=sigmas,
                    metadata=curve_metadata,
                )
            )

        return curve_data

    def _series_curve_fit(
        self,
        curve_data: List[CurveEntry],
        p0: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **options,
    ) -> AnalysisResult:
        r"""Perform a linearized multi-objective non-linear least squares fit.

        This solves the optimization problem

        .. math::
            \Theta_{\mbox{opt}} = \arg\min_\Theta \sum_{k} w_k
                \sum_{i} \sigma_{k, i}^{-2}
                (f_k(x_{k, i}, \Theta) -  y_{k, i})^2

        for multiple series of :math:`x_k, y_k, \sigma_k` data evaluated using
        a list of objective functions :math:`[f_k]`
        using ``scipy.optimize.curve_fit``.

        Args:
            p0: initial guess for optimization parameters.
            weights: Optional, a 1D float list of weights :math:`w_k` for each
                     component function :math:`f_k`.
            bounds: Optional, lower and upper bounds for optimization
                    parameters.
            options: additional kwargs for scipy.optimize.curve_fit.

        Returns:
            result containing ``popt`` the optimal fit parameters,
            ``popt_err`` the standard error estimates popt,
            ``pcov`` the covariance matrix for the fit,
            ``reduced_chisq`` the reduced chi-squared parameter of fit,
            ``dof`` the degrees of freedom of the fit,
            ``xrange`` the range of xdata values used for fit.

        Raises:
            QiskitError:
                - When number of weights are not identical to the curve_data entries.
        """
        num_curves = len(curve_data)

        if weights is None:
            sig_weights = np.ones(num_curves)
        else:
            if len(weights) != num_curves:
                raise QiskitError(
                    "weights should be the same length as the curve_data. "
                    f"{len(weights)} != {num_curves}"
                )
            sig_weights = weights

        # Concatenate all curve data
        flat_xvals = np.empty(0, dtype=float)
        flat_yvals = np.empty(0, dtype=float)
        flat_yerrs = np.empty(0, dtype=float)
        separators = np.empty(num_curves - 1)
        for idx, (datum, weight) in enumerate(zip(curve_data, sig_weights)):
            flat_xvals = np.concatenate((flat_xvals, datum.x_values))
            flat_yvals = np.concatenate((flat_yvals, datum.y_values))
            if datum.y_sigmas is not None:
                datum_yerrs = datum.y_sigmas / np.sqrt(weight)
            else:
                datum_yerrs = 1 / np.sqrt(weight)
            flat_yerrs = np.concatenate((flat_yerrs, datum_yerrs))
            separators[idx] = len(datum.x_values)
        separators = np.cumsum(separators)[:-1]

        # Define multi-objective function
        def multi_objective_fit(x, *params):
            y = []
            xs = np.split(x, separators)
            for i, xi in range(num_curves, xs):
                yi = self._fit_curve(curve_data[i].curve_name, xi, *params)
                y.append(yi)
            return np.asarray(y, dtype=float)

        # To make sure argument mapping for user defined fit module.
        fitter_args = {
            "func": multi_objective_fit,
            "xdata": flat_xvals,
            "ydata": flat_yvals,
            "p0": p0,
            "sigma": flat_yerrs,
            "bounds": bounds
        }
        fitter_args.update(options)

        # pylint: disable=redundant-keyword-arg
        analysis_result = self.__base_fitter__(**fitter_args)
        analysis_result["popt_keys"] = self.__param_names__

        return analysis_result

    def _fit_curve(
            self,
            curve_name: str,
            xvals: np.ndarray,
            *params
    ) -> np.ndarray:
        """A helper method to run fitting with series definition.

        Fit function is selected based on ``curve_name`` and the parameters list is truncated
        based on parameter matching between one defined in __series__ and self.__param_names__.

        Examples:
            Assuming the class has following definition:

            .. code-block::

                self.__series__ = [
                    Series(name="curve1", param_names=["p1", "p2", "p4"], fit_func_index=0),
                    Series(name="curve2", param_names=["p1", "p2", "p3"], fit_func_index=1)
                ]

                self.__fit_funcs__ = [func1, func2]

                self.__param_names__ = ["p1", "p2", "p3", "p4"]

            When we call this method with ``curve_name="curve1", params = [0, 1, 2, 3]``,
            the ``func1`` is called with parameters ``[0, 1, 3]``.

        Args:
            curve_name: A name of curve. This should be defined in __series__ attribute.
            xvals: Array of x values.
            *params: Full fit parameters.

        Returns:
            Fit y values.

        Raises:
            QiskitError:
                - When function parameter is not defined in the class parameter list.
                - When fit function index is out of range.
                - When curve information is not defined in class attribute __series__.
        """
        for curve_properties in self.__series__:
            if curve_properties.name == curve_name:

                # remap parameters
                series_params = curve_properties.param_names
                mapped_params = []
                for series_param in series_params:
                    try:
                        param_idx = self.__param_names__.index(series_param)
                    except ValueError as ex:
                        raise QiskitError(
                            f"Local function parameter {series_param} is not defined in "
                            f"this class. {series_param} not in {self.__param_names__}."
                        ) from ex
                    mapped_params.append(params[param_idx])

                # find fit function
                f_index = curve_properties.fit_func_index
                try:
                    return self.__fit_funcs__[f_index](xvals, *mapped_params)
                except IndexError as ex:
                    raise QiskitError(f"Fit function of index {f_index} is not defined.") from ex

        raise QiskitError(f"A curve {curve_name} is not defined in this class.")

    def _run_analysis(self, experiment_data: ExperimentData, **options):
        """Run analysis on circuit data.

        Args:
            experiment_data: the experiment data to analyze.
            options: kwarg options for analysis function.

        Returns:
            tuple: A pair ``(analysis_results, figures)`` where
                   ``analysis_results`` may be a single or list of
                   AnalysisResult objects, and ``figures`` may be
                   None, a single figure, or a list of figures.
        """
        analysis_result = AnalysisResult()

        # Extract curve entries from experiment data
        try:
            curve_data = self._data_processing(experiment_data)
            analysis_result["raw_data"] = curve_data
        except DataProcessorError as ex:
            analysis_result["error_message"] = str(ex)
            analysis_result["success"] = False
            return analysis_result, list()

        # Run fitting
        # pylint: disable=broad-except
        try:
            fit_data = self._run_fitting(curve_data=curve_data, **options)
            analysis_result.update(fit_data)
            analysis_result["success"] = True
        except Exception as ex:
            analysis_result["error_message"] = str(ex)
            analysis_result["success"] = False

        # Post-process analysis data
        analysis_result = self._post_processing(analysis_result)

        # Create figures
        figures = self._create_figure(curve_data=curve_data, fit_data=analysis_result)

        # Store raw data
        raw_data = dict()
        for datum in curve_data:
            raw_data[datum.curve_name] = {
                "x_values": datum.x_values,
                "y_values": datum.y_values,
                "y_sigmas": datum.y_sigmas,
            }
        analysis_result["raw_data"] = raw_data

        return analysis_result, figures
