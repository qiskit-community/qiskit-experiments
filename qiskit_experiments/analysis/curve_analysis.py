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
from collections import defaultdict
from typing import Any, NamedTuple, Dict, List, Optional, Tuple, Callable, Union

import numpy as np
import scipy.optimize as opt
from qiskit.exceptions import QiskitError

from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.data_processing import DataProcessor
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.experiment_data import AnalysisResult, ExperimentData

# Description of data properties for single curve entry
SeriesDef = NamedTuple(
    "SeriesDef",
    [
        ("name", str),
        ("p0_signature", Dict[str, str]),
        ("fit_func_index", int),
        ("filter_kwargs", Optional[Dict[str, Any]]),
        ("data_option_keys", Optional[List[str]]),
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
    f: Callable,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: Dict[str, float],
    sigma: np.ndarray,
    bounds: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
    **kwargs,
) -> AnalysisResult:
    r"""A helper function to perform a non-linear least squares to fit

    This solves the optimization problem

    .. math::
        \Theta_{\mbox{opt}} = \arg\min_\Theta \sum_i \sigma_i^{-2} (f(x_i, \Theta) -  y_i)^2

    using ``scipy.optimize.curve_fit``.

    Args:
        f: a fit function `f(x, *params)`.
        xdata: a 1D float array of x-data.
        ydata: a 1D float array of y-data.
        p0: initial guess for optimization parameters.
        sigma: a 1D array of standard deviations in ydata in absolute units.
        bounds: lower and upper bounds for optimization parameters.
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
    # Format p0 parameters
    param_keys = list(p0.keys())
    param_p0 = list(p0.values())

    lower = [bounds[0][key] for key in param_keys]
    upper = [bounds[1][key] for key in param_keys]
    param_bounds = (lower, upper)

    # Check that degrees of freedom is greater than 0
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
        f=f, xdata=xdata, ydata=ydata, sigma=sigma, p0=param_p0, bounds=param_bounds, **kwargs
    )
    popt_err = np.sqrt(np.diag(pcov))

    # Calculate the reduced chi-squared for fit
    yfits = f(xdata, *popt)
    residues = (yfits - ydata) ** 2
    if sigma is not None:
        residues = residues / (sigma ** 2)
    reduced_chisq = np.sum(residues) / dof

    # Compute xdata range for fit
    xdata_range = [min(xdata), max(xdata)]

    result = {
        "popt": popt,
        "popt_keys": param_keys,
        "popt_err": popt_err,
        "pcov": pcov,
        "reduced_chisq": reduced_chisq,
        "dof": dof,
        "xrange": xdata_range,
    }

    return AnalysisResult(result)


def level2_probability(data: Dict[str, Any], outcome: Optional[str] = None) -> Tuple[float, float]:
    """Return the outcome probability mean and variance.

    Args:
        data: A data dict containing count data.
        outcome: bitstring for desired outcome probability.

    Returns:
        (p_mean, p_var) of the probability mean and variance estimated from the counts.

    .. note::

        This assumes a binomial distribution where :math:`K` counts
        of the desired outcome from :math:`N` shots the
        mean probability is :math:`p = K / N` and the variance is
        :math:`\\sigma^2 = p (1-p) / N`.
    """
    # TODO fix sigma definition
    # When the count is 100% zero (i.e. simulator), this yields sigma=0.
    # This crashes scipy fitter when it calculates covariance matrix (zero-div error).

    counts = data["counts"]
    outcome = outcome or "1" * len(list(counts.keys())[0])

    shots = sum(counts.values())
    p_mean = counts.get(outcome, 0.0) / shots
    p_var = p_mean * (1 - p_mean) / shots
    return p_mean, p_var


class FitOptions(dict):
    """Fit options passed to the fitter function."""

    def validate(self):
        """Validate format of fitting options.

        Raises:
            TypeError:
                - When parameter is not given by dictionary. Need mapping to parameter name.
            KeyError:
                - When necessary options are not fully defined.
        """

        # Initial guess
        try:
            p0 = self["p0"]
            if not isinstance(p0, dict):
                raise TypeError(
                    "Initial guess is not a dictionary. Need parameter name "
                    "associated with each parameter as a key."
                )
        except KeyError as ex:
            raise KeyError("Initial parameters are not set.") from ex

        # Boundaries
        try:
            lb, ub = self["bounds"]
            if not isinstance(lb, dict) or not isinstance(ub, dict):
                raise TypeError(
                    "Parameter boundaries are not a dictionary. Need parameter name "
                    "associated with each parameter as a key."
                )

        except KeyError as ex:
            raise KeyError("Parameter boundaries are not set.") from ex

        except TypeError as ex:
            raise TypeError(
                "Parameter boundaries are invalid format. Boundaries should be "
                "lower and upper values as a tuple of two dictionary."
            ) from ex


class CurveAnalysis(BaseAnalysis):
    """A base class for curve fit type analysis.

    The subclasses can override class attributes to define the behavior of
    data extraction and fitting. This docstring describes how code developers can
    create a new curve fit analysis subclass inheriting from this base class.

    Class Attributes:

        __x_key__: Key in the circuit metadata under which to find the value for
            the horizontal axis.
        __series__: A set of data points that will be fit to a the same parameters
            in the fit function. If this analysis contains multiple curves,
            the same number of series definitions should be listed.
            Each series definition is SeriesDef element, that may be initialized with::

                name: Name of the curve. This is arbitrary field.
                p0_signature: Mapping of parameter name to argument of the fit function
                    specified by fit_func_index. The parameter names (dict values)
                    should be defined in __param_names__.
                fit_func_index: Index of fitting function associated with this curve.
                    The fitting function should be listed in __fit_funcs__.
                filter_kwargs: Circuit metadata key and value associated with this curve.
                    The data points of the curve is extracted from ExperimentData based on
                    this information.
                data_option_keys: Circuit metadata keys that are passed to the data processor.
                    The key should conform to the data processor API.

            See the Examples below for more details.
        __fit_funcs__: List of callables to fit parameters. This is order sensitive.
            The list index corresponds to the function index specified by __series__ definition.
        __param_names__: Name of parameters to fit. This is order sensitive.
        __base_fitter__: A callable to perform single curve fitting.
            The function API should conform to the scipy curve fit module.

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
                        p0_signature={"amp": "p0", "lamb": "p1", "baseline": "p2"},
                        fit_func_index=0,
                        filter_kwargs=None,
                        data_option_keys=["outcome"]
                    ),
                ]

                __fit_funcs__ = [fit_functions.exponential_decay]

                __param_names__ = ["p0", "p1", "p2"]

        The signature of fit_functions.exponential may be::

        .. code-block::

            def exponential_decay(x, amp, lamb, base, x0, baseline) -> np.ndarray:

        The p0_signature key represents a mapping of the fit parameters given by the analysis to
        the arguments of the fit function. In this example, fit parameter "p0" is substituted
        in "amp" of the exponential_decay function. Note that the fit function defines a
        default value for each parameter, thus the default values are used for
        unspecified parameters here, i.e. "base" and "x0".


        A fitting for two exponential decay curve with partly shared parameter
        ======================================================================

        In this type of experiment, the analysis deals with two curves.
        We need a __series__ definition for each curve, and filter_kwargs should be
        properly defined to extract each curve data from the entire experiment data.

        .. code-block::

            class AnalysisExample(CurveAnalysis):

                __x_key__ = "scan_val"

                __series__ = [
                    SeriesDef(
                        name="my_experiment1",
                        p0_signature={"amp": "p0", "lamb": "p1", "baseline": "p3"},
                        fit_func_index=0,
                        filter_kwargs={"my_option1": True},
                        data_option_keys=["outcome"]
                    ),
                    SeriesDef(
                        name="my_experiment2",
                        p0_signature={"amp": "p0", "lamb": "p2", "baseline": "p3"},
                        fit_func_index=0,
                        filter_kwargs={"my_option1": False},
                        data_option_keys=["outcome"]
                    ),
                ]

                __fit_funcs__ = [fit_functions.exponential_decay]

                __param_names__ = ["p0", "p1", "p2", "p3"]

        Note that two series share the fit function exponential_decay.
        However, these series fit two curves with different "lamb" parameters.
        In total, there are 4 parameters defined in __param_names__.

        The series 1 (my_experiment1) performs::

        .. code-block::

            def exponential_decay(x, amp=p0, lamb=p1, baseline=p3) -> np.ndarray:

        The series 2 (my_experiment1) performs::

        .. code-block::

            def exponential_decay(x, amp=p0, lamb=p2, baseline=p3) -> np.ndarray:

        Thus both curves assume the same "amp" and "baseline".
        This parameter remapping is managed by the base analysis class behind the scene.


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
                        param_names={"amp": "p0", "freq": "p1", "phase": "p3", "baseline": "p4"},
                        fit_func_index=0,
                        filter_kwargs={"my_option1": "X"},
                        data_option_keys=["outcome"]
                    ),
                    SeriesDef(
                        name="my_experiment2",
                        param_names={"amp": "p0", "freq": "p1", "phase": "p3", "baseline": "p4"},
                        fit_func_index=1,
                        filter_kwargs={"my_option1": "Y"},
                        data_option_keys=["outcome"]
                    )
                ]

                __fit_funcs__ = [fit_functions.cos, fit_functions.sin]

                __param_names__ = ["p0", "p1", "p3", "p4"]

        The signature of each fit function may be::

        .. code-block::

            def cos(x, amp, freq, phase, baseline) -> np.ndarray:

            def sin(x, amp, freq, phase, baseline) -> np.ndarray:

        Note that series 1 (2) is linked to fit_functions.cos (sin) by the fit_func_index.
        The parameters are totally shared with two curves.

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

    #: List[SeriesDef]: List of mapping representing a data series
    __series__ = None

    #: List[Callable]: A callback function to define the expected curve
    __fit_funcs__ = None

    #: List[str]: Parameter name list
    __param_names__ = list()

    # Callable: Default curve fitter. This can be overwritten.
    __base_fitter__ = scipy_curve_fit_wrapper

    # Union[Callable, DataProcessor]: Data processor to format experiment data.
    __default_data_processor__ = level2_probability

    # pylint: disable = unused-argument, missing-return-type-doc
    def _create_figures(self, curve_data: List[CurveEntry], fit_data: AnalysisResult):
        """Create new figures with the fit result and raw data.

        Subclass can override this method to return figures.

        Args:
            curve_data: List of raw curve data points with metadata.
            fit_data: Analysis result containing fit parameters.

        Returns:
            List of figures (format TBD).
        """
        # TODO implement default figure. Will wait for Qiskit-terra #5499
        return list()

    # pylint: disable = unused-argument
    def _setup_fitting(self, curve_data: List[CurveEntry], **options) -> List[FitOptions]:
        """Setup initial guesses, fit boundaries and other options passed to optimizer.

        Subclass can override this method to provide proper optimization options.

        .. notes::
            This method returns list of FitOptions dictionary, and the options are
            passed to the optimizer as a keyword arguments.
            This should conform to the API which you specified in __base_fitter__.
            This defaults to scipy curve_fit. If you create multiple FitOptions dictionaries,
            fit is performed with each FitOptions and the fit result with the minimum
            `reduced_chisq` will be returned as a final result.

        Args:
            curve_data: List of raw curve data points to fit.
            options: User provided fit options.

        Returns:
            List of FitOptions that are passed to fitter function.
        """

        def _dictionarize(argvar, default_val):
            if argvar is not None:
                try:
                    argvar = list(argvar)
                    return dict(zip(self.__param_names__, argvar))
                except TypeError:
                    return argvar
            else:
                return {pname: default_val for pname in self.__param_names__}

        # Set initial guess
        usr_p0 = options.pop("p0", None)
        p0 = _dictionarize(usr_p0, default_val=0.0)

        # Set boundaries
        usr_lb, usr_ub = options.pop("bounds", (None, None))
        lb = _dictionarize(usr_lb, default_val=-np.inf)
        ub = _dictionarize(usr_ub, default_val=np.inf)

        fit_option = FitOptions(p0=p0, bounds=(lb, ub), **options)

        return [fit_option]

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
    ) -> List[CurveEntry]:
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
                - When __series__ is not defined.
            KeyError:
                - When circuit metadata doesn't provide required data processor options.
        """
        if self.__series__ is None:
            raise QiskitError("Curve __series__ is not provided for this analysis.")

        def _is_target_series(datum, **filters):
            try:
                return all(datum["metadata"][key] == val for key, val in filters.items())
            except KeyError:
                return False

        curve_data = list()
        for curve_properties in self.__series__:
            if curve_properties.filter_kwargs:
                # filter data
                series_data = [
                    datum
                    for datum in experiment_data.data()
                    if _is_target_series(datum, **curve_properties.filter_kwargs)
                ]
            else:
                # use data as-is
                series_data = experiment_data.data()

            # Format x, y, yerr data
            try:
                xvals = [datum["metadata"][self.__x_key__] for datum in series_data]
            except KeyError as ex:
                raise QiskitError(
                    f"X value key {self.__x_key__} is not defined in circuit metadata."
                ) from ex

            option_keys = curve_properties.data_option_keys or dict()

            def _data_processing(datum):
                # A helper function to receive data processor runtime option from metadata
                try:
                    # Extract data processor options
                    dp_options = {key: datum["metadata"][key] for key in option_keys}
                except KeyError as ex:
                    raise KeyError(
                        "Required data processor options are not provided by circuit metadata."
                    ) from ex
                return data_processor(datum, **dp_options)

            yvals, yerrs = zip(*map(_data_processing, series_data))

            # Apply data pre-processing
            prepared_xvals, prepared_yvals, prepared_yerrs = self._data_pre_processing(
                x_values=np.asarray(xvals, dtype=float),
                y_values=np.asarray(yvals, dtype=float),
                y_sigmas=np.asarray(yerrs, dtype=float),
            )

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
                    x_values=prepared_xvals,
                    y_values=prepared_yvals,
                    y_sigmas=prepared_yerrs,
                    metadata=dict(curve_metadata),
                )
            )

        return curve_data

    def _run_fitting(
        self,
        curve_data: List[CurveEntry],
        weights: Optional[np.ndarray] = None,
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
            curve_data: A list of curve data to fit.
            weights: Optional, a 1D float list of weights :math:`w_k` for each
                     component function :math:`f_k`.
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
            KeyError:
                - When fit function doesn't return Chi squared value.
        """
        num_curves = len(curve_data)

        # Setup fitting options
        fit_options = self._setup_fitting(curve_data, **options)

        # Validate weights
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
        separators = np.empty(num_curves)

        for idx, (datum, weight) in enumerate(zip(curve_data, sig_weights)):
            flat_xvals = np.concatenate((flat_xvals, datum.x_values))
            flat_yvals = np.concatenate((flat_yvals, datum.y_values))
            if datum.y_sigmas is not None:
                datum_yerrs = datum.y_sigmas / np.sqrt(weight)
            else:
                datum_yerrs = 1 / np.sqrt(weight)
            flat_yerrs = np.concatenate((flat_yerrs, datum_yerrs))
            separators[idx] = len(datum.x_values)
        separators = list(map(int, np.cumsum(separators)[:-1]))

        # Define multi-objective function
        def multi_objective_fit(x, *params):
            y = np.empty(0, dtype=float)
            xs = np.split(x, separators) if len(separators) > 0 else [x]
            for i, xi in enumerate(xs):
                yi = self._calculate_curve(curve_data[i].curve_name, xi, *params)
                y = np.concatenate((y, yi))
            return y

        # Try fit with each fit option
        fit_results = [
            self.__base_fitter__.__func__(
                f=multi_objective_fit,
                xdata=flat_xvals,
                ydata=flat_yvals,
                sigma=flat_yerrs,
                **fit_option,
            )
            for fit_option in fit_options
        ]

        # Sort by fit error
        try:
            fit_results = sorted(fit_results, key=lambda r: r["reduced_chisq"])
        except KeyError as ex:
            raise KeyError(
                "Returned analysis result does not provide reduced Chi squared value."
            ) from ex

        best_analysis_result = fit_results[0]
        best_analysis_result["popt_keys"] = self.__param_names__

        return best_analysis_result

    def _calculate_curve(self, curve_name: str, xvals: np.ndarray, *params: float) -> np.ndarray:
        """A helper method to manage parameter remapping for each series.

        This method calculate curve based on the fit function specified by the
        fit_func_index in each series definition.

        Args:
            curve_name: A name of curve. This should be defined in __series__ attribute.
            xvals: Array of x values.
            params: Full fit parameters specified in __param_names__.

        Returns:
            Fit y values.

        Raises:
            QiskitError:
                - When function parameter is not defined in the class parameter list.
                - When fit function index is out of range.
                - When curve information is not defined in class attribute __series__.
                - When series parameter is not defined in __param_names__.
        """
        named_params = dict(zip(self.__param_names__, params))

        for curve_properties in self.__series__:
            if curve_properties.name == curve_name:
                # remap parameters
                kw_params = {}
                for key, pname in curve_properties.p0_signature.items():
                    try:
                        kw_params[key] = named_params[pname]
                    except KeyError as ex:
                        raise QiskitError(
                            f"Series parameter {key} is not found in the fit parameter. "
                            f"{key} not in {', '.join(named_params.keys())}."
                        ) from ex

                # find fit function
                f_index = curve_properties.fit_func_index
                try:
                    return self.__fit_funcs__[f_index](xvals, **kw_params)
                except IndexError as ex:
                    raise QiskitError(f"Fit function of index {f_index} is not defined.") from ex

        raise QiskitError(f"A curve {curve_name} is not defined in this class.")

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
        """
        analysis_result = AnalysisResult()

        # Setup data processor
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

        # Extract curve entries from experiment data
        try:
            curve_data = self._extract_curves(data, data_processor)
        except DataProcessorError as ex:
            analysis_result["error_message"] = str(ex)
            analysis_result["success"] = False
            return [analysis_result], list()

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
        figures = self._create_figures(curve_data=curve_data, fit_data=analysis_result)

        # Store raw data
        raw_data = dict()
        for datum in curve_data:
            raw_data[datum.curve_name] = {
                "x_values": datum.x_values,
                "y_values": datum.y_values,
                "y_sigmas": datum.y_sigmas,
            }
        analysis_result["raw_data"] = raw_data

        return [analysis_result], figures
