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
Curve fitting functions for experiment analysis
"""
# pylint: disable = invalid-name

from typing import List, Dict, Tuple, Callable, Optional, Union

import numpy as np
import scipy.optimize as opt
from qiskit.exceptions import QiskitError
from qiskit_experiments.base_analysis import AnalysisResult
from qiskit_experiments.analysis.data_processing import filter_data


def curve_fit(
    func: Callable,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: Union[Dict[str, float], np.ndarray],
    sigma: Optional[np.ndarray] = None,
    bounds: Optional[Union[Dict[str, Tuple[float, float]], Tuple[np.ndarray, np.ndarray]]] = None,
    **kwargs,
) -> AnalysisResult:
    r"""Perform a non-linear least squares to fit

    This solves the optimization problem

    .. math::
        \Theta_{\mbox{opt}} = \arg\min_\Theta \sum_i
            \sigma_i^{-2} (f(x_i, \Theta) -  y_i)^2

    using :func:`scipy.optimize.curve_fit`.

    Args:
        func: a fit function `f(x, *params)`.
        xdata: a 1D float array of x-data.
        ydata: a 1D float array of y-data.
        p0: initial guess for optimization parameters.
        sigma: Optional, a 1D array of standard deviations in ydata
               in absolute units.
        bounds: Optional, lower and upper bounds for optimization
                parameters.
        kwargs: additional kwargs for :func:`scipy.optimize.curve_fit`.

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
        the `absolute_sigma=False` kwarg of scipy
        :func:`~scipy.optimize.curve_fit` must be used. This affects the
        returned covariance ``pcov`` and error ``popt_err`` parameters via
        ``pcov(absolute_sigma=False) = pcov * reduced_chisq``
        ``popt_err(absolute_sigma=False) = popt_err * sqrt(reduced_chisq)``.
    """
    # Format p0 parameters if specified as dictionary
    if isinstance(p0, dict):
        param_keys = list(p0.keys())
        param_p0 = list(p0.values())

        # Convert bounds
        if bounds:
            lower = [bounds[key][0] for key in param_keys]
            upper = [bounds[key][1] for key in param_keys]
            param_bounds = (lower, upper)
        else:
            param_bounds = None

        # Convert fit function
        def fit_func(x, *params):
            return func(x, **dict(zip(param_keys, params)))

    else:
        param_keys = None
        param_p0 = p0
        param_bounds = bounds
        fit_func = func

    # Check the degrees of freedom is greater than 0
    dof = len(ydata) - len(param_p0)
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
    # TODO: Add error handling so if fitting fails we can return an analysis
    #       result containing this information
    # pylint: disable = unbalanced-tuple-unpacking
    popt, pcov = opt.curve_fit(
        fit_func, xdata, ydata, sigma=sigma, p0=param_p0, bounds=param_bounds, **kwargs
    )
    popt_err = np.sqrt(np.diag(pcov))

    # Calculate the reduced chi-squared for fit
    yfits = fit_func(xdata, *popt)
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


def multi_curve_fit(
    funcs: List[Callable],
    series: np.ndarray,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    bounds: Optional[Union[Dict[str, Tuple[float, float]], Tuple[np.ndarray, np.ndarray]]] = None,
    **kwargs,
) -> AnalysisResult:
    r"""Perform a linearized multi-objective non-linear least squares fit.

    This solves the optimization problem

    .. math::
        \Theta_{\mbox{opt}} = \arg\min_\Theta \sum_{k} w_k
            \sum_{i} \sigma_{k, i}^{-2}
            (f_k(x_{k, i}, \Theta) -  y_{k, i})^2

    for multiple series of :math:`x_k, y_k, \sigma_k` data evaluated using
    a list of objective functions :math:`[f_k]`
    using :func:`scipy.optimize.curve_fit`.

    Args:
        funcs: a list of objective functions :math:`[f_0, f_1, ...]` where
               each function has signature :math`f_k(x, \Theta)`.
        series: a 1D int array that specifies the component objective
                function :math:`f_k` to evaluate corresponding x and y
                data with.
        xdata: a 1D float array of xdata.
        ydata: a 1D float array of ydata.
        p0: initial guess for optimization parameters.
        sigma: Optional, a 1D array of standard deviations in ydata
               in absolute units.
        weights: Optional, a 1D float list of weights :math:`w_k` for each
                 component function :math:`f_k`.
        bounds: Optional, lower and upper bounds for optimization
                parameters.
        kwargs: additional kwargs for :func:`scipy.optimize.curve_fit`.

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
        the `absolute_sigma=False` kwarg of scipy
        :func:`~scipy.optimize.curve_fit` must be used. This affects the
        returned covariance ``pcov`` and error ``popt_err`` parameters via
        ``pcov(absolute_sigma=False) = pcov * reduced_chisq``
        ``popt_err(absolute_sigma=False) = popt_err * sqrt(reduced_chisq)``.
    """
    num_funcs = len(funcs)

    # Get positions for indexes data sets
    series = np.asarray(series, dtype=int)
    idxs = [series == i for i in range(num_funcs)]

    # Combine weights and sigma for transformation
    if weights is None:
        wsigma = sigma
    else:
        wsigma = np.zeros(ydata.size)
        if sigma is None:
            for i in range(num_funcs):
                wsigma[idxs[i]] = 1 / np.sqrt(weights[i])
        else:
            for i in range(num_funcs):
                wsigma[idxs[i]] = sigma[idxs[i]] / np.sqrt(weights[i])

    # Define multi-objective function
    def f(x, *args, **kwargs):
        y = np.zeros(x.size)
        for i in range(num_funcs):
            xi = x[idxs[i]]
            yi = funcs[i](xi, *args, **kwargs)
            y[idxs[i]] = yi
        return y

    # Run linearized curve_fit
    analysis_result = curve_fit(f, xdata, ydata, p0, sigma=wsigma, bounds=bounds, **kwargs)

    return analysis_result


def process_curve_data(
    data: List[Dict[str, any]], data_processor: Callable, x_key: str = "xval", **filters
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tuple of arrays (x, y, sigma) data for curve fitting.

    Args
        data: list of circuit data dictionaries containing counts.
        data_processor: callable for processing data to y, sigma
        x_key: key for extracting xdata value from metadata (Default: "xval").
        filters: additional kwargs to filter metadata on.

    Returns:
        tuple: ``(x, y, sigma)`` tuple of arrays of x-values,
               y-values, and standard deviations of y-values.
    """
    filtered_data = filter_data(data, **filters)
    size = len(filtered_data)
    xdata = np.zeros(size, dtype=float)
    ydata = np.zeros(size, dtype=float)
    ydata_var = np.zeros(size, dtype=float)

    for i, datum in enumerate(filtered_data):
        metadata = datum["metadata"]
        xdata[i] = metadata[x_key]
        y_mean, y_var = data_processor(datum)
        ydata[i] = y_mean
        ydata_var[i] = y_var

    return xdata, ydata, np.sqrt(ydata_var)


def process_multi_curve_data(
    data: List[Dict[str, any]],
    data_processor: Callable,
    x_key: str = "xval",
    series_key: str = "series",
    **filters,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return tuple of arrays (series, x, y, sigma) data for multi curve fitting.

    Args
        data: list of circuit data dictionaries.
        data_processor: callable for processing data to y, sigma
        x_key: key for extracting xdata value from metadata (Default: "xval").
        series_key: key for extracting series value from metadata (Default: "series").
        filters: additional kwargs to filter metadata on.

    Returns:
        tuple: ``(series, x, y, sigma)`` tuple of arrays of series values,
               x-values, y-values, and standard deviations of y-values.
    """
    filtered_data = filter_data(data, **filters)
    size = len(filtered_data)
    series = np.zeros(size, dtype=int)
    xdata = np.zeros(size, dtype=float)
    ydata = np.zeros(size, dtype=float)
    ydata_var = np.zeros(size, dtype=float)

    for i, datum in enumerate(filtered_data):
        metadata = datum["metadata"]
        series[i] = metadata[series_key]
        xdata[i] = metadata[x_key]
        y_mean, y_var = data_processor(datum)
        ydata[i] = y_mean
        ydata_var[i] = y_var

    return series, xdata, ydata, np.sqrt(ydata_var)
