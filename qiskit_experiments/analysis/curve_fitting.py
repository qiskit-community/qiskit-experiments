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

from typing import List, Callable, Optional

import numpy as np
import scipy.optimize as opt

from qiskit.exceptions import QiskitError
from qiskit_experiments.base_analysis import AnalysisResult


def curve_fit(
    func: Callable,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    **kwargs,
) -> AnalysisResult:
    """Use non-linear least squares to fit a function, f, to data

    Wraps scipy.optimize.curve_fit

    Args:
        func: a fit function `f(x *params)`.
        xdata: a 1D array of x-data
        ydata: a 1D array of y-data
        p0: initial guess for optimization parameters.
        sigma: Optional, a 1D array of standard deviations in ydata.
        kwargs: additional kwargs for scipy.optimize.curve_fit.

    Returns:
        AnalysisResult: result containing `popt` the optimal fit parameters,
                        `popt_err` the standard error estimates popt,
                        `pcov` the covariance matrix for the fit,
                        `chisq` the chi-squared parameter of fit,
                        `xrange` the range of xdata values used for fit.
    """

    # Run curve fit
    # pylint: disable unbackend-tuple-unpacking
    popt, pcov = opt.curve_fit(func, xdata, ydata, sigma=sigma, p0=p0, **kwargs)
    popt_err = np.sqrt(np.diag(pcov))

    # Compute chi-squared for fit
    yfits = func(xdata, *popt)
    chisq = np.mean(((yfits - ydata) / sigma) ** 2)

    # Compute xdata range for fit
    xdata_range = [min(xdata), max(xdata)]

    result = {
        "popt": popt,
        "popt_err": popt_err,
        "pcov": pcov,
        "chisq": chisq,
        "xrange": xdata_range,
    }
    # TODO:
    #  1. Add some basic validation of computer good / bad based on fit result.
    #  2. Add error handling so if fitting fails we can return an analysis
    #     result containing this information
    return AnalysisResult(result)


def multi_curve_fit(
    funcs: List[Callable],
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    **kwargs,
):
    """Use non-linear least squares to fit a list of functions, f_i, to data

    Args:
        funcs: a list of objective functions with signatures `f_i(x, *params)`.
        xdata: a 2D array of xdata and function function indexes.
        ydata: a 1D array of ydata
        p0: initial guess for optimization parameters.
        sigma: Optional, a 1D array of standard deviations in ydata.
        weights: Optional, a 1D list of numeric weights for each function.
        kwargs: additional kwargs for scipy.optimize.curve_fit.

    Returns:
        AnalysisResult: result containing `popt` the optimal fit parameters,
                        `popt_err` the standard error estimates popt,
                        `pcov` the covariance matrix for the fit,
                        `chisq` the chi-squared parameter of fit.
                        `xrange` the range of xdata values used for fit.

    Raises:
        QiskitError: if input xdata is not 2D.
    """
    num_funcs = len(funcs)

    # Get 1D xdata and indices from 2D input xdata
    xdata = np.asarray(xdata, dtype=float)
    if xdata.ndim != 2:
        raise QiskitError("multi_curve_fit requires 2D xdata.")
    xdata1d, xindex = xdata.T

    # Get positions for indexes data sets
    idxs = [xindex == i for i in range(num_funcs)]

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
    def f(x, *params):
        y = np.zeros(x.size)
        for i in range(num_funcs):
            xi = x[idxs[i]]
            yi = funcs[i](xi, *params)
            y[idxs[i]] = yi
        return y

    # Run linearized curve_fit
    analysis_result = curve_fit(f, xdata1d, ydata, p0, sigma=wsigma, **kwargs)

    return analysis_result
