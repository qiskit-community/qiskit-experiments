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
Functions for fitting parameters.
"""

import numpy as np
from scipy.optimize import curve_fit


def exp_fit_fun(x, a, tau, c):
    """
    Exponential fit function
    """
    return a * np.exp(-x / tau) + c


# pylint: disable = invalid-name
def curve_fit_wrapper(f, xdata, ydata, sigma, **kwargs):
    """
    A wrapper to curve_fit that calculates and returns fit_err
    (square root of the diagonal of the covariance matrix) and chi square.

    Args:
        f (callable): see documentation of curve_fit in scipy.optimize
        xdata (list): see documentation of curve_fit in scipy.optimize
        ydata (list): see documentation of curve_fit in scipy.optimize
        sigma (list): see documentation of curve_fit in scipy.optimize
        kwargs: additional paramters to be passed to curve_fit

    Returns:
        list: fitted parameters
        list: error on fitted parameters
              (square root of the diagonal of the covariance matrix)
        matrix: the covariance matrix
        float: chi-square
    """
    fit_out, fit_cov = curve_fit(f, xdata, ydata, sigma=sigma, **kwargs)

    chisq = 0
    for x, y, sig in zip(xdata, ydata, sigma):
        chisq += (f(x, *fit_out) - y) ** 2 / sig ** 2
    chisq /= len(xdata)

    fit_err = np.sqrt(np.diag(fit_cov))

    return fit_out, fit_err, fit_cov, chisq
