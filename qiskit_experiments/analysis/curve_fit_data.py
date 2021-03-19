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
Curve fitter analysis class
"""

from typing import List, Tuple, Dict, Callable, Optional

import numpy as np
from scipy.optimize import curve_fit as scipy_curve_fit

from qiskit.exceptions import QiskitError
from qiskit_experiments.base_analysis import AnalysisResult

try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# pylint: disable = invalid-name
def curve_fit(
    data: List[Dict[str, any]],
    func: Callable,
    p0: Optional[List[float]] = None,
    p0_func: Optional[Callable] = None,
    bounds: Optional[Tuple] = None,
    fit_mean_data: bool = False,
    plot: bool = True,
    ax: Optional["AxesSubplot"] = None,
    **kwargs
):
    """Run curve fit analysis on circuit data.

    Args:
        data (ExperimentData): the experiment data to analyze.
        func: fit function `f(x, *params)`.
        p0: Optional, initial parameter values for curve_fit.
        p0_func: Optional, function for calculating initial p0.
        bounds: Optional, parameter bounds for curve_fit.
        fit_mean_data: Optional, if True pass means of data points to curve_fit.
        plot: If True generate a plot of fitted data.
        ax: Optional, matplotlib axis to add plot to.
        kwargs: additional kwargs to pass to curve_fit.

    Returns:
        tuple: A pair ``(analysis_results, figures)`` where
                ``analysis_results`` may be a single or list of
                AnalysisResult objects, and ``figures`` may be
                None, a single figure, or a list of figures.
    """
    # Compute curve fit data
    if fit_mean_data:
        xdata, ydata, sigma, xraw, yraw = average_curve_fit_data(
            data, return_raw=True)
    else:
        xdata, ydata, sigma = curve_fit_data(data)

    if p0_func is not None and p0 is None:
        p0 = p0_func(xdata, ydata, sigma=sigma)

    # Run curve fit
    popt, pcov = scipy_curve_fit(
        func, xdata, ydata, sigma=sigma, p0=p0, bounds=bounds, **kwargs)
    popt_err = np.sqrt(np.diag(pcov))

    # Compute chi-squared for fit
    yfits = func(xdata, *popt)
    chisq = np.mean(((yfits - ydata) / sigma) ** 2)

    result = AnalysisResult({
        "popt": popt,
        "popt_err": popt_err,
        "pcov": pcov,
        "chisq": chisq,
    })

    if plot and HAS_MATPLOTLIB:
        mean_data = (xdata, ydata, sigma) if fit_mean_data else None
        ax = curve_fit_plot(func, popt, popt_err, xraw,
                            ydata=yraw, mean_data=mean_data, ax=ax)
        # TODO: figure out what to do with plots
        return result, ax

    return result, None


def curve_fit_plot(
    func: Callable,
    popt: np.ndarray,
    popt_err: np.ndarray,
    xdata: np.ndarray,
    ydata: Optional[np.ndarray] = None,
    mean_data: Optional[Tuple[np.ndarray]] = None,
    num_fit_points: int = 100,
    ax: Optional["AxesSubplot"] = None,
):
    """Generate plot of raw and fitted data"""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            'curve_fit_plot requires matplotlib to generate curve fit plot.'
            ' Run "pip install matplotlib" before.')

    if ax is None:
        plt.figure()
        ax = plt.gca()

    # Plot fit data
    xs = np.linspace(np.min(xdata), np.max(xdata), num_fit_points)
    ys_fit = func(xs, *popt)
    ax.plot(xs, ys_fit, color="blue", linestyle="-", linewidth=2)

    # Plot standard error interval
    ys_upper = func(xs, *(popt + popt_err))
    ys_lower = func(xs, *(popt - popt_err))
    ax.fill_between(xs, ys_lower, ys_upper, color="blue", alpha=0.1)

    # Plot raw data
    if ydata is not None:
        ax.scatter(xdata, ydata, c="grey", marker="x")

    # Error bar plot of mean data
    if mean_data is not None:
        ax.errorbar(mean_data[0], mean_data[1], mean_data[2],
                    marker='.', markersize=9, linestyle='--', color='red')

    # Formatting
    ax.tick_params(labelsize=14)
    ax.grid(True)
    return ax


def curve_fit_data(data: List[Dict[str, any]]) -> Tuple[np.ndarray]:
    """Return array of (x, y, sigma) data for curve fitting.

    Args
        data: list of circuit data dictionaries.

    Returns:
        tuple: ``(x, y, sigma)`` tuple of arrays of x-values,
               y-values, and standard deviations of y-values.

    Raises:
        QiskitError: if input data is not level-2 measurement.

    .. note::

        This requires metadata to contain `"xdata"`, `"ylabel"`
        keys containing the numeric x-value for fitting, and the outcome
        bitstring for computing y-value probability from counts.

    .. note::

        Currently only level-2 (count) measurement data is supported.
    """
    size = len(data)
    xdata = np.zeros(size, dtype=int)
    ydata = np.zeros(size, dtype=float)
    ydata_var = np.zeros(size, dtype=float)

    for i, datum in enumerate(data):
        metadata = datum["metadata"]
        meas_level = metadata.get("meas_level", 2)
        xdata[i] = metadata["xdata"]
        if meas_level == 2:
            y_mean, y_var = level2_probability(
                datum["counts"], metadata["ylabel"])
            ydata[i] = y_mean
            ydata_var[i] = y_var
        else:
            # Adding support for level-1 measurment data is still todo.
            raise QiskitError("Measurement level 1 is not supported.")

    return xdata, ydata, np.sqrt(ydata_var)


def average_curve_fit_data(data: List[Dict[str, any]],
                           return_raw: bool = False) -> Tuple[np.ndarray]:
    """Return array of (x, y_mean, sigma) data for curve fitting.

    Compared to :func:`curve_fit_data` this computes the sample mean and
    standard deviation of each set of y-data with the same x value.

    Args
        data: list of circuit data dictionaries.

    Returns:
        tuple: ``(x, y_mean, sigma)`` if ``return_raw==False``, where
               ``x`` is an arrays of unique x-values, ``y`` is an array of
               sample mean y-values, and ``sigma`` is an array of sample standard
               deviation of y values.
        tuple: ``(x, y_mean, sigma, x_raw, y_raw) where ``x_raw, y_raw`` are the
                full set of x and y data arrays before averaging.

    .. note::

        This requires metadata to contain `"xdata"`, `"ylabel"`
        keys containing the numeric x-value for fitting, and the outcome
        bitstring for computing y-value probability from counts.

    .. note::

        Currently only level-2 (count) measurement data is supported.
    """
    x_raw, y_raw, _ = curve_fit_data(data)

    # Note this assumes discrete X-data
    x_means = np.unique(x_raw)
    y_means = np.zeros(x_means.size)
    y_sigmas = np.zeros(x_means.size)
    for i in range(x_means.size):
        ys = y_raw[x_raw == x_means[i]]
        num_samples = len(ys)
        sample_mean =  np.mean(ys)
        sample_var = np.sum((sample_mean - ys) ** 2) / (num_samples - 1)
        y_means[i] = sample_mean
        y_sigmas[i] = np.sqrt(sample_var)
    if return_raw:
        return x_means, y_means, y_sigmas, x_raw, y_raw
    return x_means, y_means, y_sigmas


def level2_probability(counts: "Counts", outcome: str) -> Tuple[float]:
    """Return the outcome probability mean and variance.

    Args:
        counts: A counts object.
        outcome: bitstring for desired outcome probability.

    Returns:
        tuple: (p_mean, p_var) of the probability mean and variance
               estimated from the counts.

    .. note::

        This assumes a binomial distribution where :math:`K` counts
        of the desired outcome from :math:`N` shots the
        mean probability is :math:`p = K / N` and the variance is
        :math:`\\sigma^2 = N p (1-p)`.
    """
    shots = sum(counts.values())
    p_mean = counts.get(outcome, 0.0) / shots
    p_var = shots * p_mean * (1 - p_mean)
    return p_mean, p_var
