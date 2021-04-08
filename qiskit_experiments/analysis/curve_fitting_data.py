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
Curve fitting xydata for experiment analysis
"""
# pylint: disable = invalid-name

from typing import List, Dict, Tuple, Callable
import numpy as np


def curve_fit_data(
    data: List[Dict[str, any]], data_processor: Callable, x_key: str = "xval"
) -> Tuple[np.ndarray]:
    """Return array of (x, y, sigma) data for curve fitting.

    Args
        data: list of circuit data dictionaries containing counts.
        data_processor: callable for processing data to y, sigma
        x_key: key for extracting xdata value from metadata (Default: "xval").

    Returns:
        tuple: ``(x, y, sigma)`` tuple of arrays of x-values,
               y-values, and standard deviations of y-values.

    Raises:
        QiskitError: if input data is not level-2 measurement.
    """
    size = len(data)
    xdata = np.zeros(size, dtype=int)
    ydata = np.zeros(size, dtype=float)
    ydata_var = np.zeros(size, dtype=float)

    for i, datum in enumerate(data):
        metadata = datum["metadata"]
        xdata[i] = metadata[x_key]
        y_mean, y_var = data_processor(datum)
        ydata[i] = y_mean
        ydata_var[i] = y_var

    return xdata, ydata, np.sqrt(ydata_var)


def multi_curve_fit_data(
    data: List[Dict[str, any]],
    data_processor: Callable,
    x_key: str = "xval",
    series_key: str = "series",
) -> Tuple[np.ndarray]:
    """Return array of (x, y, sigma) data for curve fitting.

    Args
        data: list of circuit data dictionaries.
        data_processor: callable for processing data to y, sigma
        x_key: key for extracting xdata value from metadata (Default: "xval").
        series_key: key for extracting series value from metadata (Default: "series").

    Returns:
        tuple: ``(x, y, sigma)`` tuple of arrays of x-values,
               y-values, and standard deviations of y-values.

    Raises:
        QiskitError: if input data is not level-2 measurement.
    """
    size = len(data)
    xdata = np.zeros((size, 2), dtype=float)
    ydata = np.zeros(size, dtype=float)
    ydata_var = np.zeros(size, dtype=float)

    for i, datum in enumerate(data):
        metadata = datum["metadata"]
        xdata[i, 0] = metadata[x_key]
        xdata[i, 1] = metadata[series_key]
        y_mean, y_var = data_processor(datum)
        ydata[i] = y_mean
        ydata_var[i] = y_var

    return xdata, ydata, np.sqrt(ydata_var)


def mean_xy_data(xdata: np.ndarray, ydata: np.ndarray) -> Tuple[np.ndarray]:
    """Return (x, y_mean, sigma) data.

    The mean is taken over all ydata values with the same xdata value.

    Args
        xdata: 1D or 2D array of xdata from curve_fit_data or
               multi_curve_fit_data
        ydata: array of ydata returned from curve_fit_data or
               multi_curve_fit_data

    Returns:
        tuple: ``(x, y_mean, sigma)`` if ``return_raw==False``, where
               ``x`` is an arrays of unique x-values, ``y`` is an array of
               sample mean y-values, and ``sigma`` is an array of sample standard
               deviation of y values.
    """
    x_means = np.unique(xdata, axis=0)
    y_means = np.zeros(x_means.size)
    y_sigmas = np.zeros(x_means.size)
    for i in range(x_means.size):
        ys = ydata[xdata == x_means[i]]
        num_samples = len(ys)
        sample_mean = np.mean(ys)
        sample_var = np.sum((sample_mean - ys) ** 2) / (num_samples - 1)
        y_means[i] = sample_mean
        y_sigmas[i] = np.sqrt(sample_var)
    return x_means, y_means, y_sigmas


def level2_probability(data: Dict[str, any], outcome: str) -> Tuple[float]:
    """Return the outcome probability mean and variance.

    Args:
        data: A data dict containing count data.
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
    counts = data["counts"]
    shots = sum(counts.values())
    p_mean = counts.get(outcome, 0.0) / shots
    p_var = shots * p_mean * (1 - p_mean)
    return p_mean, p_var
