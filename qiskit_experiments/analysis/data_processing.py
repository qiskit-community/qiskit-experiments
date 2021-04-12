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
Data processing utility functions for curve fitting experiments
"""
# pylint: disable = invalid-name

from typing import List, Dict, Tuple, Optional
import numpy as np


def filter_data(data: List[Dict[str, any]], **filters) -> List[Dict[str, any]]:
    """Return the list of filtered data

    Args:
        data: list of data dicts.
        filters: kwargs for filtering based on metadata
                 values.

    Returns:
        The list of filtered data. If no filters are provided this will be the
        input list.
    """
    if not filters:
        return data
    filtered_data = []
    for datum in data:
        include = True
        metadata = datum["metadata"]
        for key, val in filters.items():
            if key not in metadata or metadata[key] != val:
                include = False
                break
        if include:
            filtered_data.append(datum)
    return filtered_data


def mean_xy_data(
    xdata: np.ndarray, ydata: np.ndarray, sigma: Optional[np.ndarray] = None
) -> Tuple[np.ndarray]:
    """Return (x, y_mean, sigma) data.

    The mean is taken over all ydata values with the same xdata value.
    If `sigma=None` the sample mean and biased sample variance is used,
    otherwise the inverse-variance weighted mean and variance is used.

    Args
        xdata: 1D or 2D array of xdata from curve_fit_data or
               multi_curve_fit_data
        ydata: array of ydata returned from curve_fit_data or
               multi_curve_fit_data
        sigma: Optional, array of standard deviations in ydata.

    Returns:
        tuple: ``(x, y_mean, sigma)`` if ``return_raw==False``, where
               ``x`` is an arrays of unique x-values, ``y`` is an array of
               sample mean y-values, and ``sigma`` is an array of sample standard
               deviation of y values.
    """
    x_means = np.unique(xdata, axis=0)
    y_means = np.zeros(x_means.size)
    y_sigmas = np.zeros(x_means.size)
    if sigma is None:
        sigma = np.ones(xdata.size)
    for i in range(x_means.size):
        # Get positions of y to average
        idxs = xdata == x_means[i]
        ys = ydata[idxs]

        if sigma is not None:
            # Compute the inverse-variance weighted y mean and variance
            weights = 1 / sigma[idxs] ** 2
            y_var = 1 / np.sum(weights)
            y_mean = y_var * np.sum(weights * ys)
        else:
            # Compute biased sample mean and variance
            y_mean = np.mean(ys)
            y_var = np.sum((y_mean - ys) ** 2) / len(ys)

        y_means[i] = y_mean
        y_sigmas[i] = np.sqrt(y_var)
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
