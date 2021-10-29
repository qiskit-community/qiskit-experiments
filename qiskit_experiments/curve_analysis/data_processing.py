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

from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from qiskit.exceptions import QiskitError


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
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    shots: Optional[np.ndarray] = None,
    method: str = "sample",
) -> Tuple[np.ndarray, ...]:
    r"""Return (x, y_mean, sigma) data.

    The mean is taken over all ydata values with the same xdata value using
    the specified method. For each x the mean :math:`\overline{y}` and variance
    :math:`\sigma^2` are computed as

    * ``"sample"`` (default) *Sample mean and variance*
      :math:`\overline{y} = \sum_{i=1}^N y_i / N`,
      :math:`\sigma^2 = \sum_{i=1}^N ((\overline{y} - y_i)^2) / N`
    * ``"iwv"`` *Inverse-weighted variance*
      :math:`\overline{y} = (\sum_{i=1}^N y_i / \sigma_i^2 ) \sigma^2`
      :math:`\sigma^2 = 1 / (\sum_{i=1}^N 1 / \sigma_i^2)`
    * ``"shots_weighted_variance"`` *Sample mean and variance with weights from shots*
      :math:`\overline{y} = \sum_{i=1}^N n_i y_i / M`,
      :math:`\sigma^2 = \sum_{i=1}^N (n_i \sigma_i / M)^2`,
      where :math:`n_i` is the number of shots per data point and :math:`M = \sum_{i=1}^N n_i`
      is a total number of shots from different circuit execution at the same x value.
      If ``shots`` is not provided, this applies uniform weights to all values.

    Args
        xdata: 1D or 2D array of xdata from curve_fit_data or
               multi_curve_fit_data
        ydata: array of ydata returned from curve_fit_data or
               multi_curve_fit_data
        sigma: Optional, array of standard deviations in ydata.
        shots: Optional, array of shots used to get a data point.
        method: The method to use for computing y means and
                standard deviations sigma (default: "sample").

    Returns:
        tuple: ``(x, y_mean, sigma, shots)``, where
               ``x`` is an arrays of unique x-values, ``y`` is an array of
               sample mean y-values, ``sigma`` is an array of sample standard
               deviation of y values, and ``shots`` are the total number of experiment shots
               used to evaluate the data point. If ``shots`` in the function call is ``None``,
               the numbers appear in the returned value will represent just a number of
               duplicated x value entries.

    Raises:
        QiskitError: if "ivw" method is used without providing a sigma.
    """
    x_means = np.unique(xdata, axis=0)
    y_means = np.zeros(x_means.size)
    y_sigmas = np.zeros(x_means.size)
    y_shots = np.zeros(x_means.size)

    if shots is None or any(np.isnan(shots)):
        # this will become standard average
        shots = np.ones_like(xdata)

    # Sample mean and variance method
    if method == "sample":
        for i in range(x_means.size):
            # Get positions of y to average
            idxs = xdata == x_means[i]
            ys = ydata[idxs]
            ns = shots[idxs]

            # Compute sample mean and biased sample variance
            y_means[i] = np.mean(ys)
            y_sigmas[i] = np.sqrt(np.mean((y_means[i] - ys) ** 2) / ys.size)
            y_shots[i] = np.sum(ns)

        return x_means, y_means, y_sigmas, y_shots

    # Inverse-weighted variance method
    if method == "iwv":
        if sigma is None:
            raise QiskitError(
                "The inverse-weighted variance method cannot be used with" " `sigma=None`"
            )
        for i in range(x_means.size):
            # Get positions of y to average
            idxs = xdata == x_means[i]
            ys = ydata[idxs]
            ns = shots[idxs]

            # Compute the inverse-variance weighted y mean and variance
            weights = 1 / sigma[idxs] ** 2
            y_var = 1 / np.sum(weights)
            y_means[i] = y_var * np.sum(weights * ys)
            y_sigmas[i] = np.sqrt(y_var)
            y_shots[i] = np.sum(ns)

        return x_means, y_means, y_sigmas, y_shots

    # Quadrature sum of variance
    if method == "shots_weighted":
        for i in range(x_means.size):
            # Get positions of y to average
            idxs = xdata == x_means[i]
            ys = ydata[idxs]
            ss = sigma[idxs]
            ns = shots[idxs]
            weights = ns / np.sum(ns)

            # Compute sample mean and sum of variance with weights based on shots
            y_means[i] = np.sum(weights * ys)
            y_sigmas[i] = np.sqrt(np.sum(weights ** 2 * ss ** 2))
            y_shots[i] = np.sum(ns)

        return x_means, y_means, y_sigmas, y_shots

    # Invalid method
    raise QiskitError(f"Unsupported method {method}")


def multi_mean_xy_data(
    series: np.ndarray,
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    shots: Optional[np.ndarray] = None,
    method: str = "sample",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Take mean of multi series data set.

    Args:
        series: Series index.
        xdata: 1D or 2D array of xdata from curve_fit_data or
               multi_curve_fit_data
        ydata: array of ydata returned from curve_fit_data or
               multi_curve_fit_data
        sigma: Optional, array of standard deviations in ydata.
        shots: Optional, array of shots used to get a data point.
        method: The method to use for computing y means and
                standard deviations sigma (default: "sample").

    Returns:
        Tuple of (series, xdata, ydata, sigma, shots)

    See also:
        :py:func:`~qiskit_experiments.curve_analysis.data_processing.mean_xy_data`
    """
    series_vals = np.unique(series)

    series_means = []
    xdata_means = []
    ydata_means = []
    sigma_means = []
    shots_sums = []

    # Get x, y, sigma data for series and process mean data
    for series_val in series_vals:
        idxs = series == series_val
        sigma_i = sigma[idxs] if sigma is not None else None
        shots_i = shots[idxs] if shots is not None else None

        x_mean, y_mean, sigma_mean, shots_sum = mean_xy_data(
            xdata[idxs], ydata[idxs], sigma=sigma_i, shots=shots_i, method=method
        )
        series_means.append(np.full(x_mean.size, series_val, dtype=int))
        xdata_means.append(x_mean)
        ydata_means.append(y_mean)
        sigma_means.append(sigma_mean)
        shots_sums.append(shots_sum)

    # Concatenate lists
    return (
        np.concatenate(series_means),
        np.concatenate(xdata_means),
        np.concatenate(ydata_means),
        np.concatenate(sigma_means),
        np.concatenate(shots_sums),
    )


def data_sort(
    series: np.ndarray,
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    shots: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort data.

    Input x values may not be lined up in order, since experiment may accept user input array,
    or data may be concatenated with previous scan. This sometimes confuses the algorithmic
    generation of initial guesses especially when guess depends on derivative.

    This returns data set that is sorted by xdata and series in ascending order.

    Args:
        series: Series index.
        xdata: 1D or 2D array of xdata from curve_fit_data or
               multi_curve_fit_data
        ydata: array of ydata returned from curve_fit_data or
               multi_curve_fit_data
        sigma: Optional, array of standard deviations in ydata.
        shots: Optional, array of shots used to get a data point.

    Returns:
        Tuple of (series, xdata, ydata, sigma, shots) sorted in ascending order of xdata and series.
    """
    if sigma is None:
        sigma = np.full(series.size, np.nan, dtype=float)

    if shots is None:
        shots = np.full(series.size, np.nan, dtype=float)

    sorted_data = sorted(zip(series, xdata, ydata, sigma, shots), key=lambda d: (d[0], d[1]))

    return np.asarray(sorted_data).T


def level2_probability(data: Dict[str, any], outcome: str) -> Tuple[float, float]:
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
        :math:`\\sigma^2 = p (1-p) / N`.
    """
    counts = data["counts"]

    shots = sum(counts.values())
    p_mean = counts.get(outcome, 0.0) / shots
    p_var = p_mean * (1 - p_mean) / shots
    return p_mean, p_var


def probability(outcome: str) -> Callable:
    """Return probability data processor callback used by the analysis classes."""

    def data_processor(data):
        return level2_probability(data, outcome)

    return data_processor
