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

from typing import List, Dict, Tuple, Callable
import numpy as np
from qiskit_experiments.curve_analysis.utils import filter_data


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
