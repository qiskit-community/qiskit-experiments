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

"""Light-weight class of analysis ready data."""

from typing import Iterator
import numpy as np


class ProcessedData:
    """ProcessedData is a light-weight class of data ready for analysis."""

    def __init__(self):
        """Setup the empty data container."""
        self._data = {}

    def add_data_point(self, xval: float, yval: float, series: str = None):
        """
        Args:
            xval: The value of the independent variable.
            yval: The value of the dependent variable.
            series: The series to which this data point belongs.
        """

        if series is None:
            series = 'default'

        if series not in self._data:
            self._data[series] = {'xvals': [], 'yvals': []}

        self._data[series]['xvals'].append(xval)
        self._data[series]['yvals'].append(yval)

    def series(self) -> Iterator:
        """
        Returns:
            iterator: where the return values are tuples of (xvals, yvals, series)

        Yields:
            The tuple (xvals, yvals, series_name) contained in self.
        """
        for series_name, data in self._data.items():
            yield np.array(data['xvals']), np.array(data['yvals']), series_name
