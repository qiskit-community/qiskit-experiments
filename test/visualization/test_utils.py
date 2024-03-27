# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Test visualization utilities.
"""

import itertools as it
from test.base import QiskitExperimentsTestCase
from typing import List, Tuple

import numpy as np
from ddt import data, ddt
from qiskit.exceptions import QiskitError

from qiskit_experiments.visualization.utils import DataExtentCalculator
from qiskit_experiments.framework.package_deps import version_is_at_least


@ddt
class TestDataExtentCalculator(QiskitExperimentsTestCase):
    """Test DataExtentCalculator"""

    @classmethod
    def _dummy_data(
        cls,
        extent: Tuple[float, float, float, float] = (-1, 1, -5, 0),
        n_data: int = 5,
        n_points: int = 16,
    ) -> List[np.ndarray]:
        # Create a list of bin edges by which to divide the target extent
        bin_edges = [
            np.histogram_bin_edges(extent[0:2], bins=n_data).tolist(),
            np.histogram_bin_edges(extent[2:], bins=n_data).tolist(),
        ]

        # Iterate over pairs of adjacent bin edges, which define the maximum and minimum for the region.
        # This is done by generating sliding windows of bin_edges as follows:
        #      [[a], [b], [c], [d], [e], [f]], g]
        #  [a, [[b], [c], [d], [e], [f], [g]]
        # The result is a list of pairs representing a moving window of size 2.
        # TODO: remove the old code once numpy is above 1.20.
        dummy_data = []
        if version_is_at_least("numpy", "1.20"):
            for (x_min, x_max), (y_min, y_max) in it.product(
                *np.lib.stride_tricks.sliding_window_view(bin_edges, 2, 1)
            ):
                _dummy_data = np.asarray(
                    [
                        np.linspace(x_min, x_max, n_points),
                        np.linspace(y_min, y_max, n_points),
                    ]
                )
                dummy_data.append(_dummy_data.swapaxes(-1, -2))
        else:
            for (x_min, x_max), (y_min, y_max) in it.product(
                *tuple(list(zip(b[0:-1], b[1:])) for b in bin_edges)
            ):
                _dummy_data = np.asarray(
                    [
                        np.linspace(x_min, x_max, n_points),
                        np.linspace(y_min, y_max, n_points),
                    ]
                )
                dummy_data.append(_dummy_data.swapaxes(-1, -2))
        return dummy_data

    @data(*list(it.product([1.0, 1.1, 2.0], [None, 1.0, np.sqrt(2)])))
    def test_end_to_end(self, args):
        """Test end-to-end functionality.

        Results that are asserted include the range of the final extent tuple and its midpoint.
        """
        # Test args
        multiplier, aspect_ratio = args[0], args[1]

        # Problem inputs
        extent = (-1, 1, -5, 1)

        n_data = 6
        dummy_data = self._dummy_data(extent, n_data=n_data)
        ext_calc = DataExtentCalculator(multiplier=multiplier, aspect_ratio=aspect_ratio)
        # Add data as 2D and 1D arrays to test both methods
        for d in dummy_data[0 : int(n_data / 2)]:
            ext_calc.register_data(d)
        for d in dummy_data[int(n_data / 2) :]:
            for i_dim in range(2):
                ext_calc.register_data(d[:, i_dim], dim=i_dim)

        # Check extent
        actual_extent = ext_calc.extent()

        # Check that range was scaled. Given we also have an aspect ratio, we may have a range that is
        # larger than the original scaled by the multiplier. At the minimum, the range should be exactly
        # equal to the original scaled by the multiplier
        expected_range = multiplier * np.diff(np.asarray(extent).reshape((2, 2)), axis=1).flatten()
        actual_range = np.diff(np.reshape(actual_extent, (2, 2)), axis=1).flatten()
        for act, exp in zip(actual_range, expected_range):
            self.assertTrue(act >= exp)

        # Check that the midpoints are the same.
        expected_midpoint = np.mean(np.reshape(extent, (2, 2)), axis=1).flatten()
        actual_midpoint = np.mean(np.reshape(actual_extent, (2, 2)), axis=1).flatten()
        np.testing.assert_almost_equal(
            actual_midpoint,
            expected_midpoint,
        )

    def test_no_data_error(self):
        """Test that a QiskitError is raised if no data was set."""
        ext_calc = DataExtentCalculator()
        with self.assertRaises(QiskitError):
            ext_calc.extent()
