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
Test integration of plotter.
"""

from copy import deepcopy
from test.base import QiskitExperimentsTestCase

from .mock_drawer import MockDrawer
from .mock_plotter import MockPlotter


class TestPlotter(QiskitExperimentsTestCase):
    """Test the generic plotter interface."""

    def test_warn_unknown_data_key(self):
        """Test that setting an unknown data-key raises a warning."""
        plotter = MockPlotter(MockDrawer())

        # TODO: Add check for no-warnings. assertNoWarns only available from Python 3.10+

        # An unknown data-key must raise a warning if it is used to set series data.
        with self.assertWarns(UserWarning):
            plotter.set_series_data("dummy_series", unknown_data_key=[0, 1, 2])

    def test_series_data_end_to_end(self):
        """Test end-to-end for series data setting and retrieving."""
        plotter = MockPlotter(MockDrawer())

        series_data = {
            "seriesA": {
                "x": 0,
                "y": "1",
                "z": [2],
            },
            "seriesB": {
                "x": 1,
                "y": 0.5,
            },
        }
        unexpected_data = ["a", True, 0]
        expected_series_data = deepcopy(series_data)
        expected_series_data["seriesA"]["unexpected_data"] = unexpected_data

        for series, data in series_data.items():
            plotter.set_series_data(series, **data)

        with self.assertWarns(UserWarning):
            plotter.set_series_data("seriesA", unexpected_data=unexpected_data)

        for series, data in expected_series_data.items():
            self.assertTrue(series in plotter.series)
            self.assertTrue(plotter.data_exists_for(series, list(data.keys())))
            for data_key, value in data.items():
                # Must index [0] for `data_for` as it returns a tuple.
                self.assertEqual(value, plotter.data_for(series, data_key)[0])

    def test_supplementary_data_end_to_end(self):
        """Test end-to-end for figure data setting and retrieval."""
        plotter = MockPlotter(MockDrawer())

        expected_supplementary_data = {
            "report_text": "Lorem ipsum",
            "supplementary_data_key": 3e9,
        }

        plotter.set_supplementary_data(**expected_supplementary_data)

        # Check if figure data has been stored and can be retrieved
        for key, expected_value in expected_supplementary_data.items():
            actual_value = plotter.supplementary_data[key]
            self.assertEqual(
                expected_value,
                actual_value,
                msg=f"Actual figure data value for {key} data-key is not as expected: {actual_value} "
                f"(actual) vs {expected_value} (expected)",
            )
