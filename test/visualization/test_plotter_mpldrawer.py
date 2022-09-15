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
Test integration of plotter with Matplotlib drawer.
"""

from test.base import QiskitExperimentsTestCase

import matplotlib
from qiskit_experiments.visualization import MplDrawer

from .mock_plotter import MockPlotter


class TestPlotterAndMplDrawer(QiskitExperimentsTestCase):
    """Test generic plotter with Matplotlib drawer."""

    def test_end_to_end(self):
        """Test whether plotter with MplDrawer returns a figure."""
        plotter = MockPlotter(MplDrawer())
        plotter.set_series_data(
            "seriesA", x=[0, 1, 2, 3, 4, 5], y=[0, 1, 0, 1, 0, 1], z=[0.1, 0.1, 0.3, 0.4, 0.0]
        )
        fig = plotter.figure()

        # Expect something
        self.assertTrue(fig is not None)

        # Expect a specific type
        self.assertTrue(isinstance(fig, matplotlib.pyplot.Figure))
