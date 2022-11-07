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

from itertools import product
from test.base import QiskitExperimentsTestCase

import ddt
import matplotlib
import numpy as np

from qiskit_experiments.visualization import MplDrawer

from .mock_plotter import MockPlotter


@ddt.ddt
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

    @ddt.data(
        *list(product([(-3, "m"), (0, ""), (3, "k"), (6, "M")], [True, False], [True, False]))
    )
    def test_unit_scale(self, args):
        """Test whether axis labels' unit-prefixes scale correctly."""
        (exponent, prefix), xval_unit_scale, yval_unit_scale = args
        input_unit_x = "DUMMYX"
        input_unit_y = "DUMMYY"
        plotter = MockPlotter(MplDrawer(), plotting_enabled=True)
        plotter.set_figure_options(
            xlabel=" ",  # Dummy labels to force drawing of units.
            ylabel=" ",  #
            xval_unit=input_unit_x,
            yval_unit=input_unit_y,
            xval_unit_scale=xval_unit_scale,
            yval_unit_scale=yval_unit_scale,
        )

        n_points = 128
        plotter.set_series_data(
            "seriesA",
            x=np.random.rand(n_points) * 2 * (10**exponent),
            y=np.random.rand(n_points) * 2 * (10**exponent),
            z=np.random.rand(128) * (10**exponent),
        )

        plotter.figure()

        expected_unit_x = prefix + input_unit_x if xval_unit_scale else input_unit_x
        expected_unit_y = prefix + input_unit_y if yval_unit_scale else input_unit_y

        # Get actual labels
        ax = plotter.drawer._axis
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()

        # Check if expected units exist in the axis labels.
        for axis, actual_label, expected_units in zip(
            ["X", "Y"], [xlabel, ylabel], [expected_unit_x, expected_unit_y]
        ):
            self.assertTrue(actual_label is not None)
            self.assertTrue(
                actual_label.find(expected_units) > 0,
                msg=f"{axis} axis label does not contain unit: Could not find '{expected_units}' "
                f"in '{actual_label}'.",
            )
