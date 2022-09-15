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
Test integration of plotters and drawers.
"""

from copy import copy
from test.base import QiskitExperimentsTestCase

from qiskit_experiments.visualization import PlotStyle

from .mock_drawer import MockDrawer
from .mock_plotter import MockPlotter


class TestPlotterAndDrawerIntegration(QiskitExperimentsTestCase):
    """Test plotter and drawer integration."""

    def test_plot_options(self):
        """Test copying and passing of plot-options between plotter and drawer."""
        plotter = MockPlotter(MockDrawer())

        # Expected options
        expected_plot_options = copy(plotter.drawer.plot_options)
        expected_plot_options.xlabel = "xlabel"
        expected_plot_options.ylabel = "ylabel"
        expected_plot_options.figure_title = "figure_title"

        # Expected style
        expected_custom_style = PlotStyle(
            test_param="test_param", overwrite_param="new_overwrite_param_value"
        )
        expected_full_style = PlotStyle.merge(
            plotter.drawer.options.default_style, expected_custom_style
        )
        expected_plot_options.custom_style = expected_custom_style

        # Set dummy plot options to update
        plotter.set_plot_options(
            xlabel="xlabel",
            ylabel="ylabel",
            figure_title="figure_title",
            non_drawer_options="should not be set",
        )
        plotter.set_options(
            style=PlotStyle(test_param="test_param", overwrite_param="new_overwrite_param_value")
        )

        # Call plotter.figure() to force passing of plot_options to drawer
        plotter.figure()

        ## Test values
        # Check style as this is a more detailed plot-option than others.
        self.assertEqual(expected_full_style, plotter.drawer.style)

        # Check individual plot-options.
        for key, value in expected_plot_options._fields.items():
            self.assertEqual(
                getattr(plotter.drawer.plot_options, key),
                value,
                msg=f"Expected equal values for plot option '{key}'",
            )

        # Coarse equality check of plot_options
        self.assertEqual(
            expected_plot_options,
            plotter.drawer.plot_options,
            msg=rf"expected_plot_options = {expected_plot_options}\nactual_plot_options ="
            rf"{plotter.drawer.plot_options}",
        )
