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

from itertools import combinations, product
from test.base import QiskitExperimentsTestCase
from typing import Any, Dict, List

import ddt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.testing.compare import calculate_rms

from qiskit_experiments.visualization import MplDrawer

from .mock_plotter import MockPlotter


@ddt.ddt
class TestPlotterAndMplDrawer(QiskitExperimentsTestCase):
    """Test generic plotter with Matplotlib drawer."""

    def tearDown(self):
        """Clean up test case state"""
        plt.close("all")
        super().tearDown()

    def test_end_to_end_short(self):
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

    def test_scale(self):
        """Test the xscale and yscale figure options."""
        plotter = MockPlotter(MplDrawer(), plotting_enabled=True)
        plotter.set_figure_options(
            xscale="quadratic",
            yscale="log",
        )

        plotter.figure()
        ax = plotter.drawer._axis
        self.assertEqual(ax.get_xscale(), "function")
        self.assertEqual(ax.get_yscale(), "log")

    @ddt.data(
        {str: ["0", "1", "2"], int: [0, 1, 2]},
        {str: [str(0.0), str(1.0), str(2.0)], float: [0.0, 1.0, 2.0]},
    )
    def test_series_names_different_types(self, series_names: Dict[type, List[Any]]):
        """Test whether plotter with MplDrawer draws the correct figure regardless of series-name type.

        This test creates a MockPlotter for types ``str`` and ``int``. The series-names are integers but
        of both these types (see ``series_names`` list). MplDrawer should treat both of these the same
        and the legends should look the same. This test makes sure the generated figures are equivalent,
        showing that the type for series-names doesn't matter as long as they can be converted into a
        string.
        """

        # Create Matplotlib axes that use a PNG backend. The default backend, FigureCanvasSVG, does not
        # have `buffer_rgba()` which is needed to compute the difference between two figures in this
        # method. We need to set the axes as MplDrawer will use
        # `qiskit_experiments.framework.matplotlib.get_non_gui_ax` by default; which uses an SVG backend.
        plt.switch_backend("Agg")
        axes = {}
        for key in series_names.keys():
            fig = plt.figure()
            axes[key] = fig.subplots(1, 1)

        # Create plotters, one per type, and set the axis.
        plotters = {t: MockPlotter(MplDrawer(), plotting_enabled=True) for t in series_names.keys()}
        for key in plotters.keys():
            plotters[key].set_options(axis=axes[key])

        # Tolerance to be used when comparing images (i.e., with calculate_rms)
        tol = 1e-2

        # Get generic series data-keys (i.e., excluding "x", "y", and "z") to assist with adding data to
        # plotters.
        data_keys = [
            k for k in plotters[str].expected_series_data_keys() if k not in ["x", "y", "z"]
        ]

        # Create list of legend plot-types to use with `MockPlotter.enable_legend_for`.
        legend_plot_types = ["scatter"]

        # Add plotters using series-names of specific types
        for ((plotter_type, plotter), (series_name_type, type_series_names)), data_key in product(
            zip(plotters.items(), series_names.items()), data_keys
        ):
            for series_name in type_series_names:
                # Sanity check for plotter and series_name types
                self.assertEqual(plotter_type, series_name_type)
                # Verify that the series_name type is the same as the stated series_name type.
                self.assertEqual(series_name_type, type(series_name))

                # Add random data for the given data-key and series-name. Values do not matter.
                plotter.set_series_data(series_name, **{data_key: [0, 1, 2, 3]})

                # Enable legends.
                for plot_type in legend_plot_types:
                    plotter.enable_legend_for(series_name, plot_type)

        # Generate figure and save to buffers for comparison.
        figure_data = {}
        for plotter_type, plotter in plotters.items():
            figure = plotter.figure().figure
            figure.canvas.draw()
            figure_data[plotter_type] = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)

        # Compare root-mean-squared error between two images.
        for (fig1_type, fig1), (fig2_type, fig2) in combinations(figure_data.items(), 2):
            rms = calculate_rms(fig1, fig2)
            self.assertLessEqual(
                rms,
                tol,
                msg="RMS of per-pixel difference between figures for types "
                f"{fig1_type} and {fig2_type}. RMS {rms:0.4e} is greater than tolerance {tol:0.4e}.",
            )
