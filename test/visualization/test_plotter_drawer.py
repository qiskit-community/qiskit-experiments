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
from itertools import product
from test.base import QiskitExperimentsTestCase

from qiskit_experiments.framework import Options
from qiskit_experiments.visualization import PlotStyle

from .mock_drawer import MockDrawer
from .mock_plotter import MockPlotter


def dummy_plotter(plotting_enabled: bool = False) -> MockPlotter:
    """Return a MockPlotter with dummy option values.

    Args:
        plotting_enabled: Whether the returned plotter should actually draw.

    Returns:
        MockPlotter: A dummy plotter.
    """
    plotter = MockPlotter(MockDrawer(), plotting_enabled)
    # Set dummy plot options to update
    plotter.set_figure_options(
        xlabel="xlabel",
        ylabel="ylabel",
        figure_title="figure_title",
        non_drawer_options="should not be set",
    )
    plotter.set_options(
        style=PlotStyle(test_param="test_param", overwrite_param="new_overwrite_param_value")
    )
    return plotter


class TestPlotterAndDrawerIntegration(QiskitExperimentsTestCase):
    """Test plotter and drawer integration."""

    def assertOptionsEqual(
        self,
        options1: Options,
        options2: Options,
        msg_prefix: str = "",
        only_assert_for_intersection: bool = False,
    ):
        """Asserts that two options are the same by checking each individual option.

        This method is easier to read than a standard equality assertion as individual option names are
        printed.

        Args:
            options1: The first Options instance to check.
            options2: The second Options instance to check.
            msg_prefix: A prefix to add before assert messages.
            only_assert_for_intersection: If True, will only check options that are in both Options
                instances. Defaults to False.
        """
        # Get combined field names
        if only_assert_for_intersection:
            fields = set(options1._fields.keys()).intersection(set(options2._fields.keys()))
        else:
            fields = set(options1._fields.keys()).union(set(options2._fields.keys()))

        # Check individual options.
        for key in fields:
            # Check if the option exists in both
            self.assertTrue(
                hasattr(options1, key),
                msg=f"[{msg_prefix}] Expected field {key} in both, but only found in one: not in "
                f"{options1}.",
            )
            self.assertTrue(
                hasattr(options2, key),
                msg=f"[{msg_prefix}] Expected field {key} in both, but only found in one: not in "
                f"{options2}.",
            )
            self.assertEqual(
                getattr(options1, key),
                getattr(options2, key),
                msg=f"[{msg_prefix}] Expected equal values for option '{key}': "
                f"{getattr(options1, key),} vs {getattr(options2,key)}",
            )

    def test_figure_options(self):
        """Test copying and passing of plot-options between plotter and drawer."""
        plotter = dummy_plotter()

        # Expected options
        expected_figure_options = copy(plotter.drawer.figure_options)
        expected_figure_options.xlabel = "xlabel"
        expected_figure_options.ylabel = "ylabel"
        expected_figure_options.figure_title = "figure_title"

        # Expected style
        expected_custom_style = PlotStyle(
            test_param="test_param", overwrite_param="new_overwrite_param_value"
        )
        plotter.set_options(style=expected_custom_style)
        expected_full_style = PlotStyle.merge(
            plotter.drawer.options.default_style, expected_custom_style
        )
        expected_figure_options.custom_style = expected_custom_style

        # Call plotter.figure() to force passing of figure_options to drawer
        plotter.figure()

        ## Test values
        # Check style as this is a more detailed plot-option than others.
        self.assertEqual(expected_full_style, plotter.drawer.style)

        # Check individual plot-options, but only the intersection as those are the ones we expect to be
        # updated.
        self.assertOptionsEqual(expected_figure_options, plotter.drawer.figure_options, True)

        # Coarse equality check of figure_options
        self.assertEqual(
            expected_figure_options,
            plotter.drawer.figure_options,
            msg=rf"expected_figure_options = {expected_figure_options}\nactual_figure_options ="
            rf"{plotter.drawer.figure_options}",
        )

    def test_serializable(self):
        """Test that plotter is serializable."""
        original_plotter = dummy_plotter()

        def check_options(original, new):
            """Verifies that ``new`` plotter has the same options as ``original`` plotter."""
            self.assertOptionsEqual(original.options, new.options, "options")
            self.assertOptionsEqual(original.figure_options, new.figure_options, "figure_options")
            self.assertOptionsEqual(original.drawer.options, new.drawer.options, "drawer.options")
            self.assertOptionsEqual(
                original.drawer.figure_options, new.drawer.figure_options, "drawer.figure_options"
            )

        ## Check that plotter, BEFORE PLOTTING, survives serialization correctly.
        # HACK: A dedicated JSON encoder and decoder class would be better.
        # __json_<encode/decode>__ are not typically called, instead json.dumps etc. is called
        encoded = original_plotter.__json_encode__()
        decoded_plotter = original_plotter.__class__.__json_decode__(encoded)
        check_options(original_plotter, decoded_plotter)

        ## Check that plotter, AFTER PLOTTING, survives serialization correctly.
        original_plotter.figure()
        # HACK: A dedicated JSON encoder and decoder class would be better.
        # __json_<encode/decode>__ are not typically called, instead json.dumps etc. is called
        encoded = original_plotter.__json_encode__()
        decoded_plotter = original_plotter.__class__.__json_decode__(encoded)
        check_options(original_plotter, decoded_plotter)

    def test_end_to_end(self):
        """Tests end-to-end functionality of plotter with various data-keys."""
        plotter = dummy_plotter(plotting_enabled=True)

        # Add dummy data. We ignore "x", "y", and "z" as those are for other tests; but we want to add
        # data for all other expected series data-keys.
        data_keys = [k for k in plotter.expected_series_data_keys() if k not in ["x", "y", "z"]]
        series_names = ["seriesA", "seriesB"]
        for series_name, data_key in product(series_names, data_keys):
            plotter.set_series_data(series_name, **{data_key: [0, 1, 2, 3]})

        # Generate figure
        plotter.figure()

        # Check that all data-keys were logged exactly once per series.
        for series_name, data_key in product(series_names, data_keys):
            self.assertEqual(
                plotter.plotted_data_counter(series_name, data_key),
                1,
                msg=f"{data_key} was not plotted exactly once for series {series_name}: expected "
                "plotted data counter of 1 but got "
                f"{plotter.plotted_data_counter(series_name,data_key)} instead.",
            )
