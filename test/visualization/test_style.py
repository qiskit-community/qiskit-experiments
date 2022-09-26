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
Test visualization plotter.
"""

from copy import copy
from test.base import QiskitExperimentsTestCase
from typing import Tuple

from qiskit_experiments.visualization import PlotStyle


class TestPlotStyle(QiskitExperimentsTestCase):
    """Test PlotStyle"""

    @classmethod
    def _dummy_styles(cls) -> Tuple[PlotStyle, PlotStyle, PlotStyle, PlotStyle]:
        """Returns dummy input styles for PlotStyle tests.

        Returns:
            PlotStyle: First input style.
            PlotStyle: Second input style.
            PlotStyle: Expected style combining second into first.
            PlotStyle: Expected style combining first into second.
        """
        custom_1 = PlotStyle(overwrite_field=0, unchanged_field_A="Test", none_field=[0, 1, 2, 3])
        custom_2 = PlotStyle(overwrite_field=6, unchanged_field_B=0.5, none_field=None)
        expected_12 = PlotStyle(
            overwrite_field=6,
            unchanged_field_A="Test",
            unchanged_field_B=0.5,
            none_field=None,
        )
        expected_21 = PlotStyle(
            overwrite_field=0,
            unchanged_field_A="Test",
            unchanged_field_B=0.5,
            none_field=[0, 1, 2, 3],
        )
        return custom_1, custom_2, expected_12, expected_21

    def test_default_contains_necessary_fields(self):
        """Test that expected fields are set in the default style."""
        default = PlotStyle.default_style()
        expected_not_none_fields = [
            "figsize",
            "legend_loc",
            "tick_label_size",
            "axis_label_size",
            "text_box_rel_pos",
            "text_box_text_size",
        ]
        for field in expected_not_none_fields:
            self.assertIsNotNone(getattr(default, field))

    def test_update(self):
        """Test that styles can be updated."""
        custom_1, custom_2, expected_12, expected_21 = self._dummy_styles()

        # copy(...) is needed as .update() modifies the style instance
        actual_12 = copy(custom_1)
        actual_12.update(custom_2)
        actual_21 = copy(custom_2)
        actual_21.update(custom_1)

        self.assertEqual(actual_12, expected_12)
        self.assertEqual(actual_21, expected_21)

    def test_merge(self):
        """Test that styles can be merged."""
        custom_1, custom_2, expected_12, expected_21 = self._dummy_styles()

        self.assertEqual(PlotStyle.merge(custom_1, custom_2), expected_12)
        self.assertEqual(PlotStyle.merge(custom_2, custom_1), expected_21)

    def test_field_access(self):
        """Test that fields are accessed correctly"""
        dummy_style = PlotStyle(
            x="x",
            # y isn't assigned and therefore doesn't exist in dummy_style
        )

        self.assertEqual(dummy_style.x, "x")

        # This should throw as we haven't assigned y
        with self.assertRaises(AttributeError):
            # Disable pointless-statement as accessing style fields can raise an exception.
            # pylint: disable = pointless-statement
            dummy_style.y

    def test_dict(self):
        """Test that PlotStyle can be converted into a dictionary."""
        dummy_style = PlotStyle(
            a="a",
            b=0,
            c=[1, 2, 3],
        )
        expected_dict = {
            "a": "a",
            "b": 0,
            "c": [1, 2, 3],
        }
        actual_dict = dummy_style.__dict__
        self.assertDictEqual(actual_dict, expected_dict, msg="PlotStyle dict is not as expected.")

        # Add a new variable
        dummy_style.new_variable = 5e9
        expected_dict["new_variable"] = 5e9
        actual_dict = dummy_style.__dict__
        self.assertDictEqual(
            actual_dict,
            expected_dict,
            msg="PlotStyle dict is not as expected, with post-init variables.",
        )

    def test_update_dict(self):
        """Test that PlotStyle dictionary is correct when updated."""
        custom_1, custom_2, expected_12, expected_21 = self._dummy_styles()

        # copy(...) is needed as .update() modifies the style instance
        actual_12 = copy(custom_1)
        actual_12.update(custom_2)
        actual_21 = copy(custom_2)
        actual_21.update(custom_1)

        self.assertDictEqual(actual_12.__dict__, expected_12.__dict__)
        self.assertDictEqual(actual_21.__dict__, expected_21.__dict__)

    def test_merge(self):
        """Test that PlotStyle dictionary is correct when merged."""
        custom_1, custom_2, expected_12, expected_21 = self._dummy_styles()

        self.assertDictEqual(PlotStyle.merge(custom_1, custom_2).__dict__, expected_12.__dict__)
        self.assertDictEqual(PlotStyle.merge(custom_2, custom_1).__dict__, expected_21.__dict__)
