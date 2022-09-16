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
Test Matplotlib Drawer.
"""

from copy import copy
from test.base import QiskitExperimentsTestCase

import matplotlib
from qiskit_experiments.visualization import MplDrawer


class TestMplDrawer(QiskitExperimentsTestCase):
    """Test MplDrawer."""

    def test_end_to_end(self):
        """Test that MplDrawer generates something."""
        drawer = MplDrawer()

        # Draw dummy data
        drawer.initialize_canvas()
        drawer.draw_raw_data([0, 1, 2], [0, 1, 2], "seriesA")
        drawer.draw_formatted_data([0, 1, 2], [0, 1, 2], [0.1, 0.1, 0.1], "seriesA")
        drawer.draw_line([3, 2, 1], [1, 2, 3], "seriesB")
        drawer.draw_confidence_interval([0, 1, 2, 3], [1, 2, 1, 2], [-1, -2, -1, -2], "seriesB")
        drawer.draw_report(r"Dummy report text with LaTex $\beta$")

        # Get result
        fig = drawer.figure

        # Check that
        self.assertTrue(fig is not None)
        self.assertTrue(isinstance(fig, matplotlib.pyplot.Figure))
