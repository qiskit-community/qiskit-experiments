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
        drawer.scatter([0, 1, 2], [0, 1, 2], name="seriesA")
        drawer.scatter([0, 1, 2], [0, 1, 2], [0.1, 0.1, 0.1], None, name="seriesA")
        drawer.line([3, 2, 1], [1, 2, 3], name="seriesB")
        drawer.filled_x_area([0, 1, 2, 3], [1, 2, 1, 2], [-1, -2, -1, -2], name="seriesB")
        drawer.filled_y_area([-1, 0, 1, 2], [-1, -2, -1, -2], [1, 2, 1, 2], name="seriesB")
        drawer.textbox(r"Dummy report text with LaTex $\beta$")

        # Get result
        fig = drawer.figure

        # Check that
        self.assertTrue(fig is not None)
        self.assertTrue(isinstance(fig, matplotlib.pyplot.Figure))
