# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Package to test fitting in calibration module."""

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit_experiments.calibration.analysis.trigonometric import freq_guess


class TestCalibrationFitting(QiskitTestCase):
    """Class to test the functionality of fitters."""

    def test_frequency_guess(self):
        """Test the initial frequency estimation function."""

        xvals = np.linspace(0, 1, 200)
        yvals = np.cos(2*np.pi*xvals) + 1.1

        freq = freq_guess(xvals, yvals)

        self.assertAlmostEqual(freq, 1.0, places=2)
