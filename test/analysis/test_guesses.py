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

"""Test parameter guess functions."""

import numpy as np
from ddt import ddt, data, unpack
from qiskit.test import QiskitTestCase

from qiskit_experiments.analysis import guesses


@ddt
class TestGuesses(QiskitTestCase):
    """Test for initial guess functions."""

    __tolerance_percent__ = 0.2

    def assertAlmostEqualAbsolute(self, value: float, ref_value: float):
        """A helper validation function that assumes relative error tolerance."""
        delta = TestGuesses.__tolerance_percent__ * np.abs(ref_value)
        self.assertAlmostEqual(value, ref_value, delta=delta)

    @data(1.1, 2.0, 1.6, -1.4, 4.5)
    def test_frequency(self, freq: float):
        """Test for frequency guess."""
        x = np.linspace(-1, 1, 101)
        y = 0.3 * np.cos(2 * np.pi * freq * x + 0.5) + 1.2

        freq_guess = guesses.frequency(x, y)

        self.assertAlmostEqualAbsolute(freq_guess, np.abs(freq))

    @data(1.2, -0.6, 0.1, 3.5, -4.1, 3.0)
    def test_exp_decay(self, alpha: float):
        """Test for exponential decay guess."""
        x = np.linspace(0, 1, 100)
        y = np.exp(alpha * x)

        alpha_guess = guesses.exp_decay(x, y)

        self.assertAlmostEqualAbsolute(alpha_guess, alpha)

    @data([1.2, 1.4], [-0.6, 2.5], [0.1, 2.3], [3.5, 1.1], [-4.1, 6.5], [3.0, 1.2])
    @unpack
    def test_exp_osci_decay(self, alpha, freq):
        """Test of exponential decay guess with oscillation."""
        x = np.linspace(0, 1, 100)
        y = np.exp(alpha * x) * np.cos(2 * np.pi * freq * x)

        alpha_guess = guesses.oscillation_exp_decay(x, y)

        self.assertAlmostEqualAbsolute(alpha_guess, alpha)
