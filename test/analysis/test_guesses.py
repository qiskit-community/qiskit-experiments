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
from qiskit.test import QiskitTestCase
from ddt import ddt, data
from qiskit_experiments.analysis import guesses


@ddt
class TestGuesses(QiskitTestCase):
    __tolerance_percent__ = 0.2

    def setUp(self):
        super().setUp()

    def assertAlmostEqualAbsolute(self, value: float, ref_value: float):
        delta = TestGuesses.__tolerance_percent__ * np.abs(ref_value)
        self.assertAlmostEqual(value, ref_value, delta=delta)

    @data(1.1, 2.0, 1.6, -1.4, 4.5)
    def test_frequency_fft(self, freq: float):
        x = np.linspace(-1, 1, 101)
        y = 0.3 * np.cos(2 * np.pi * freq * x + 0.5) + 1.2

        freq_guess = guesses.frequency(x, y, method="FFT")

        # breakpoint()
        self.assertAlmostEqualAbsolute(freq_guess, np.abs(freq))

    @data(1.1, 2.0, 1.6, -1.4, 4.5)
    def test_frequency_acf(self, freq: float):
        x = np.linspace(-1, 1, 101)
        y = 0.3 * np.cos(2 * np.pi * freq * x + 0.5) + 1.2

        freq_guess = guesses.frequency(x, y, method="ACF")
        self.assertAlmostEqualAbsolute(freq_guess, np.abs(freq))




