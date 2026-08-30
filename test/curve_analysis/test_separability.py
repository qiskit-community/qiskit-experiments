# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test separability diagnostics."""

from test.base import QiskitExperimentsTestCase

from qiskit_experiments.curve_analysis.separability import (
    TASK_EXPONENTS,
    peak_separability,
)


class TestPeakSeparability(QiskitExperimentsTestCase):
    """Test peak_separability."""

    def test_regimes(self):
        """Regime labels follow the documented thresholds."""
        self.assertEqual(peak_separability(10, 1).regime, "resolved")
        self.assertEqual(peak_separability(1, 1).regime, "marginal")
        self.assertEqual(peak_separability(0.1, 1).regime, "unresolved")

    def test_no_amplification_when_resolved(self):
        """Well separated features have amplification one for all tasks."""
        report = peak_separability(10, 1)
        for value in report.amplification.values():
            self.assertEqual(value, 1.0)

    def test_hierarchy(self):
        """Overlapping features amplify with exponents 1, 2 and 3."""
        report = peak_separability(0.1, 1)
        amp = report.amplification
        self.assertAlmostEqual(amp["amplitudes_fixed_frequencies"], 10.0)
        self.assertAlmostEqual(amp["frequencies"], 100.0)
        self.assertAlmostEqual(amp["amplitudes_free_frequencies"], 1000.0)
        self.assertEqual(
            set(amp), set(TASK_EXPONENTS)
        )

    def test_invalid_inputs(self):
        """Non positive inputs raise."""
        with self.assertRaises(ValueError):
            peak_separability(0, 1)
        with self.assertRaises(ValueError):
            peak_separability(1, -1)
