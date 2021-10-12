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

"""Test the half angle experiment."""

import numpy as np

from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase

from qiskit_experiments.test.mock_iq_backend import MockIQBackend
from qiskit_experiments.library import HalfAngle


class HalfAngleTestBackend(MockIQBackend):
    """A simple and primitive backend, to be run by the half angle tests."""

    def __init__(self, error: float):
        """Initialize the class."""
        super().__init__()
        self._error = error

    def _compute_probability(self, circuit: QuantumCircuit) -> float:
        """Returns the probability of measuring the excited state."""

        n_gates = circuit.metadata["xval"]

        return 0.5 * np.sin((-1) ** n_gates * n_gates * self._error) + 0.5


class TestHalfAngle(QiskitTestCase):
    """Class to test the half angle experiment."""

    def test_end_to_end(self):
        """Test a full experiment end to end."""

        tol = 0.005
        for error in [-0.05, -0.02, 0.02, 0.05]:
            hac = HalfAngle(0)
            exp_data = hac.run(HalfAngleTestBackend(error)).block_for_results()
            d_theta = exp_data.analysis_results(1).value.value

            self.assertTrue(abs(d_theta - error) < tol)
