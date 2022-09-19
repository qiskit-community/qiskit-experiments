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

"""Test ZZ Phase experiments."""

from typing import Dict, List

from test.base import QiskitExperimentsTestCase

import numpy as np

from qiskit import QuantumCircuit

from qiskit_experiments.library import ZZRamsey
from qiskit_experiments.test.mock_iq_backend import MockIQBackend
from qiskit_experiments.test.mock_iq_helpers import MockIQExperimentHelper


class ZZRamseyHelper(MockIQExperimentHelper):
    """A mock backend for the ZZRamsey experiment"""

    def __init__(self, zz: float):
        super().__init__()
        self.zz_freq = zz

    def compute_probabilities(self, circuits: List[QuantumCircuit]) -> List[Dict[str, float]]:
        """Return the probability of the circuit."""

        probabilities = []
        for circuit in circuits:
            series = circuit.metadata["series"]
            delay = circuit.metadata["xval"]

            if series == "0":
                freq = (-1 * self.zz_freq) / 2
            else:
                freq = self.zz_freq / 2
            rz, _, _ = next(i for i in circuit.data if i[0].name == "u1")
            phase = float(rz.params[0])

            prob1 = 0.5 - 0.5 * np.cos(2 * np.pi * freq * delay + phase)

            probabilities.append({"0": 1 - prob1, "1": prob1})

        return probabilities


class TestZZRamsey(QiskitExperimentsTestCase):
    """Tests for the ZZ Ramsey experiment."""

    def test_end_to_end(self):
        """Test that we can run on a mock backend and perform a fit."""

        test_tol = 0.05

        ramsey = ZZRamsey((0, 1))

        for zz in [2e5, -3e5]:
            backend = MockIQBackend(ZZRamseyHelper(zz))
            test_data = ramsey.run(backend)
            self.assertExperimentDone(test_data)
            meas_shift = test_data.analysis_results("zz").value.n
            self.assertLess(abs(meas_shift - zz), abs(test_tol * zz))

    def test_experiment_config(self):
        """Test config roundtrips"""
        exp = ZZRamsey((0, 1))
        loaded_exp = ZZRamsey.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = ZZRamsey((0, 1))
        self.assertRoundTripSerializable(exp, self.json_equiv)
