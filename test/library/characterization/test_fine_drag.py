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

"""Test fine drag calibration experiment."""

from test.base import QiskitExperimentsTestCase

from qiskit.circuit import Gate
from qiskit_ibm_runtime.fake_provider import FakeArmonkV2

from qiskit_experiments.library import FineDrag, FineXDrag
from qiskit_experiments.test.mock_iq_backend import MockIQBackend
from qiskit_experiments.test.mock_iq_helpers import MockIQFineDragHelper as FineDragHelper


class TestFineDrag(QiskitExperimentsTestCase):
    """Tests of the fine DRAG experiment."""

    def test_circuits(self):
        """Test the circuits of the experiment."""

        drag = FineDrag([0], Gate("Drag", num_qubits=1, params=[]))
        drag.backend = FakeArmonkV2()
        for circuit in drag.circuits()[1:]:
            for idx, name in enumerate(["Drag", "rz", "Drag", "rz"]):
                self.assertEqual(circuit.data[idx].operation.name, name)

    def test_end_to_end(self):
        """A simple test to check if the experiment will run and fit data."""
        exp_data = FineXDrag([0]).run(MockIQBackend(FineDragHelper()))
        self.assertExperimentDone(exp_data)

        self.assertEqual(exp_data.analysis_results("d_theta").quality, "good")

    def test_circuits_roundtrip_serializable(self):
        """Test circuits serialization of the experiment."""
        drag = FineXDrag([0])
        drag.backend = FakeArmonkV2()
        self.assertRoundTripSerializable(drag._transpiled_circuits())

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = FineDrag([0], Gate("Drag", num_qubits=1, params=[]))
        config = exp.config()
        loaded_exp = FineDrag.from_config(config)
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqual(config, loaded_exp.config())
