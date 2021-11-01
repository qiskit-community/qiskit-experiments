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

import numpy as np

from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeArmonk
import qiskit.pulse as pulse

from qiskit_experiments.library import FineDrag, FineXDrag
from qiskit_experiments.test.mock_iq_backend import DragBackend


class FineDragTestBackend(DragBackend):
    """A simple and primitive backend, to be run by the rough drag tests."""

    def _compute_probability(self, circuit: QuantumCircuit) -> float:
        """Returns the probability based on the beta, number of gates, and leakage."""
        n_gates = circuit.count_ops().get("rz", 0) // 2

        return 0.5 * np.sin(n_gates * self._error) + 0.5


class TestFineDrag(QiskitTestCase):
    """Tests of the fine DRAG experiment."""

    def setUp(self):
        """Setup test variables."""
        super().setUp()

        with pulse.build(name="Drag") as schedule:
            pulse.play(pulse.Drag(160, 0.5, 40, 0.3), pulse.DriveChannel(0))

        self.schedule = schedule

    def test_circuits(self):
        """Test the circuits of the experiment."""

        drag = FineDrag(0)
        drag.set_experiment_options(schedule=self.schedule)
        drag.backend = FakeArmonk()
        for circuit in drag.circuits()[1:]:
            for idx, name in enumerate(["Drag", "rz", "Drag", "rz"]):
                self.assertEqual(circuit.data[idx][0].name, name)

    def test_end_to_end(self):
        """A simple test to check if the experiment will run and fit data."""

        drag = FineDrag(0)
        drag.set_experiment_options(schedule=self.schedule)
        drag.set_transpile_options(basis_gates=["rz", "Drag", "ry"])
        exp_data = drag.run(FineDragTestBackend()).block_for_results()

        self.assertEqual(exp_data.analysis_results(0).quality, "good")

    def test_end_to_end_no_schedule(self):
        """Test that we can run without a schedule."""

        exp_data = FineXDrag(0).run(FineDragTestBackend()).block_for_results()

        self.assertEqual(exp_data.analysis_results(0).quality, "good")

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = FineDrag(0)
        config = exp.config
        loaded_exp = FineDrag.from_config(config)
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqual(config, loaded_exp.config)
