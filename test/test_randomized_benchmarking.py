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

"""Test version string generation."""

from qiskit.test import QiskitTestCase
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error
from qiskit import Aer
from qiskit_experiments.randomized_benchmarking import RBExperiment, InterleavedRBExperiment, PurityRBExperiment

class TestStandardRB(QiskitTestCase):
    """Test version string generation."""
    def setUp(self):
        super().setUp()
        self.nQ = 3
        self.nseeds = 5
        self.nCliffs = [1,10,20]
        self.noise_model = NoiseModel()
        p1Q = 0.002
        p2Q = 0.01
        self.noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), 'x')
        self.noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), 'sx')
        self.noise_model.add_all_qubit_quantum_error(depolarizing_error(p2Q, 2), 'cx')
        self.backend = Aer.get_backend('qasm_simulator')
        self.shots = 200

    def test_experiment_data(self):
        """Test version string generation."""
        exp = RBExperiment(nseeds=self.nseeds, qubits=[1], lengths=self.nCliffs)
        experiment_data = exp.run(self.backend, noise_model=self.noise_model, shots=self.shots)
        for d in experiment_data._data:
            self.assertEqual(sum(d['counts'].values()), self.shots)
            meta = d['metadata']
            self.assertEqual(meta['group_type'], 'clifford')
            self.assertRegex(meta['circuit_name'], 'rb_length_\d+_seed_\d+')
            self.assertEqual(meta['experiment_type'], RBExperiment.__name__)
