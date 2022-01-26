"""Tests for the restless simulator."""

import numpy as np

from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase

from qiskit_experiments.test.mock_iq_backend import RestlessSimulator


class TestRestlessSimulator(QiskitTestCase):
    def test_single_qubit_circuit(self):
        """Test that we get the correct outcomes for simple circuits."""

        # Test a Hadamard circuit
        circ = QuantumCircuit(1)
        circ.h(0)
        circ.measure_all()

        sim = RestlessSimulator()
        sim([circ])

        probs = sim.probabilities

        self.assertEqual(len(probs), 2)
        self.assertTrue(np.allclose(probs[(0, "0")], [0.5, 0.5]))
        self.assertTrue(np.allclose(probs[(0, "1")], [0.5, 0.5]))

        # Test an X gate circuit
        circ = QuantumCircuit(1)
        circ.x(0)
        circ.measure_all()

        sim = RestlessSimulator()
        sim([circ])

        probs = sim.probabilities

        self.assertEqual(len(probs), 2)
        self.assertTrue(np.allclose(probs[(0, "0")], [0.0, 1.0]))
        self.assertTrue(np.allclose(probs[(0, "1")], [1.0, 0.0]))

    def test_two_qubit_circuit(self):
        """Test a simple two-qubit circuit."""
        circ1 = QuantumCircuit(2)
        circ1.h(0)
        circ1.x(1)
        circ1.measure_all()

        sim = RestlessSimulator()
        sim([circ1])

        probs = sim.probabilities
        self.assertEqual(len(probs), 4)
        self.assertTrue(np.allclose(probs[(0, "00")], [0.0, 0.0, 0.5, 0.5]))
        self.assertTrue(np.allclose(probs[(0, "01")], [0.0, 0.0, 0.5, 0.5]))
        self.assertTrue(np.allclose(probs[(0, "10")], [0.5, 0.5, 0.0, 0.0]))
        self.assertTrue(np.allclose(probs[(0, "11")], [0.5, 0.5, 0.0, 0.0]))

        circ2 = QuantumCircuit(2)
        circ2.h(0)
        circ2.h(1)
        circ2.measure_all()

        sim = RestlessSimulator()
        sim([circ2])

        probs = sim.probabilities
        for state in ["00", "01", "10", "11"]:
            self.assertTrue(np.allclose(probs[(0, state)], [0.25] * 4))

        sim = RestlessSimulator()
        memory = sim([circ1, circ2])

        probs = sim.probabilities
        self.assertTrue(np.allclose(probs[(0, "00")], [0.0, 0.0, 0.5, 0.5]))
        self.assertTrue(np.allclose(probs[(0, "01")], [0.0, 0.0, 0.5, 0.5]))
        self.assertTrue(np.allclose(probs[(0, "10")], [0.5, 0.5, 0.0, 0.0]))
        self.assertTrue(np.allclose(probs[(0, "11")], [0.5, 0.5, 0.0, 0.0]))
        for state in ["00", "01", "10", "11"]:
            self.assertTrue(np.allclose(probs[(1, state)], [0.25] * 4))

        outcomes, expected = set(memory), {"00", "01", "10", "11"}
        self.assertEqual(outcomes, expected)
        self.assertEqual(len(memory), 2 * sim.shots)

    def test_experiment_data(self):
        """Test that we can get back some experiment data."""

        circ1 = QuantumCircuit(1)
        circ1.x(0)
        circ1.measure_all()
        circ1.metadata = {"test": 1}

        circ2 = QuantumCircuit(1)
        circ2.measure_all()
        circ2.metadata = {"test": 2}

        sim = RestlessSimulator(shots=3)

        _ = sim([circ1, circ2])
        exp_data = sim.experiment_data

        data1 = {"memory": ["0x1", "0x0", "0x1"], "metadata": {"test": 1}}
        data2 = {"memory": ["0x1", "0x0", "0x1"], "metadata": {"test": 2}}

        self.assertEqual(exp_data.data(0), data1)
        self.assertEqual(exp_data.data(1), data2)
