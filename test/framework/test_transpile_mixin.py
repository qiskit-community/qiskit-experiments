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

"""Tests for transpile mixin."""

from typing import Sequence

from test.base import QiskitExperimentsTestCase
from test.fake_experiment import FakeAnalysis

from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2

from qiskit_experiments.framework import SimpleCircuitExtender, BaseExperiment
from qiskit_experiments.framework.composite import ParallelExperiment


class TestSimpleCircuitExtender(QiskitExperimentsTestCase):
    """A test for SimpleCircuitExtender MixIn."""

    def test_transpiled_single_qubit_circuits(self):
        """Test fast-transpile with single qubit circuit."""

        class _MockExperiment(SimpleCircuitExtender, BaseExperiment):
            def circuits(self) -> list:
                qc1 = QuantumCircuit(1, 1)
                qc1.x(0)
                qc1.measure(0, 0)
                qc1.metadata = {"test_val": "123"}

                qc2 = QuantumCircuit(1, 1)
                qc2.sx(0)
                qc2.measure(0, 0)
                qc2.metadata = {"test_val": "456"}
                return [qc1, qc2]

        num_qubits = 10

        mock_backend = GenericBackendV2(num_qubits, basis_gates=["sx", "rz", "x"])
        exp = _MockExperiment((3,), backend=mock_backend)
        test_circs = exp._transpiled_circuits()

        self.assertEqual(len(test_circs), 2)
        c0, c1 = test_circs

        # output size
        self.assertEqual(len(c0.qubits), num_qubits)
        self.assertEqual(len(c1.qubits), num_qubits)

        # metadata
        self.assertDictEqual(c0.metadata, {"test_val": "123"})

        # qubit index of X gate
        self.assertEqual(c0.qubits.index(c0.data[0][1][0]), 3)

        # creg index of measure
        self.assertEqual(c0.clbits.index(c0.data[1][2][0]), 0)

        # metadata
        self.assertDictEqual(c1.metadata, {"test_val": "456"})

        # qubit index of SX gate
        self.assertEqual(c1.qubits.index(c1.data[0][1][0]), 3)

        # creg index of measure
        self.assertEqual(c1.clbits.index(c1.data[1][2][0]), 0)

    def test_transpiled_two_qubit_circuits(self):
        """Test fast-transpile with two qubit circuit."""

        class _MockExperiment(SimpleCircuitExtender, BaseExperiment):
            def circuits(self) -> list:
                qc = QuantumCircuit(2, 2)
                qc.cx(0, 1)
                qc.measure(0, 0)
                qc.measure(1, 1)
                return [qc]

        num_qubits = 10

        mock_backend = GenericBackendV2(num_qubits, basis_gates=["sx", "rz", "x", "cx"])
        exp = _MockExperiment((9, 2), backend=mock_backend)
        test_circ = exp._transpiled_circuits()[0]

        self.assertEqual(len(test_circ.qubits), num_qubits)

        # qubit index of CX control qubit
        self.assertEqual(test_circ.qubits.index(test_circ.data[0][1][0]), 9)

        # qubit index of CX target qubit
        self.assertEqual(test_circ.qubits.index(test_circ.data[0][1][1]), 2)

        # creg index of measure
        self.assertEqual(test_circ.clbits.index(test_circ.data[1][2][0]), 0)
        self.assertEqual(test_circ.clbits.index(test_circ.data[2][2][0]), 1)

    def test_empty_backend(self):
        """Test fast-transpile without backend."""

        class _MockExperiment(SimpleCircuitExtender, BaseExperiment):
            def circuits(self) -> list:
                qc = QuantumCircuit(1, 1)
                qc.x(0)
                qc.measure(0, 0)

                return [qc]

        exp = _MockExperiment((10,))
        test_circ = exp._transpiled_circuits()[0]

        self.assertEqual(len(test_circ.qubits), 11)

        # qubit index of X gate
        self.assertEqual(test_circ.qubits.index(test_circ.data[0][1][0]), 10)

    def test_empty_backend_with_parallel(self):
        """Test fast-transpile without backend. Circuit qubit location must not overlap."""

        class _MockExperiment(SimpleCircuitExtender, BaseExperiment):
            def __init__(self, physical_qubits):
                super().__init__(physical_qubits, FakeAnalysis())

            def circuits(self) -> list:
                qc = QuantumCircuit(1, 1)
                qc.x(0)
                qc.measure(0, 0)

                return [qc]

        exp1 = _MockExperiment((3,))
        exp2 = _MockExperiment((15,))
        pexp = ParallelExperiment([exp1, exp2], flatten_results=True)
        test_circ = pexp._transpiled_circuits()[0]

        self.assertEqual(len(test_circ.qubits), 16)

        # qubit index of X gate
        self.assertEqual(test_circ.qubits.index(test_circ.data[0][1][0]), 3)
        self.assertEqual(test_circ.qubits.index(test_circ.data[2][1][0]), 15)
