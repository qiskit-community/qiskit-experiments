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

"""
Test tomography basis classes
"""

from typing import Sequence
from math import prod
from cmath import isclose
import itertools as it
from test.base import QiskitExperimentsTestCase
import qiskit.quantum_info as qi
from qiskit.circuit.library import HGate, XGate, YGate
from qiskit_experiments.library.tomography.basis import (
    PreparationBasis,
    MeasurementBasis,
    PauliMeasurementBasis,
    PauliPreparationBasis,
    Pauli6PreparationBasis,
    LocalMeasurementBasis,
    LocalPreparationBasis,
)


class TestLocalBasis(QiskitExperimentsTestCase):
    """Test tomography basis classes"""

    def _test_ideal_basis(self, basis: MeasurementBasis, qubits: Sequence[int]):
        """Test an ideal basis.

        For preparation bases this tests that the state matrices are
        equal to the state prepared by ideal simulation of the input
        circuit.

        For measurement bases this tests that the outcome matrices are
        eigenstates for basis circuit. By evolving the computation basis
        states for each outcome by the inverse of the circuit.
        """
        index_shape = basis.index_shape(qubits)
        indices = list(it.product(*[range(i) for i in index_shape]))

        if isinstance(basis, PreparationBasis):
            for index in indices:
                circ = basis.circuit(index, qubits)
                state = qi.Statevector(circ)
                mat = basis.matrix(index, qubits)
                expval = state.expectation_value(mat)
                self.assertTrue(isclose(expval, 1.0))

        elif isinstance(basis, MeasurementBasis):
            outcome_shape = basis.outcome_shape(qubits)
            outcomes = list(it.product(*[range(i) for i in outcome_shape]))
            for index in indices:
                circ = basis.circuit(index, qubits)
                circ.remove_final_measurements()
                adjoint = circ.inverse()
                for outcome_tup in outcomes:
                    outcome = prod(outcome_tup)
                    state = qi.Statevector.from_int(outcome, dims=2**circ.num_qubits)
                    state = state.evolve(adjoint)
                    mat = basis.matrix(index, outcome, qubits)
                    expval = state.expectation_value(mat)
                    self.assertTrue(
                        isclose(expval, 1.0), msg=f"{basis.name}, index={index}, outcome={outcome}"
                    )

    def test_pauli_mbasis_1q(self):
        """Test 1-qubit PauliMeasurementBasis ideal circuits and states"""
        self._test_ideal_basis(PauliMeasurementBasis(), [0])

    def test_pauli_mbasis_2q(self):
        """Test 2-qubit PauliMeasurementBasis ideal circuits and states"""
        self._test_ideal_basis(PauliMeasurementBasis(), [0, 1])

    def test_pauli_pbasis_1q(self):
        """Test 1-qubit PauliPreparationBasis ideal circuits and states"""
        self._test_ideal_basis(PauliPreparationBasis(), [0])

    def test_pauli_pbasis_2q(self):
        """Test 2-qubit PauliPreparationBasis ideal circuits and states"""
        self._test_ideal_basis(PauliPreparationBasis(), [0, 1])

    def test_pauli6_pbasis_1q(self):
        """Test 1-qubit PauliPreparationBasis ideal circuits and states"""
        self._test_ideal_basis(Pauli6PreparationBasis(), [0])

    def test_pauli6_pbasis_2q(self):
        """Test 2-qubit PauliPreparationBasis ideal circuits and states"""
        self._test_ideal_basis(Pauli6PreparationBasis(), [0, 1])

    def test_local_pbasis_inst(self):
        """Test custom local measurement basis"""
        basis = LocalPreparationBasis("custom_basis", [XGate(), YGate(), HGate()])
        self._test_ideal_basis(basis, [0, 1])

    def test_local_pbasis_unitary(self):
        """Test custom local measurement basis"""
        size = 5
        unitaries = [qi.random_unitary(2, seed=10 + i) for i in range(size)]
        basis = LocalPreparationBasis("unitary_basis", unitaries)
        self._test_ideal_basis(basis, [0, 1])

    def test_local_mbasis_inst(self):
        """Test custom local measurement basis"""
        basis = LocalMeasurementBasis("custom_basis", [XGate(), YGate(), HGate()])
        self._test_ideal_basis(basis, [0, 1])

    def test_local_mbasis_unitary(self):
        """Test custom local measurement basis"""
        size = 5
        unitaries = [qi.random_unitary(2, seed=10 + i) for i in range(size)]
        basis = LocalMeasurementBasis("unitary_basis", unitaries)
        self._test_ideal_basis(basis, [0, 1])
