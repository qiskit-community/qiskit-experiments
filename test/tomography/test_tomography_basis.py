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

    def test_local_pbasis_default_statevector(self):
        """Test default states kwarg"""
        default_states = [qi.random_statevector(2, seed=30 + i) for i in range(3)]
        basis = LocalPreparationBasis("fitter_pbasis", default_states=default_states)
        with self.assertRaises(NotImplementedError):
            basis.circuit([0], [0])
        for i, state in enumerate(default_states):
            basis_state = qi.DensityMatrix(basis.matrix([i], [0]))
            fid = qi.state_fidelity(state, basis_state)
            self.assertTrue(isclose(fid, 1))

    def test_local_pbasis_default_densitymatrix(self):
        """Test default states kwarg"""
        default_states = [qi.random_density_matrix(2, seed=30 + i) for i in range(3)]
        basis = LocalPreparationBasis("fitter_pbasis", default_states=default_states)
        with self.assertRaises(NotImplementedError):
            basis.circuit([0], [0])
        for i, state in enumerate(default_states):
            basis_state = qi.DensityMatrix(basis.matrix([i], [0]))
            fid = qi.state_fidelity(state, basis_state)
            self.assertTrue(isclose(fid, 1))

    def test_local_pbasis_qubit_states(self):
        """Test default states kwarg"""
        size = 3
        qubits = [0, 2]
        qubit_states = {
            qubits[0]: [qi.random_density_matrix(2, seed=30 + i) for i in range(size)],
            qubits[1]: [qi.random_statevector(2, seed=40 + i) for i in range(size)],
        }
        basis = LocalPreparationBasis("fitter_pbasis", qubit_states=qubit_states)

        # No instructions so should raise an exception
        with self.assertRaises(NotImplementedError):
            basis.circuit([0], [0])

        # No default states so should raise an exception
        with self.assertRaises(ValueError):
            basis.matrix([0, 0], [0, 1])

        # Check states
        indices = it.product(range(size), repeat=2)
        for index in indices:
            basis_state = qi.DensityMatrix(basis.matrix(index, qubits))
            target0 = qi.DensityMatrix(qubit_states[qubits[0]][index[0]])
            target1 = qi.DensityMatrix(qubit_states[qubits[1]][index[1]])
            target = target0.expand(target1)
            fid = qi.state_fidelity(basis_state, target)
            self.assertTrue(isclose(fid, 1))

    def test_local_pbasis_default_and_qubit_states(self):
        """Test default states kwarg"""
        size = 3
        qubits = [2, 0]
        default_states = [qi.random_density_matrix(2, seed=20 + i) for i in range(size)]
        qubit_states = {2: [qi.random_statevector(2, seed=40 + i) for i in range(size)]}
        basis = LocalPreparationBasis(
            "fitter_pbasis", default_states=default_states, qubit_states=qubit_states
        )
        # No instructions so should raise an exception
        with self.assertRaises(NotImplementedError):
            basis.circuit([0], [0])

        # Check states
        indices = it.product(range(size), repeat=2)
        states0 = qubit_states[qubits[0]] if qubits[0] in qubit_states else default_states
        states1 = qubit_states[qubits[1]] if qubits[1] in qubit_states else default_states
        for index in indices:
            basis_state = qi.DensityMatrix(basis.matrix(index, qubits))
            target = qi.DensityMatrix(states0[index[0]]).expand(states1[index[1]])
            fid = qi.state_fidelity(basis_state, target)
            self.assertTrue(isclose(fid, 1))
