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

from cmath import isclose
import itertools as it
from test.base import QiskitExperimentsTestCase
import numpy as np
import qiskit.quantum_info as qi
from qiskit.circuit.library import HGate, XGate, SXGate
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

    def _test_ideal_basis(self, basis, qubits):
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
                self.assertTrue(
                    isclose(expval, 1.0), msg=f"{basis.name}, index={index}, {expval} != 1"
                )

        elif isinstance(basis, MeasurementBasis):
            outcome_shape = basis.outcome_shape(qubits)
            outcomes = list(it.product(*[range(i) for i in outcome_shape]))
            for index in indices:
                circ = basis.circuit(index, qubits)
                circ.remove_final_measurements()
                adjoint = circ.inverse()
                for outcome_tup in outcomes:
                    outcome = self._outcome_tup_to_int(outcome_tup)
                    state = qi.Statevector.from_int(outcome, dims=2**circ.num_qubits)
                    state = state.evolve(adjoint)
                    mat = basis.matrix(index, outcome, qubits)
                    expval = state.expectation_value(mat)
                    self.assertTrue(
                        isclose(expval, 1.0),
                        msg=f"{basis.name}, index={index}, outcome={outcome}, {expval} != 1",
                    )

    def _outcome_tup_to_int(self, outcome):
        return int("".join((str(i) for i in reversed(outcome))), 2)

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
        basis = LocalPreparationBasis("custom_basis", [XGate(), SXGate(), HGate()])
        self._test_ideal_basis(basis, [0, 1])

    def test_local_pbasis_unitary(self):
        """Test custom local measurement basis"""
        size = 5
        unitaries = [qi.random_unitary(2, seed=10 + i) for i in range(size)]
        basis = LocalPreparationBasis("unitary_basis", unitaries)
        self._test_ideal_basis(basis, [0, 1])

    def test_local_mbasis_inst(self):
        """Test custom local measurement basis"""
        basis = LocalMeasurementBasis("custom_basis", [XGate(), SXGate(), HGate()])
        self._test_ideal_basis(basis, [0, 1])

    def test_local_mbasis_unitary(self):
        """Test custom local measurement basis"""
        size = 5
        unitaries = [qi.random_unitary(2, seed=10 + i) for i in range(size)]
        basis = LocalMeasurementBasis("unitary_basis", unitaries)
        self._test_ideal_basis(basis, [0, 1])

    def test_local_pbasis_no_inst(self):
        """Test circuits method raises if no instructions"""
        default_states = [qi.random_statevector(2, seed=30 + i) for i in range(2)]
        basis = LocalPreparationBasis("fitter_basis", default_states=default_states)
        with self.assertRaises(NotImplementedError):
            basis.circuit([0], [0])

    def test_local_pbasis_default_statevector(self):
        """Test default states kwarg"""
        default_states = [qi.random_statevector(2, seed=30 + i) for i in range(3)]
        basis = LocalPreparationBasis("fitter_basis", default_states=default_states)
        for i, state in enumerate(default_states):
            basis_state = qi.DensityMatrix(basis.matrix([i], [0]))
            fid = qi.state_fidelity(state, basis_state)
            self.assertTrue(isclose(fid, 1), msg=f"Incorrect state matrix ({i}, F = {fid})")

    def test_local_pbasis_default_densitymatrix(self):
        """Test default states kwarg"""
        default_states = [qi.random_density_matrix(2, seed=30 + i) for i in range(3)]
        basis = LocalPreparationBasis("fitter_basis", default_states=default_states)
        for i, state in enumerate(default_states):
            basis_state = qi.DensityMatrix(basis.matrix([i], [0]))
            fid = qi.state_fidelity(state, basis_state)
            self.assertTrue(isclose(fid, 1), msg=f"Incorrect state matrix ({i}, F = {fid})")

    def test_local_pbasis_qubit_states_no_default(self):
        """Test matrix method raises for invalid qubit with no default states"""
        size = 2
        qubits = [0, 2]
        qubit_states = {
            qubits[0]: [qi.random_density_matrix(2, seed=30 + i) for i in range(size)],
            qubits[1]: [qi.random_statevector(2, seed=40 + i) for i in range(size)],
        }
        basis = LocalPreparationBasis("fitter_basis", qubit_states=qubit_states)
        # No default states so should raise an exception
        with self.assertRaises(ValueError):
            basis.matrix([0, 0], [0, 1])

    def test_local_pbasis_qubit_states(self):
        """Test qubit states kwarg"""
        size = 3
        qubits = [0, 2]
        qubit_states = {
            qubits[0]: [qi.random_density_matrix(2, seed=30 + i) for i in range(size)],
            qubits[1]: [qi.random_statevector(2, seed=40 + i) for i in range(size)],
        }
        basis = LocalPreparationBasis("fitter_basis", qubit_states=qubit_states)

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
        """Test qubit states kwarg"""
        size = 3
        qubits = [2, 0]
        default_states = [qi.random_density_matrix(2, seed=20 + i) for i in range(size)]
        qubit_states = {2: [qi.random_statevector(2, seed=40 + i) for i in range(size)]}
        basis = LocalPreparationBasis(
            "fitter_basis", default_states=default_states, qubit_states=qubit_states
        )

        # Check states
        indices = it.product(range(size), repeat=2)
        states0 = qubit_states[qubits[0]] if qubits[0] in qubit_states else default_states
        states1 = qubit_states[qubits[1]] if qubits[1] in qubit_states else default_states
        for index in indices:
            basis_state = qi.DensityMatrix(basis.matrix(index, qubits))
            target = qi.DensityMatrix(states0[index[0]]).expand(states1[index[1]])
            fid = qi.state_fidelity(basis_state, target)
            self.assertTrue(isclose(fid, 1))

    def test_local_pbasis_qubit_states_chan(self):
        """Test noisy preparation basis construction from qubit states"""
        instructions = PauliPreparationBasis()._instructions
        num_qubits = 3
        p_err = 0.1
        err_chans = [
            (1 - p_err) * qi.SuperOp(np.eye(4)) + p_err * qi.random_quantum_channel(2, seed=100 + i)
            for i in range(num_qubits)
        ]
        qubit_states_int = {i: [chan] for i, chan in enumerate(err_chans)}
        qubit_states_tup = {(i,): [chan] for i, chan in enumerate(err_chans)}

        for qubit_states in [qubit_states_int, qubit_states_tup]:
            pbasis = LocalPreparationBasis(
                "NoisyPauliPrep",
                instructions=instructions,
                qubit_states=qubit_states,
            )

            for qubit, chan in enumerate(err_chans):
                prep0 = qi.DensityMatrix.from_label("0").evolve(chan)
                for index, inst in enumerate(instructions):
                    value = pbasis.matrix([index], [qubit])
                    target = prep0.evolve(inst)
                    infid = 1 - qi.state_fidelity(value, target)
                    self.assertLess(infid, 1e-7)

    def test_local_mbasis_no_inst(self):
        """Test circuits method raises if no instructions"""
        default_povms = [qi.random_unitary(2, seed=30 + i) for i in range(2)]
        basis = LocalMeasurementBasis("fitter_basis", default_povms=default_povms)
        with self.assertRaises(NotImplementedError):
            basis.circuit([0], [0])

    def test_local_mbasis_default_unitary(self):
        """Test default povms kwarg"""
        default_povms = [qi.random_unitary(2, seed=30 + i) for i in range(3)]
        basis = LocalMeasurementBasis("fitter_basis", default_povms=default_povms)
        for i, povm in enumerate(default_povms):
            adjoint = povm.adjoint()
            for outcome in range(2):
                state = qi.Statevector.from_int(outcome, dims=2**adjoint.num_qubits)
                state = state.evolve(adjoint)
                basis_state = qi.DensityMatrix(basis.matrix([i], outcome, [0]))
                fid = qi.state_fidelity(state, basis_state)
                self.assertTrue(isclose(fid, 1))

    def test_local_mbasis_default_statevector(self):
        """Test default povms kwarg"""
        size = 2
        outcomes = 3
        default_povms = [
            [qi.random_statevector(2, seed=30 + i + j) for j in range(outcomes)]
            for i in range(size)
        ]
        basis = LocalMeasurementBasis("fitter_basis", default_povms=default_povms)
        for i, povm in enumerate(default_povms):
            for outcome, effect in enumerate(povm):
                basis_state = qi.DensityMatrix(basis.matrix([i], outcome, [0]))
                fid = qi.state_fidelity(effect, basis_state)
                self.assertTrue(isclose(fid, 1))

    def test_local_mbasis_qubit_povm_no_default(self):
        """Test matrix method raises for invalid qubit with no default states"""
        size = 2
        qubits = [0, 2]
        qubit_povms = {
            qubits[0]: [qi.random_unitary(2, seed=30 + i) for i in range(size)],
            qubits[1]: [qi.random_unitary(2, seed=40 + i) for i in range(size)],
        }
        basis = LocalMeasurementBasis("fitter_basis", qubit_povms=qubit_povms)
        # No default states so should raise an exception
        with self.assertRaises(ValueError):
            basis.matrix([0, 0], 0, [0, 1])

    def test_local_mbasis_qubit_povms(self):
        """Test qubit povms kwarg"""
        size = 2
        outcomes = 2
        qubits = [0, 2]
        qubit_povms = {
            qubits[0]: [
                [qi.random_density_matrix(2, seed=30 + i + j) for i in range(outcomes)]
                for j in range(size)
            ],
            qubits[1]: [
                [qi.random_density_matrix(2, seed=40 + i + j) for i in range(outcomes)]
                for j in range(size)
            ],
        }
        basis = LocalMeasurementBasis("fitter_basis", qubit_povms=qubit_povms)

        # Check states
        indices = it.product(range(size), repeat=len(qubits))
        outcomes = it.product(range(outcomes), repeat=len(qubits))
        for index in indices:
            for outcome in outcomes:
                basis_state = qi.DensityMatrix(
                    basis.matrix(index, self._outcome_tup_to_int(outcome), qubits)
                )
                target0 = qi.DensityMatrix(qubit_povms[qubits[0]][index[0]][outcome[0]])
                target1 = qi.DensityMatrix(qubit_povms[qubits[1]][index[1]][outcome[1]])
                target = target0.expand(target1)
                fid = qi.state_fidelity(basis_state, target)
                self.assertTrue(isclose(fid, 1))

    def test_local_mbasis_default_and_qubit_povm(self):
        """Test qubit and default povm kwarg"""
        size = 3
        outcomes = 2
        qubits = [2, 0]
        default_povms = (
            [
                [qi.random_statevector(2, seed=20 + i + j) for i in range(outcomes)]
                for j in range(size)
            ],
        )
        qubit_povms = {
            qubits[0]: [
                [qi.random_density_matrix(2, seed=30 + i + j) for i in range(outcomes)]
                for j in range(size)
            ],
            qubits[1]: [
                [qi.random_density_matrix(2, seed=40 + i + j) for i in range(outcomes)]
                for j in range(size)
            ],
        }
        basis = LocalMeasurementBasis(
            "fitter_basis", default_povms=default_povms, qubit_povms=qubit_povms
        )

        # Check states
        states0 = qubit_povms[qubits[0]] if qubits[0] in qubit_povms else default_povms
        states1 = qubit_povms[qubits[1]] if qubits[1] in qubit_povms else default_povms
        indices = it.product(range(size), repeat=2)
        outcomes = it.product(range(outcomes), repeat=len(qubits))
        for index in indices:
            for outcome in outcomes:
                basis_state = qi.DensityMatrix(
                    basis.matrix(index, self._outcome_tup_to_int(outcome), qubits)
                )
                target0 = qi.DensityMatrix(states0[index[0]][outcome[0]])
                target1 = qi.DensityMatrix(states1[index[1]][outcome[1]])
                target = target0.expand(target1)
                fid = qi.state_fidelity(basis_state, target)
                self.assertTrue(isclose(fid, 1))

    def test_local_mbasis_qubit_povm_chan(self):
        """Test noisy preparation basis construction from qubit states"""
        instructions = PauliMeasurementBasis()._instructions
        num_qubits = 3
        p_err = 0.1
        err_chans = [
            (1 - p_err) * qi.SuperOp(np.eye(4)) + p_err * qi.random_quantum_channel(2, seed=100 + i)
            for i in range(num_qubits)
        ]
        qubit_povm_int = {i: [chan] for i, chan in enumerate(err_chans)}
        qubit_povm_tup = {(i,): [chan] for i, chan in enumerate(err_chans)}

        for qubit_povms in [qubit_povm_int, qubit_povm_tup]:
            mbasis = LocalMeasurementBasis(
                "NoisyMeas",
                instructions=instructions,
                qubit_povms=qubit_povms,
            )

            for qubit, chan in enumerate(err_chans):
                outcome0 = qi.DensityMatrix.from_label("0").evolve(chan.adjoint())
                outcome1 = qi.DensityMatrix.from_label("1").evolve(chan.adjoint())
                for index, inst in enumerate(instructions):
                    for outcome, povm in enumerate([outcome0, outcome1]):
                        value = mbasis.matrix([index], outcome, [qubit])
                        target = povm.evolve(inst.inverse())
                        norm = np.linalg.norm(target.data - value)
                        self.assertLess(norm, 1e-10)
