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

"""
A Tester for the Clifford utilities
"""
from test.base import QiskitExperimentsTestCase

import numpy as np
from ddt import ddt, data
from numpy.random import default_rng

from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.circuit.library import (
    IGate,
    XGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    SdgGate,
    SXGate,
    RZGate,
)
from qiskit.quantum_info import Operator, Clifford, random_clifford
from qiskit_experiments.library.randomized_benchmarking.clifford_utils import (
    CliffordUtils,
    num_from_1q_circuit,
    num_from_2q_circuit,
    compose_1q,
    compose_2q,
    inverse_1q,
    inverse_2q,
    _num_from_layer_indices,
    _layer_indices_from_num,
    _CLIFFORD_LAYER,
    _CLIFFORD_INVERSE_2Q,
    _synthesize_clifford,
)


@ddt
class TestCliffordUtils(QiskitExperimentsTestCase):
    """A test for the Clifford manipulations, including number to and from Clifford mapping"""

    basis_gates = ["rz", "sx", "cx"]
    seed = 123

    def test_clifford_1_qubit_generation(self):
        """Verify 1-qubit clifford indeed generates the correct group"""
        clifford_dicts = [
            {"stabilizer": ["+Z"], "destabilizer": ["+X"]},
            {"stabilizer": ["+X"], "destabilizer": ["+Z"]},
            {"stabilizer": ["+Y"], "destabilizer": ["+X"]},
            {"stabilizer": ["+X"], "destabilizer": ["+Y"]},
            {"stabilizer": ["+Z"], "destabilizer": ["+Y"]},
            {"stabilizer": ["+Y"], "destabilizer": ["+Z"]},
            {"stabilizer": ["-Z"], "destabilizer": ["+X"]},
            {"stabilizer": ["+X"], "destabilizer": ["-Z"]},
            {"stabilizer": ["-Y"], "destabilizer": ["+X"]},
            {"stabilizer": ["+X"], "destabilizer": ["-Y"]},
            {"stabilizer": ["-Z"], "destabilizer": ["-Y"]},
            {"stabilizer": ["-Y"], "destabilizer": ["-Z"]},
            {"stabilizer": ["-Z"], "destabilizer": ["-X"]},
            {"stabilizer": ["-X"], "destabilizer": ["-Z"]},
            {"stabilizer": ["+Y"], "destabilizer": ["-X"]},
            {"stabilizer": ["-X"], "destabilizer": ["+Y"]},
            {"stabilizer": ["-Z"], "destabilizer": ["+Y"]},
            {"stabilizer": ["+Y"], "destabilizer": ["-Z"]},
            {"stabilizer": ["+Z"], "destabilizer": ["-X"]},
            {"stabilizer": ["-X"], "destabilizer": ["+Z"]},
            {"stabilizer": ["-Y"], "destabilizer": ["-X"]},
            {"stabilizer": ["-X"], "destabilizer": ["-Y"]},
            {"stabilizer": ["+Z"], "destabilizer": ["-Y"]},
            {"stabilizer": ["-Y"], "destabilizer": ["+Z"]},
        ]
        cliffords = [Clifford.from_dict(i) for i in clifford_dicts]
        for n in range(24):
            clifford = CliffordUtils.clifford_1_qubit(n)
            self.assertEqual(clifford, cliffords[n])

    def test_number_to_clifford_mapping_single_gate(self):
        """Test that the methods num_from_1q_clifford_single_gate and
        clifford_1_qubit_circuit perform the reverse operations from each other"""
        transpiled_cliff_list = [
            SXGate(),
            RZGate(np.pi),
            RZGate(-np.pi),
            RZGate(np.pi / 2),
            RZGate(-np.pi / 2),
        ]
        general_cliff_list = [
            IGate(),
            HGate(),
            SdgGate(),
            SGate(),
            XGate(),
            SXGate(),
            YGate(),
            ZGate(),
        ]
        for inst in transpiled_cliff_list + general_cliff_list:
            qc_from_inst = QuantumCircuit(1)
            qc_from_inst.append(inst, [0])
            num = num_from_1q_circuit(qc_from_inst)
            qc_from_num = CliffordUtils.clifford_1_qubit_circuit(num)
            self.assertTrue(Operator(qc_from_num).equiv(Operator(qc_from_inst)))

    def test_number_to_clifford_mapping_2q(self):
        """Test if num -> circuit -> num round-trip succeeds for 2Q Cliffords."""
        for i in range(CliffordUtils.NUM_CLIFFORD_2_QUBIT):
            qc = CliffordUtils.clifford_2_qubit_circuit(i)
            num = num_from_2q_circuit(qc)
            self.assertEqual(i, num)

    def test_compose_by_num_1q(self):
        """Compare compose using num and Clifford to compose using two Cliffords, for a single qubit"""
        num_tests = 50
        rng = default_rng(seed=self.seed)
        for _ in range(num_tests):
            num1 = rng.integers(CliffordUtils.NUM_CLIFFORD_1_QUBIT)
            num2 = rng.integers(CliffordUtils.NUM_CLIFFORD_1_QUBIT)
            cliff1 = CliffordUtils.clifford_1_qubit(num1)
            cliff2 = CliffordUtils.clifford_1_qubit(num2)
            clifford_expected = cliff1.compose(cliff2)
            clifford_from_num = CliffordUtils.clifford_1_qubit(compose_1q(num1, num2))
            clifford_from_circuit = Clifford(cliff1.to_circuit().compose(cliff2.to_circuit()))
            self.assertEqual(clifford_expected, clifford_from_num)
            self.assertEqual(clifford_expected, clifford_from_circuit)

    def test_compose_by_num_2q(self):
        """Compare compose using num and Clifford to compose using two Cliffords, for two qubits"""
        num_tests = 100
        rng = default_rng(seed=self.seed)
        for _ in range(num_tests):
            num1 = rng.integers(CliffordUtils.NUM_CLIFFORD_2_QUBIT)
            num2 = rng.integers(CliffordUtils.NUM_CLIFFORD_2_QUBIT)
            cliff1 = CliffordUtils.clifford_2_qubit(num1)
            cliff2 = CliffordUtils.clifford_2_qubit(num2)
            clifford_expected = cliff1.compose(cliff2)
            clifford_from_num = CliffordUtils.clifford_2_qubit(compose_2q(num1, num2))
            clifford_from_circuit = Clifford(cliff1.to_circuit().compose(cliff2.to_circuit()))
            self.assertEqual(clifford_expected, clifford_from_num)
            self.assertEqual(clifford_expected, clifford_from_circuit)

    def test_inverse_by_num_1q(self):
        """Compare inverse using num to inverse using Clifford"""
        num_tests = 24
        for num in range(num_tests):
            cliff = CliffordUtils.clifford_1_qubit(num)
            clifford_expected = cliff.adjoint()
            clifford_from_num = CliffordUtils.clifford_1_qubit(inverse_1q(num))
            clifford_from_circuit = Clifford(cliff.to_circuit().inverse())
            self.assertEqual(clifford_expected, clifford_from_num)
            self.assertEqual(clifford_expected, clifford_from_circuit)

    def test_inverse_by_num_2q(self):
        """Compare inverse using num to inverse using Clifford"""
        num_tests = 100
        rng = default_rng(seed=self.seed)
        for _ in range(num_tests):
            num = rng.integers(CliffordUtils.NUM_CLIFFORD_2_QUBIT)
            cliff = CliffordUtils.clifford_2_qubit(num)
            clifford_expected = cliff.adjoint()
            clifford_from_num = CliffordUtils.clifford_2_qubit(inverse_2q(num))
            clifford_from_circuit = Clifford(cliff.to_circuit().inverse())
            self.assertEqual(clifford_expected, clifford_from_num)
            self.assertEqual(clifford_expected, clifford_from_circuit)

    def test_num_layered_circuit_num_round_trip(self):
        """Test if num -> circuit with layers -> num round-trip succeeds for 2Q Cliffords."""
        for i in range(CliffordUtils.NUM_CLIFFORD_2_QUBIT):
            self.assertEqual(i, compose_2q(0, i))

    def test_mapping_layers_to_num(self):
        """Test the mapping from numbers to layer indices"""
        for i in range(CliffordUtils.NUM_CLIFFORD_2_QUBIT):
            indices = _layer_indices_from_num(i)
            reverse_i = _num_from_layer_indices(indices)
            self.assertEqual(i, reverse_i)

    def test_num_from_layer(self):
        """Check if 2Q clifford from standard/layered circuit has a common integer representation."""
        for i in range(CliffordUtils.NUM_CLIFFORD_2_QUBIT):
            standard = CliffordUtils.clifford_2_qubit(i)
            circ = QuantumCircuit(2)
            for layer, idx in enumerate(_layer_indices_from_num(i)):
                circ.compose(_CLIFFORD_LAYER[layer][idx], inplace=True)
            layered = Clifford(circ)
            self.assertEqual(standard, layered)

    def test_num_from_2q_circuit(self):
        """Check conversion of circuits to integers with num_from_2q_circuit"""
        qc = QuantumCircuit(2)
        qc.h(0)
        num = num_from_2q_circuit(qc)
        self.assertEqual(num, 5760)
        qc = QuantumCircuit(2)
        qc.u(0, 0, np.pi, 0)
        with self.assertRaises(QiskitError):
            # raising an error is ok, num_from_2q_circuit does not support all 2-qubit gates
            num_from_2q_circuit(qc)

        # regression test for using the dense multiplication table
        qc = QuantumCircuit(2)
        qc.cz(1, 0)
        num = num_from_2q_circuit(qc)
        self.assertEqual(num, 368)

    def test_clifford_inverse_table(self):
        """Check correctness of the Clifford inversion table"""
        for lhs, rhs in enumerate(_CLIFFORD_INVERSE_2Q):
            c = compose_2q(lhs, rhs)
            self.assertEqual(c, 0)

    @data(1, 2, 3, 4)
    def test_clifford_synthesis_linear_connectivity(self, num_qubits):
        """Check if clifford synthesis with linear connectivity does not change Clifford"""
        basis_gates = tuple(["rz", "h", "cz"])
        coupling_tuple = (
            None if num_qubits == 1 else tuple((i, i + 1) for i in range(num_qubits - 1))
        )
        for seed in range(10):
            expected = random_clifford(num_qubits=num_qubits, seed=seed)
            circuit = _synthesize_clifford(expected, basis_gates, coupling_tuple)
            synthesized = Clifford(circuit)
            self.assertEqual(expected, synthesized)

    @data(3, 4, 6)
    def test_clifford_synthesis_non_linear_connectivity(self, num_qubits):
        """Check if clifford synthesis with non-linear connectivity does not change Clifford"""
        basis_gates = tuple(["rz", "sx", "cx"])
        # star
        coupling_tuple = tuple((0, i) for i in range(1, num_qubits))
        for seed in range(5):
            expected = random_clifford(num_qubits=num_qubits, seed=seed)
            circuit = _synthesize_clifford(expected, basis_gates, coupling_tuple)
            synthesized = Clifford(circuit)
            self.assertEqual(expected, synthesized)

        # cycle
        coupling_tuple = tuple((i, (i + 1) % num_qubits) for i in range(num_qubits))
        for seed in range(5):
            expected = random_clifford(num_qubits=num_qubits, seed=seed)
            circuit = _synthesize_clifford(expected, basis_gates, coupling_tuple)
            synthesized = Clifford(circuit)
            self.assertEqual(expected, synthesized)
