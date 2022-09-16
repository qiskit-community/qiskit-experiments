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
This is a script used to create the npz files in `data` directory.
"""
import itertools

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import (
    IGate,
    HGate,
    SXdgGate,
    SGate,
    XGate,
    SXGate,
    YGate,
    ZGate,
    SdgGate,
    CXGate,
    CZGate,
)
from qiskit.quantum_info.operators.symplectic import Clifford
from qiskit_experiments.library.randomized_benchmarking.clifford_utils import (
    CliffordUtils,
    NUM_CLIFFORD_2Q,
    CLIFF_SINGLE_GATE_MAP_2Q,
    _hash_cliff,
)
from qiskit_experiments.library.randomized_benchmarking.clifford_utils import _TO_INT as _TO_INT_1Q

_TO_CLIFF = {i: CliffordUtils.clifford_2_qubit(i) for i in range(NUM_CLIFFORD_2Q)}
_TO_INT = {_hash_cliff(cliff): i for i, cliff in _TO_CLIFF.items()}


def create_clifford_inverse_2q():
    """Create table data for integer 2Q Clifford inversion"""
    invs = np.zeros(NUM_CLIFFORD_2Q, dtype=int)
    for i in range(NUM_CLIFFORD_2Q):
        invs[i] = _TO_INT[_hash_cliff(_TO_CLIFF[i].adjoint())]
    return invs


def create_clifford_compose_2q_gate():
    """Create table data for integer 2Q Clifford composition.

    Note that the full compose table of all-Cliffords by all-Cliffords is *NOT* created.
    Instead, only the Cliffords that consist of a single gate defined in `CLIFF_SINGLE_GATE_MAP_2Q`
    are considered for the target Clifford. That means the compose table of
    all-Cliffords by single-gate-cliffords is created. It is sufficient because
    every Clifford on the right hand side can be broken down into single gate Cliffords,
    and do the composition one gate at a time. This greatly reduces the storage space for the array of
    composition results (from O(n^2) to O(n)), where n is the number of Cliffords.
    """
    products = np.zeros((NUM_CLIFFORD_2Q, len(CLIFF_SINGLE_GATE_MAP_2Q)), dtype=int)
    for lhs in range(NUM_CLIFFORD_2Q):
        for gate, (_, rhs) in enumerate(CLIFF_SINGLE_GATE_MAP_2Q.items()):
            composed = _TO_CLIFF[lhs].compose(_TO_CLIFF[rhs])
            products[lhs][gate] = _TO_INT[_hash_cliff(composed)]
    return products


gate_list_1q = [
    IGate(),
    HGate(),
    SXdgGate(),
    SGate(),
    XGate(),
    SXGate(),
    YGate(),
    ZGate(),
    SdgGate(),
]


def create_cliff_single_1q_gate_map():
    """
    Generates a dict mapping numbers to 1Q Cliffords and the reverse dict.
    Based on these structures, we build a mapping from every single-gate-clifford to its number.
    The mapping actually looks like {(gate, (0, )): num}.
    """
    table = {}
    for gate in gate_list_1q:
        qc = QuantumCircuit(1)
        qc.append(gate, [0])
        num = _TO_INT_1Q[_hash_cliff(Clifford(qc))]
        table[(gate.name, (0,))] = num

    return table


def create_cliff_single_2q_gate_map():
    """
    Generates a dict mapping numbers to 2Q Cliffords and the reverse dict.
    Based on these structures, we build a mapping from every single-gate-clifford to its number.
    The mapping actually looks like {(gate, (0, 1)): num}.
    """
    gate_list_2q = [
        CXGate(),
        CZGate(),
    ]
    table = {}
    for gate, qubit in itertools.product(gate_list_1q, [0, 1]):
        qc = QuantumCircuit(2)
        qc.append(gate, [qubit])
        num = _TO_INT[_hash_cliff(Clifford(qc))]
        table[(gate.name, (qubit,))] = num

    for gate, qubits in itertools.product(gate_list_2q, [(0, 1), (1, 0)]):
        qc = QuantumCircuit(2)
        qc.append(gate, qubits)
        num = _TO_INT[_hash_cliff(Clifford(qc))]
        table[(gate.name, qubits)] = num

    return table


if __name__ == "__main__":
    np.savez_compressed("data/clifford_inverse_2q.npz", table=create_clifford_inverse_2q())
    np.savez_compressed(
        "data/clifford_compose_2q_gate.npz", table=create_clifford_compose_2q_gate()
    )
