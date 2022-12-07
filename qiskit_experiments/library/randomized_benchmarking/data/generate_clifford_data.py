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
This file is a stand-alone script for generating the npz files in
``~qiskit_experiment.library.randomized_benchmarking.clifford_utils.data`` directory.

The script relies on the values of ``_CLIFF_SINGLE_GATE_MAP_2Q``
in :mod:`~qiskit_experiment.library.randomized_benchmarking.clifford_utils`
so they must be set correctly before running the script.
"""
import itertools

import numpy as np
import scipy.sparse

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
    _CLIFF_SINGLE_GATE_MAP_1Q,
    _CLIFF_SINGLE_GATE_MAP_2Q,
)

NUM_CLIFFORD_1Q = CliffordUtils.NUM_CLIFFORD_1_QUBIT
NUM_CLIFFORD_2Q = CliffordUtils.NUM_CLIFFORD_2_QUBIT


def _hash_cliff(cliff):
    """Produce a hashable value that is unique for each different Clifford.  This should only be
    used internally when the classes being hashed are under our control, because classes of this
    type are mutable."""
    table = cliff.table
    abits = np.packbits(table.array)
    pbits = np.packbits(table.phase)
    return abits.tobytes(), pbits.tobytes()


_TO_CLIFF_1Q = {i: CliffordUtils.clifford_1_qubit(i) for i in range(NUM_CLIFFORD_1Q)}
_TO_INT_1Q = {_hash_cliff(cliff): i for i, cliff in _TO_CLIFF_1Q.items()}


def gen_clifford_inverse_1q():
    """Generate table data for integer 1Q Clifford inversion"""
    invs = np.empty(NUM_CLIFFORD_1Q, dtype=int)
    for i in range(NUM_CLIFFORD_1Q):
        invs[i] = _TO_INT_1Q[_hash_cliff(_TO_CLIFF_1Q[i].adjoint())]
    assert all(sorted(invs) == np.arange(0, NUM_CLIFFORD_1Q))
    return invs


def gen_clifford_compose_1q():
    """Generate table data for integer 1Q Clifford composition."""
    products = np.empty((NUM_CLIFFORD_1Q, NUM_CLIFFORD_1Q), dtype=int)
    for i in range(NUM_CLIFFORD_1Q):
        for j in range(NUM_CLIFFORD_1Q):
            cliff = _TO_CLIFF_1Q[i].compose(_TO_CLIFF_1Q[j])
            products[i, j] = _TO_INT_1Q[_hash_cliff(cliff)]
        assert all(sorted(products[i]) == np.arange(0, NUM_CLIFFORD_1Q))
    return products


_TO_CLIFF_2Q = {i: CliffordUtils.clifford_2_qubit(i) for i in range(NUM_CLIFFORD_2Q)}
_TO_INT_2Q = {_hash_cliff(cliff): i for i, cliff in _TO_CLIFF_2Q.items()}


def gen_clifford_inverse_2q():
    """Generate table data for integer 2Q Clifford inversion"""
    invs = np.empty(NUM_CLIFFORD_2Q, dtype=int)
    for i in range(NUM_CLIFFORD_2Q):
        invs[i] = _TO_INT_2Q[_hash_cliff(_TO_CLIFF_2Q[i].adjoint())]
    assert all(sorted(invs) == np.arange(0, NUM_CLIFFORD_2Q))
    return invs


def gen_clifford_compose_2q_gate():
    """Generate data for a 2Q Clifford composition table.

    Cliffords are represented as integers between 0 and 11519. Note that the full composition table
    would require :math:`11520^2` elements and is therefore *NOT* generated, as that would take
    more than 100MB. Instead, we sparsely populate the composition table only for RHS elements
    from a specific set of basis gates defined by the values of ``_CLIFF_SINGLE_GATE_MAP_2Q``.
    This is sufficient because when composing two arbitrary Cliffords, we can decompose the RHS
    into these basis gates (which needs to be done anyways), and subsequently compute the product
    in multiple steps using this sparse table.
    """
    products = scipy.sparse.lil_matrix((NUM_CLIFFORD_2Q, NUM_CLIFFORD_2Q), dtype=int)
    for lhs in range(NUM_CLIFFORD_2Q):
        for rhs in _CLIFF_SINGLE_GATE_MAP_2Q.values():
            composed = _TO_CLIFF_2Q[lhs].compose(_TO_CLIFF_2Q[rhs])
            products[lhs, rhs] = _TO_INT_2Q[_hash_cliff(composed)]
    return products.tocsr()


_GATE_LIST_1Q = [
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


def gen_cliff_single_1q_gate_map():
    """
    Generates a dict mapping numbers to 1Q Cliffords
    to be used as the value for ``_CLIFF_SINGLE_GATE_MAP_1Q``
    in :mod:`~qiskit_experiment.library.randomized_benchmarking.clifford_utils`.
    Based on it, we build a mapping from every single-gate-clifford to its number.
    The mapping actually looks like {(gate, (0, )): num}.
    """
    table = {}
    for gate in _GATE_LIST_1Q:
        qc = QuantumCircuit(1)
        qc.append(gate, [0])
        num = _TO_INT_1Q[_hash_cliff(Clifford(qc))]
        table[(gate.name, (0,))] = num

    return table


def gen_cliff_single_2q_gate_map():
    """
    Generates a dict mapping numbers to 2Q Cliffords
    to be used as the value for ``_CLIFF_SINGLE_GATE_MAP_2Q``
    in :mod:`~qiskit_experiment.library.randomized_benchmarking.clifford_utils`.
    Based on it, we build a mapping from every single-gate-clifford to its number.
    The mapping actually looks like {(gate, (0, 1)): num}.
    """
    gate_list_2q = [
        CXGate(),
        CZGate(),
    ]
    table = {}
    for gate, qubit in itertools.product(_GATE_LIST_1Q, [0, 1]):
        qc = QuantumCircuit(2)
        qc.append(gate, [qubit])
        num = _TO_INT_2Q[_hash_cliff(Clifford(qc))]
        table[(gate.name, (qubit,))] = num

    for gate, qubits in itertools.product(gate_list_2q, [(0, 1), (1, 0)]):
        qc = QuantumCircuit(2)
        qc.append(gate, qubits)
        num = _TO_INT_2Q[_hash_cliff(Clifford(qc))]
        table[(gate.name, qubits)] = num

    return table


if __name__ == "__main__":
    if _CLIFF_SINGLE_GATE_MAP_1Q != gen_cliff_single_1q_gate_map():
        raise Exception(
            "_CLIFF_SINGLE_GATE_MAP_1Q must be generated by gen_cliff_single_1q_gate_map()"
        )
    np.savez_compressed("clifford_inverse_1q.npz", table=gen_clifford_inverse_1q())
    np.savez_compressed("clifford_compose_1q.npz", table=gen_clifford_compose_1q())

    if _CLIFF_SINGLE_GATE_MAP_2Q != gen_cliff_single_2q_gate_map():
        raise Exception(
            "_CLIFF_SINGLE_GATE_MAP_2Q must be generated by gen_cliff_single_2q_gate_map()"
        )
    np.savez_compressed("clifford_inverse_2q.npz", table=gen_clifford_inverse_2q())
    scipy.sparse.save_npz("clifford_compose_2q_sparse.npz", gen_clifford_compose_2q_gate())
