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
import numpy as np

from qiskit_experiments.library.randomized_benchmarking.clifford_utils import (
    CliffordUtils,
    NUM_CLIFFORD_2Q,
    CLIFF_SINGLE_GATE_MAP_2Q,
    _hash_cliff,
)

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


if __name__ == "__main__":
    np.savez_compressed("data/clifford_inverse_2q.npz", table=create_clifford_inverse_2q())
    np.savez_compressed(
        "data/clifford_compose_2q_gate.npz", table=create_clifford_compose_2q_gate()
    )
