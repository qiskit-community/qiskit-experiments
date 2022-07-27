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
This file generates all Cliffords after transpilation. The user can select
the basis set to use for transpilation. The Cliffords are stored in a .qpy file.
"""
from typing import List
from qiskit import qpy
from qiskit.compiler import transpile
from qiskit_experiments.library.randomized_benchmarking import CliffordUtils

# basis_gates_to_generate = ["rz", "sx", "cx"]
basis_gates_to_generate = ["x", "h", "s", "cx"]


def generate_1q_transpiled_cliffs(basis_gates: List[str]):
    """Generate all 1-qubit transpiled clifford circuits"""
    transpiled_circs = []
    for num in range(0, CliffordUtils.NUM_CLIFFORD_1_QUBIT):
        circ = CliffordUtils.clifford_1_qubit_circuit(num=num)
        transpiled_circ = transpile(circuits=circ, optimization_level=1, basis_gates=basis_gates)
        transpiled_circs.append(transpiled_circ)
    file_name = CliffordUtils.file_name(num_qubits=1, basis_gates=basis_gates)
    with open(file_name, "wb") as fd:
        qpy.dump(transpiled_circs, fd)


def generate_2q_transpiled_cliffs(basis_gates: List[str]):
    """Generate all 2-qubit transpiled clifford circuits"""
    transpiled_circs = []
    for num in range(0, CliffordUtils.NUM_CLIFFORD_2_QUBIT):
        circ = CliffordUtils.clifford_2_qubit_circuit(num=num)
        transpiled_circ = transpile(circuits=circ, optimization_level=1, basis_gates=basis_gates)
        transpiled_circs.append(transpiled_circ)
    file_name = CliffordUtils.file_name(num_qubits=2, basis_gates=basis_gates)
    with open(file_name, "wb") as fd:
        qpy.dump(transpiled_circs, fd)


generate_1q_transpiled_cliffs(basis_gates=basis_gates_to_generate)
generate_2q_transpiled_cliffs(basis_gates=basis_gates_to_generate)
