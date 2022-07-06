from typing import List
from qiskit import qpy
from qiskit_experiments.library.randomized_benchmarking import CliffordUtils

basis_gates=["rz", "sx", "cx"]

def generate_1q_transpiled_clifford_circuits(basis_gates: List[str]):
    """Generate all transpiled clifford circuits"""
    transpiled_circs = []
    for num in range(0, CliffordUtils.NUM_CLIFFORD_1_QUBIT):
        circ = CliffordUtils.clifford_1_qubit_circuit(num=num)
        transpiled_circ = CliffordUtils.transpile_single_clifford(circ, basis_gates)
        transpiled_circs.append(transpiled_circ)
    with open('transpiled_circs_1q.qpy', 'wb') as fd:
        qpy.dump(transpiled_circs, fd)

def generate_2q_transpiled_clifford_circuits(basis_gates: List[str]):
    """Generate all transpiled clifford circuits"""
    transpiled_circs = []
    for num in range(0, CliffordUtils.NUM_CLIFFORD_2_QUBIT):
        circ = CliffordUtils.clifford_2_qubit_circuit(num=num)
        transpiled_circ = CliffordUtils.transpile_single_clifford(circ, basis_gates)
        transpiled_circs.append(transpiled_circ)
    with open('transpiled_circs_2q.qpy', 'wb') as fd:
        qpy.dump(transpiled_circs, fd)

generate_1q_transpiled_clifford_circuits(basis_gates=basis_gates)
generate_2q_transpiled_clifford_circuits(basis_gates=basis_gates)