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
This is a script used to create the data in clifford_data.py.
"""
import itertools

from qiskit.circuit import QuantumCircuit, Gate
from qiskit_experiments.library.randomized_benchmarking.clifford_utils import CliffordUtils
from qiskit.circuit.library import SXGate, CXGate, IGate, HGate, SXdgGate, SGate, \
    XGate, SXGate, YGate, ZGate, SdgGate, CXGate
from qiskit.quantum_info.operators.symplectic import Clifford

gate_list_1q = [
    IGate(),
    HGate(),
    SXdgGate(),
    SGate(),
    XGate(),
    SXGate(),
    YGate(),
    ZGate(),
    SdgGate()
]
num_to_cliff_1q = {}
cliff_to_num_1q = {}
num_to_cliff_2q = {}
cliff_to_num_2q = {}

def generate_nums_for_single_gate_cliffs_1q():
    # We generate an array mapping numbers to Cliffords and the reverse array.
    for i in range(24):
        cliff = CliffordUtils.clifford_1_qubit(i)
        num_to_cliff_1q[i] = cliff
        cliff_to_num_1q[cliff.__repr__()] = i
    clifford_single_gate_to_num = {}

    for gate in gate_list_1q:
        qc = QuantumCircuit(1)
        qc.append(gate, [0])
        cliff = Clifford(qc)
        repr = Clifford(cliff)
        if repr.__repr__() in cliff_to_num_1q.keys():
            num = cliff_to_num_1q[repr.__repr__()]
            clifford_single_gate_to_num[gate.name] = num
        else:
            print("not found")
    file.write(f"CLIFF_SINGLE_GATE_MAP_1Q = {clifford_single_gate_to_num}\n")
    return clifford_single_gate_to_num

def generate_nums_for_single_gate_cliffs_2q():
    # We generate an array mapping numbers to Cliffords and the reverse array.
    for i in range(CliffordUtils.NUM_CLIFFORD_2_QUBIT):
        cliff = CliffordUtils.clifford_2_qubit(i)
        num_to_cliff_2q[i] = cliff
        cliff_to_num_2q[cliff.__repr__()] = i

    clifford_single_gate_to_num = {}
    for gate, qubit in itertools.product(gate_list_1q, [0, 1]):
        qc = QuantumCircuit(2)
        qc.append(gate, [qubit])
        cliff = Clifford(qc)
        repr = Clifford(cliff)
        if repr.__repr__() in cliff_to_num_2q:
            num = cliff_to_num_2q[repr.__repr__()]
            clifford_single_gate_to_num[(gate.name, qubit)] = num
        else:
            print("not found")

    for qubits in [[0, 1], [1, 0]]:
        qc = QuantumCircuit(2)
        qc.append(CXGate(), qubits)
        cliff = Clifford(qc)
        repr = Clifford(cliff)
        if repr.__repr__() in cliff_to_num_2q.keys():
            num = cliff_to_num_2q[repr.__repr__()]
        else:
            print("not found")
        direction = "[0, 1]" if qubits == [0,1] else "[1, 0]"
        clifford_single_gate_to_num[("cx", direction)] = num
    file.write(f"CLIFF_SINGLE_GATE_MAP_2Q = {clifford_single_gate_to_num}\n")
    return clifford_single_gate_to_num

def create_compose_map_1q():
    """Creates the data in CLIFF_COMPOSE_DATA and CLIFF_INVERSE_DATA"""
    products = {}
    for i in range(CliffordUtils.NUM_CLIFFORD_1_QUBIT):
        cliff1 = num_to_cliff_1q[i]
        for gate in single_gate_clifford_map_1q:
            cliff2 = num_to_cliff_1q[single_gate_clifford_map_1q[gate]]
            cliff = cliff1.compose(cliff2)
            #products[(i, single_gate_clifford_map_1q[gate])] = cliff_to_num_1q[cliff.__repr__()]
            products[(i, gate)] = cliff_to_num_1q[cliff.__repr__()]

    invs = {}
    for i in range(CliffordUtils.NUM_CLIFFORD_1_QUBIT):
        cliff1 = num_to_cliff_1q[i]
        cliff = cliff1.adjoint()
        invs[i] = cliff_to_num_1q[cliff.__repr__()]

    file.write("CLIFF_COMPOSE_DATA_1Q = [")
    for i in products:
        file.write(f"{products[i]},")
    file.write("]")
    file.write("\n")

    file.write("CLIFF_INVERSE_DATA_1Q = [")
    for i in invs:
        file.write(f"{invs[i]},")
    file.write("]")
    file.write("\n")

def create_compose_map_2q():
    """Creates the data in CLIFF_COMPOSE_DATA and CLIFF_INVERSE_DATA"""
    products = {}
    for i in range(CliffordUtils.NUM_CLIFFORD_2_QUBIT):
        cliff1 = num_to_cliff_2q[i]
        for gate in single_gate_clifford_map_2q:
            cliff2 = num_to_cliff_2q[single_gate_clifford_map_2q[gate]]
            cliff = cliff1.compose(cliff2)
            products[(i, gate)] = cliff_to_num_2q[cliff.__repr__()]

    invs = {}
    for i in range(CliffordUtils.NUM_CLIFFORD_2_QUBIT):
        cliff1 = num_to_cliff_2q[i]
        cliff = cliff1.adjoint()
        invs[i] = cliff_to_num_2q[cliff.__repr__()]

    file.write("CLIFF_COMPOSE_DATA_2Q = [")
    for i in products:
        file.write(f"{products[i]},")
    file.write("]")
    file.write("\n")

    file.write("CLIFF_INVERSE_DATA_2Q = [")
    for i in invs:
        file.write(f"{invs[i]},")
    file.write("]")
    file.write("\n")

with open("clifford_data.py", "w") as file:
    single_gate_clifford_map_1q = generate_nums_for_single_gate_cliffs_1q()
    single_gate_clifford_map_2q = generate_nums_for_single_gate_cliffs_2q()
    create_compose_map_1q()
    create_compose_map_2q()
