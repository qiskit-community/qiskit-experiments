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


def create_compose_map():
    """Creates the data in CLIFF_COMPOSE_DATA and CLIFF_INVERSE_DATA"""
    num_to_cliff = {}
    cliff_to_num = {}

    #for i in range(CliffordUtils.NUM_CLIFFORD_2_QUBIT):
    for i in range(1000):
        cliff = CliffordUtils.clifford_2_qubit(i)
        num_to_cliff[i] = cliff
        cliff_to_num[cliff.__repr__()] = i
    clifford_single_gate_to_num = {}
    gate_list = [IGate(), HGate(), SXdgGate(), SGate(), XGate(), SXGate(), YGate(), ZGate(),
                      SdgGate()]

    for gate, qubit in itertools.product(gate_list, [0, 1]):
        qc = QuantumCircuit(2)
        qc.append(gate, [qubit])
        cliff = Clifford(qc)
        repr = Clifford(cliff)
        if repr.__repr__() in cliff_to_num.keys():
            num = cliff_to_num[repr.__repr__()]
        else:
            print("not found")
        clifford_single_gate_to_num[(gate.name, qubit)] = num

    for qubits in [[0, 1], [1, 0]]:
        qc = QuantumCircuit(2)
        qc.append(CXGate(), qubits)
        cliff = Clifford(qc)
        repr = Clifford(cliff)
        if repr.__repr__() in cliff_to_num.keys():
            num = cliff_to_num[repr.__repr__()]
        else:
            print("not found")
        direction = "[0, 1]" if qubits ==[0,1] else "[1, 0]"
        clifford_single_gate_to_num[("cx", direction)] = num
    print(clifford_single_gate_to_num)



    products = {}

    single_gate_clifford_mapping = {
        "id": 0,
        "h": 1,
        "sxdg": 2,
        "s": 4,
        "x": 6,
        "sx": 8,
        "y": 12,
        "z": 18,
        "sdg": 22,
    }

#     #for i in range(CliffordUtils.NUM_CLIFFORD_2_QUBIT):
#     for i in range(30):
#         cliff1 = num_to_cliff[i]
#         # for gate in single_gate_clifford_mapping.keys():
#         for gate in single_gate_clifford_mapping:
#             cliff2 = num_to_cliff[single_gate_clifford_mapping[gate]]
#             cliff = cliff1.compose(cliff2)
#             products[i, single_gate_clifford_mapping[gate]] = cliff_to_num[cliff.__repr__()]
#
#     invs = {}
#     #for i in range(CliffordUtils.NUM_CLIFFORD_1_QUBIT):
#     for i in range(30):
#         cliff1 = num_to_cliff[i]
#         cliff = cliff1.adjoint()
#         invs[i] = cliff_to_num[cliff.__repr__()]
#
#     print("CLIFF_COMPOSE_DATA_2Q = [")
#     for i in products:
#         print(f"    {products[i]},")
#     print("]")
#     print()
#
#     print("CLIFF_INVERSE_DATA_2Q = [")
#     for i in invs:
#         print(f"    {invs[i]},")
#     print("]")
#     print()
#
create_compose_map()
