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
Every Clifford is represented by a number. We store a list of the compositions of Cliffords represented
as numbers. For example, if Clifford1.compose(Clifford2) == Clifford3, then we conceptually,
we store {(1, 2) : 3}. we don't actually store the map, but only the results of the compose in an array.
This is more efficient in performance. The result is found using the indices of the input Cliffords.
Similarly, we store for each number representing a Clifford, the number representing the
inverse Clifford.
For compose, we don't actually store the full compose table of all-cliffords X all-cliffords.
Instead, we define an array of single-gate-cliffords. This comprises all Cliffords that consist
of a single gate. There are 8 such Cliffords for 1-qubit, and 20 such Cliffords for 2-qubits.
It is sufficient to store the compose table of all-cliffords X single-gate-cliffords,
since for every Clifford on the right hand side, we can break it down into single gate Cliffords,
and do the composition one at a time. This greatly reduces the storage space for the array of
composition results (from O(n^2) to O(n)), where n is the number of Cliffords.
"""
import itertools

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
from qiskit_experiments.library.randomized_benchmarking.clifford_utils import CliffordUtils

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
asymm_gates_2q = [
    CXGate(),
]
symm_gates_2q = [CZGate()]


class CliffordNumMapping:
    """Class that creates creates all the structures with the mappings between Cliffords
    and numbers."""

    basis_gates = ["h", "s", "sdg", "x", "cx"]
    single_gate_clifford_map_1q = {}
    single_gate_clifford_map_2q = {}
    num_to_cliff_1q = {}
    cliff_to_num_1q = {}
    num_to_cliff_2q = {}
    cliff_to_num_2q = {}
    layers_num_to_cliff_num_2q = {}
    cliff_num_to_layers_2q = {}

    @classmethod
    def gen_nums_single_gate_cliffs_1q(cls):
        """
        Generates an array mapping numbers to Cliffords and the reverse array.
        Based on this array, we build a mapping from every single-gate-clifford to its number.
        The mapping actually looks like {(gate, '[0]'): num}, where [0] represents qubit 0.
        The qubit is added to be consistent with the format for 2 qubits.
        """
        clifford_utils = CliffordUtils(1, cls.basis_gates)
        for i in range(clifford_utils.NUM_CLIFFORD_1_QUBIT):
            cliff = clifford_utils.clifford_1_qubit(i)
            cls.num_to_cliff_1q[i] = cliff
            cls.cliff_to_num_1q[repr(cliff)] = i

        for gate in gate_list_1q:
            qc = QuantumCircuit(1)
            qc.append(gate, [0])
            cliff = Clifford(qc)
            if repr(cliff) in cls.cliff_to_num_1q.keys():
                num = cls.cliff_to_num_1q[repr(cliff)]
                # qubit_as_str is not really necessary. It is only added to be consistent
                # with the representation for 2 qubits
                qubit_as_str = "[0]"
                cls.single_gate_clifford_map_1q[(gate.name, qubit_as_str)] = num
            else:
                print("not found")
        cliff_data_file.write(f"CLIFF_SINGLE_GATE_MAP_1Q = {cls.single_gate_clifford_map_1q}\n")

    @classmethod
    def gen_nums_single_gate_cliffs_2q(cls):
        """
        Generates an array mapping numbers to Cliffords and the reverse array.
        Based on this array, we build a mapping from every single-gate-clifford to its number.
        The mapping actually looks like {(gate, '[qubits]'): num}.
        """
        clifford_utils = CliffordUtils(2, cls.basis_gates)
        for i in range(clifford_utils.NUM_CLIFFORD_2_QUBIT):
            cliff = clifford_utils.clifford_2_qubit(i)
            cls.num_to_cliff_2q[i] = cliff
            cls.cliff_to_num_2q[repr(cliff)] = i

        for gate, qubit in itertools.product(gate_list_1q, [0, 1]):
            qc = QuantumCircuit(2)
            qc.append(gate, [qubit])
            cliff = Clifford(qc)
            if repr(cliff) in cls.cliff_to_num_2q:
                num = cls.cliff_to_num_2q[repr(cliff)]
                # qubit_as_str is not really necessary. It is only added to be consistent
                # with the representation for 2 qubits
                qubit_as_str = "[" + str(qubit) + "]"
                cls.single_gate_clifford_map_2q[(gate.name, qubit_as_str)] = num
            else:
                print("not found")

        for gate, qubits in itertools.product(asymm_gates_2q, [[0, 1], [1, 0]]):
            qc = QuantumCircuit(2)
            qc.append(gate, qubits)
            cliff = Clifford(qc)
            if repr(cliff) in cls.cliff_to_num_2q.keys():
                num = cls.cliff_to_num_2q[repr(cliff)]
            else:
                print("not found")
            direction = "[0, 1]" if qubits == [0, 1] else "[1, 0]"
            cls.single_gate_clifford_map_2q[(gate.name, direction)] = num

        for gate in symm_gates_2q:
            qc = QuantumCircuit(2)
            qc.append(gate, [0, 1])
            cliff = Clifford(qc)
            if repr(cliff) in cls.cliff_to_num_2q.keys():
                num = cls.cliff_to_num_2q[repr(cliff)]
            else:
                print("not found")
            cls.single_gate_clifford_map_2q[(gate.name, "[0, 1]")] = num

        cliff_data_file.write(f"CLIFF_SINGLE_GATE_MAP_2Q = {cls.single_gate_clifford_map_2q}\n")

    @classmethod
    def create_compose_map_1q(cls):
        """Creates the data in compose data in CLIFF_COMPOSE_DATA and
        the inverse data in CLIFF_INVERSE_DATA"""
        products = {}
        for i in range(CliffordUtils.NUM_CLIFFORD_1_QUBIT):
            cliff1 = cls.num_to_cliff_1q[i]
            for gate in cls.single_gate_clifford_map_1q:
                cliff2 = cls.num_to_cliff_1q[cls.single_gate_clifford_map_1q[gate]]
                cliff = cliff1.compose(cliff2)
                products[(i, gate)] = cls.cliff_to_num_1q[repr(cliff)]

        invs = {}
        for i in range(CliffordUtils.NUM_CLIFFORD_1_QUBIT):
            cliff1 = cls.num_to_cliff_1q[i]
            cliff = cliff1.adjoint()
            invs[i] = cls.cliff_to_num_1q[repr(cliff)]

        cliff_data_file.write("CLIFF_COMPOSE_DATA_1Q = [")
        for i in products:
            cliff_data_file.write(f"{products[i]},")
        cliff_data_file.write("]")
        cliff_data_file.write("\n")

        cliff_data_file.write("CLIFF_INVERSE_DATA_1Q = [")
        for i in invs:
            cliff_data_file.write(f"{invs[i]},")
        cliff_data_file.write("]")
        cliff_data_file.write("\n")

    @classmethod
    def create_compose_map_2q(cls):
        """Creates the data in CLIFF_COMPOSE_DATA and CLIFF_INVERSE_DATA"""
        products = {}
        for i in range(CliffordUtils.NUM_CLIFFORD_2_QUBIT):
            cliff1 = cls.num_to_cliff_2q[i]
            for gate in cls.single_gate_clifford_map_2q:
                cliff2 = cls.num_to_cliff_2q[cls.single_gate_clifford_map_2q[gate]]
                cliff = cliff1.compose(cliff2)
                products[(i, gate)] = cls.cliff_to_num_2q[repr(cliff)]

        invs = {}
        for i in range(CliffordUtils.NUM_CLIFFORD_2_QUBIT):
            cliff1 = cls.num_to_cliff_2q[i]
            cliff = cliff1.adjoint()
            invs[i] = cls.cliff_to_num_2q[repr(cliff)]

        cliff_data_file.write("CLIFF_COMPOSE_DATA_2Q = [")
        for i in products:
            cliff_data_file.write(f"{products[i]},")
        cliff_data_file.write("]")
        cliff_data_file.write("\n")

        cliff_data_file.write("CLIFF_INVERSE_DATA_2Q = [")
        for i in invs:
            cliff_data_file.write(f"{invs[i]},")
        cliff_data_file.write("]")
        cliff_data_file.write("\n")

    @classmethod
    def map_layers_to_cliffords_2q(cls):
        """Creates a map from a triplet describing the indices in the layers, to the
        number of the corresponding Clifford"""
        clifford_utils = CliffordUtils(2, cls.basis_gates)
        clifford_utils.transpile_2q_cliff_layers()
        length = [len(clifford_utils._transpiled_cliff_layer[i]) for i in [0, 1, 2]]
        for n0, n1, n2 in itertools.product(range(length[0]), range(length[1]), range(length[2])):
            cliff = Clifford(clifford_utils.transpiled_cliff_from_layer_nums((n0, n1, n2)))

            num = cls.cliff_to_num_2q[repr(cliff)]
            cls.layers_num_to_cliff_num_2q[(n0, n1, n2)] = num
            cls.cliff_num_to_layers_2q[num] = (n0, n1, n2)

        cliff_data_file.write("CLIFF_LAYERS_TO_NUM_2Q = [")
        for i in cls.layers_num_to_cliff_num_2q:
            cliff_data_file.write(f"{cls.layers_num_to_cliff_num_2q[i]},")
        cliff_data_file.write("]\n")

        cliff_data_file.write("CLIFF_NUM_TO_LAYERS_2Q = [")
        for i in range(len(cls.cliff_num_to_layers_2q)):
            cliff_data_file.write(f"{cls.cliff_num_to_layers_2q[i]},")
        cliff_data_file.write("]\n")

    @classmethod
    def create_clifford_data(cls):
        """Creates all the data for compose and inverse."""
        cls.gen_nums_single_gate_cliffs_1q()
        cls.gen_nums_single_gate_cliffs_2q()
        cls.create_compose_map_1q()
        cls.create_compose_map_2q()


with open("clifford_data.py", "w") as cliff_data_file:
    cliff_data_file.write("# fmt: off\n")
    cliff_data_file.write("# pylint: skip-file\n\n")
    CliffordNumMapping.create_clifford_data()
    CliffordNumMapping.map_layers_to_cliffords_2q()
    cliff_data_file.write("\n# fmt: on\n")
