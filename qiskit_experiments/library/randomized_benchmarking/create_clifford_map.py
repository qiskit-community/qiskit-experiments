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
from qiskit_experiments.library.randomized_benchmarking.clifford_utils import CliffordUtils

def create_compose_map():
    """Creates the data in CLIFF_COMPOSE_DATA and CLIFF_INVERSE_DATA"""
    num_to_cliff = {}
    cliff_to_num = {}

    for i in range(CliffordUtils.NUM_CLIFFORD_1_QUBIT):
        cliff = CliffordUtils.clifford_1_qubit(i)
        num_to_cliff[i] = cliff
        cliff_to_num[cliff.__repr__()] = i

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

    for i in range(CliffordUtils.NUM_CLIFFORD_1_QUBIT):
        cliff1 = num_to_cliff[i]
        # for gate in single_gate_clifford_mapping.keys():
        for gate in single_gate_clifford_mapping:
            cliff2 = num_to_cliff[single_gate_clifford_mapping[gate]]
            cliff = cliff1.compose(cliff2)
            products[i, single_gate_clifford_mapping[gate]] = cliff_to_num[cliff.__repr__()]

    invs = {}
    for i in range(CliffordUtils.NUM_CLIFFORD_1_QUBIT):
        cliff1 = num_to_cliff[i]
        cliff = cliff1.adjoint()
        invs[i] = cliff_to_num[cliff.__repr__()]

    print("CLIFF_COMPOSE_DATA = [")
    for i in products:
        print(f"    {products[i]},")
    print("]")
    print()

    print("CLIFF_INVERSE_DATA = [")
    for i in invs:
        print(f"    {invs[i]},")
    print("]")
    print()

create_compose_map()
