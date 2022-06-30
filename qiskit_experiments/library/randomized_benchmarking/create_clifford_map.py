# Written by Sasha
from qiskit_experiments.library.randomized_benchmarking.clifford_utils import CliffordUtils

def create_compose_map():
    utils = CliffordUtils()

    num_to_cliff = {}
    cliff_to_num = {}

    for i in range(24):
        cliff = utils.clifford_1_qubit(i)
        num_to_cliff[i] = cliff
        cliff_to_num[cliff.__repr__()] = i

    products = {}

    single_gate_clifford_mapping = {"id":0, "h":1, "sxdg":2, "s":4, "x":6, "sx":8, "y":12, "z":18, "sdg":22}
    for i in range(24):
        cliff1 = num_to_cliff[i]
        for gate in single_gate_clifford_mapping.keys():
            cliff2 = num_to_cliff[single_gate_clifford_mapping[gate]]
            cliff = cliff1.compose(cliff2)
            products[i, single_gate_clifford_mapping[gate]] = cliff_to_num[cliff.__repr__()]

    invs = {}
    for i in range(24):
        cliff1 = num_to_cliff[i]
        cliff = cliff1.adjoint()
        invs[i] = cliff_to_num[cliff.__repr__()]

    print("CLIFF_COMPOSE_DATA = {")
    for i in products:
        print(f" {i}:{products[i]},")
    print(" }")
    print()

    print("CLIFF_INVERSE_DATA = [")
    for i in invs:
        print(f" {invs[i]},")
    print(" ]")
    print()

create_compose_map()
