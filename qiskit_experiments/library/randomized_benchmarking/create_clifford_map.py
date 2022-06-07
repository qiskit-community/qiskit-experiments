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

    prods = {}

    for i in range(24):
        cliff1 = num_to_cliff[i]
        for j in range(24):
            cliff2 = num_to_cliff[j]
            cliff = cliff1.compose(cliff2)
            prods[i, j] = cliff_to_num[cliff.__repr__()]

    invs = {}
    for i in range(24):
        cliff1 = num_to_cliff[i]
        cliff = cliff1.adjoint()
        invs[i] = cliff_to_num[cliff.__repr__()]

    print("CLIFF_COMPOSE_DATA = {")
    for i in prods:
        print(f" {i}: {prods[i]},")
    #print(prods)
    print(" }")

    print("CLIFF_INVERSE_DATA = {")
    #print(invs)
    for i in invs:
        print(f" {i}: {invs[i]},")
    print(" }")

create_compose_map()
