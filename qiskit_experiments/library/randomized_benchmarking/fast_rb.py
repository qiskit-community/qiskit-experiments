
from qiskit_experiments.library.randomized_benchmarking.clifford_utils import CliffordUtils
from qiskit.providers.aer import AerSimulator
from qiskit.compiler import transpile
from .cliff_data import CLIFF_COMPOSE_DATA, CLIFF_INVERSE_DATA

import time

def build_rb_circuits(lengths, circuits, rng):
    start = time.time()
    all_clifford_circuits = []
    rand = rng.integers(0, 23)
    # choose random clifford for first element
    circ = circuits[rand].copy()
    circ.barrier(0)
    clifford_as_num = rand

    if lengths[0] == 1:
        rb_circ = circ.copy()
        inverse_num = CLIFF_INVERSE_DATA[rand]
        inverse_circ = circuits[inverse_num]
        rb_circ.compose(inverse_circ, inplace=True)
        rb_circ.measure_all()
        rb_circ.metadata = {
            "experiment_type": "rb",
            "xval": 2,
            "group": "Clifford",
            "physical_qubits": 0,
        }

    prev_length = 2
    for length in lengths:
        for i in range(prev_length, length+1):
            rand = rng.integers(0, 23)
            # choose random clifford
            next_circ = circuits[rand]
            circ.compose(next_circ,  inplace=True)
            circ.barrier(0)
            clifford_as_num = CLIFF_COMPOSE_DATA[(clifford_as_num, rand)]
            if i==length:
                rb_circ = circ.copy()
                inverse_clifford_num = CLIFF_INVERSE_DATA[clifford_as_num]
                # append the inverse
                rb_circ.compose(circuits[inverse_clifford_num],  inplace=True)
                rb_circ.measure_all()

                rb_circ.metadata = {
                    "experiment_type": "rb",
                    "xval": length + 1,
                    "group": "Clifford",
                    "physical_qubits": 0,
                }

            prev_length = i+1
        all_clifford_circuits.append(rb_circ)
        #print(rb_circ)
    end = time.time()
    print(" time for build_rb_circuits = " + str(end-start))
    return all_clifford_circuits

def generate_all_transpiled_clifford_circuits():
    utils = CliffordUtils()
    circs = []
    for num in range(0, 24):
        circ = utils.clifford_1_qubit_circuit(num=num)
        circs.append(circ)

    backend = AerSimulator()
    new_circs = []

    for i, circ in enumerate(circs):
        transpiled_circ = transpile(circ, backend, optimization_level=1, basis_gates=['sx','rz'])
        new_circ = transpiled_circ.copy() # do we need the copy?
        new_circs.append(new_circ)
        #print(i)
        #print(new_circ)
    return new_circs



