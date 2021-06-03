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
Code for generating data for the RB experiment for testing data analysis.
"""

import os
import sys
import json
import qiskit_experiments as qe
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error


def create_depolarizing_noise_model():
    """
    create noise model of depolarizing error
    Returns:
        NoiseModel: depolarizing error noise model
    """
    noise_model = NoiseModel()
    p1q = 0.002
    p2q = 0.01
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), "x")
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), "sx")
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), "rz ")
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p2q, 2), "cx")
    return noise_model


def generate_rb_fitter_data_1(results_file_path: str):
    """
    Executing standard RB experiment and storing its data in json format.
    The json is composed of a list that the first element is a dictionary containing
    the experiment attributes and the second element is a list with all the experiment
    data.
    Args:
        results_file_path(str): The json file path that the program write the data to.
    """
    exp_attributes = {
        "qubits": [0, 1],
        "lengths": list(range(1, 200, 20)),
        "num_samples": 2,
        "seed": 100,
    }
    noise_model = create_depolarizing_noise_model()
    backend = QasmSimulator()
    rb = qe.randomized_benchmarking
    rb_exp = rb.RBExperiment(
        exp_attributes["qubits"],
        exp_attributes["lengths"],
        num_samples=exp_attributes["num_samples"],
        seed=exp_attributes["seed"],
    )
    experiment_obj = rb_exp.run(backend, noise_model=noise_model)
    exp_results = experiment_obj._data
    with open(results_file_path, "w") as json_file:
        joined_list_data = [exp_attributes]
        joined_list_data.append(exp_results)
        json_file.write(json.dumps(joined_list_data))


DIRNAME = os.path.dirname(os.path.abspath(__file__))
for rb_type in sys.argv[1:]:
    if rb_type == "standard":
        generate_rb_fitter_data_1(os.path.join(DIRNAME, "rb_output_data1.json"))
    else:
        print("Skipping unknown argument " + rb_type)
