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
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error
import qiskit_experiments as qe


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
    results_file_path = os.path.join(DIRNAME, "rb_output_data1.json")
    analysis_file_path = os.path.join(DIRNAME, "rb_output_analysis1.json")
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
    exp_results = experiment_obj.data()
    with open(results_file_path, "w") as json_file:
        joined_list_data = [exp_attributes]
        joined_list_data.append(exp_results)
        json_file.write(json.dumps(joined_list_data))
    _analysis_save(experiment_obj._analysis_results, analysis_file_path)


def _analysis_save(analysis_data: list, analysis_file_path: str):
    """
    The function is creating a json file from the data of the RB experiment analysis.
    Args:
        analysis_data (list): The data from the analysis of the experiment.
        analysis_file_path (str): The path to save the json file.
    """
    samples_analysis_list = []
    for sample_analysis in analysis_data:
        sample_analysis["popt"] = list(sample_analysis["popt"])
        sample_analysis["popt_err"] = list(sample_analysis["popt_err"])
        sample_analysis["pcov"] = list(sample_analysis["pcov"])
        for idx, item in enumerate(sample_analysis["pcov"]):
            sample_analysis["pcov"][idx] = list(item)
        samples_analysis_list.append(sample_analysis)
    with open(analysis_file_path, "w") as json_file:
        json_file.write(json.dumps(samples_analysis_list))


DIRNAME = os.path.dirname(os.path.abspath(__file__))
for rb_type in sys.argv[1:]:
    if rb_type == "standard":
        generate_rb_fitter_data_1(DIRNAME)
    else:
        print("Skipping unknown argument " + rb_type)
