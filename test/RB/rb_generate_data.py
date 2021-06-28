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
    # the error parameters were took from ibmq_manila on 17 june 2021
    p1q = 0.002257
    p2q = 0.006827
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), "x")
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), "sx")
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p2q, 2), "cx")
    return noise_model


def standard_rb_exp_data_gen(dir_name: str):
    """
    Encapsulation for different experiments attributes which in turn execute.
    The data and analysis is saved to json file via "_generate_rb_fitter_data" function.
    Args:
        dir_name(str): The directory which the program will save the data and anaysis.
    """
    rb_exp_name = ["rb_standard_1qubit", "rb_standard_2qubits"]
    experiments_attributes = [
        {
            "physical_qubits": [0],
            "lengths": list(range(1, 200, 20)),
            "num_samples": 3,
            "seed": 100,
        },
        {
            "physical_qubits": [0, 1],
            "lengths": list(range(1, 200, 20)),
            "num_samples": 3,
            "seed": 100,
        },
    ]
    for idx, experiment_attributes in enumerate(experiments_attributes):
        _generate_rb_fitter_data(dir_name, rb_exp_name[idx], experiment_attributes)


def _generate_rb_fitter_data(dir_name: str, rb_exp_name: str, exp_attributes: dict):
    """
    Executing standard RB experiment and storing its data in json format.
    The json is composed of a list that the first element is a dictionary containing
    the experiment attributes and the second element is a list with all the experiment
    data.
    Args:
        dir_name: The json file name that the program write the data to.
        rb_exp_name: The experiment name for naming the output files.
        exp_attributes: attributes to config the RB experiment.
    """
    gate_error_ratio = {((0,), "id"): 1, ((0,), "rz"): 0, ((0,), "sx"): 1, ((0,), "x"): 1, ((0, 1), "cx"): 1}
    transpiled_base_gate = ['cx','sx','x']
    results_file_path = os.path.join(dir_name, str(rb_exp_name + "_output_data.json"))
    analysis_file_path = os.path.join(dir_name, str(rb_exp_name + "_output_analysis.json"))
    noise_model = create_depolarizing_noise_model()
    backend = QasmSimulator()
    rb = qe.randomized_benchmarking
    rb_exp = rb.StandardRB(
        exp_attributes["physical_qubits"],
        exp_attributes["lengths"],
        num_samples=exp_attributes["num_samples"],
        seed=exp_attributes["seed"],
    )
    rb_exp.set_analysis_options(gate_error_ratio=gate_error_ratio)
    experiment_obj = rb_exp.run(backend, noise_model=noise_model, basis_gates = transpiled_base_gate)
    exp_results = experiment_obj.data()
    with open(results_file_path, "w") as json_file:
        joined_list_data = [exp_attributes, exp_results]
        json_file.write(json.dumps(joined_list_data))
    _analysis_save(experiment_obj.analysis_result(None), analysis_file_path)


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
        epg_keys = list(sample_analysis["EPG"].keys())
        for qubits in epg_keys:
            sample_analysis["EPG"][str(qubits)] = sample_analysis["EPG"].pop(qubits)
        samples_analysis_list.append(sample_analysis)
    with open(analysis_file_path, "w") as json_file:
        json_file.write(json.dumps(samples_analysis_list))


DIRNAME = os.path.dirname(os.path.abspath(__file__))
for rb_type in sys.argv[1:]:
    if rb_type == "standard":
        standard_rb_exp_data_gen(DIRNAME)
    else:
        print("Skipping unknown argument " + rb_type)
