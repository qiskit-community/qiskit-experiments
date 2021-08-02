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
Utils to generate RB experiment data reference.
"""

import json
from typing import List

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error

from qiskit_experiments.database_service import DbAnalysisResultV1
from qiskit_experiments.database_service.json import ExperimentEncoder


def create_depolarizing_noise_model():
    """
    create noise model of depolarizing error
    Returns:
        NoiseModel: depolarizing error noise model
    """
    # the error parameters were taken from ibmq_manila on 17 june 2021
    p1q = 0.002257
    p2q = 0.006827
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), "x")
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), "sx")
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p2q, 2), "cx")
    return noise_model


def analysis_save(analysis_results: List[DbAnalysisResultV1], analysis_file_path: str):
    """
    The function is creating a json file from the data of the RB experiment analysis.
    Args:
        analysis_results: the analysis results to save.
        analysis_file_path (str): The path to save the json file.
    """
    dict_analysis_results = []
    for result in analysis_results:
        dict_analysis_results.append(
            {
                "name": result.name,
                "value": result.value,
                "extra": result.extra,
            }
        )
    print("Writing to file", analysis_file_path)
    with open(analysis_file_path, "w") as json_file:
        json.dump(dict_analysis_results, json_file, cls=ExperimentEncoder)
