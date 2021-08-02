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
Ideal probability data generation.
"""
import argparse
import json
import os

from qiskit_experiments.database_service.json import ExperimentEncoder
from qiskit_experiments.library import QuantumVolume


def _create_qv_ideal_probabilities(dir_path: str, seed_val: int):
    """
    create quantum volume circuits using seed, and save their ideal probabilities
    vector in a json

    Args:
        dir_path(str): The directory which the data will be saved to.
        seed_val (int): Seed value.
    """
    num_of_qubits = 3
    qv_exp = QuantumVolume(num_of_qubits, seed=seed_val)
    qv_exp.set_experiment_options(trials=20)
    qv_circs = qv_exp.circuits()
    simulation_probabilities = [
        list(qv_circ.metadata["ideal_probabilities"]) for qv_circ in qv_circs
    ]

    result_file_path = os.path.join(dir_path, "qv_ideal_probabilities.json")
    with open(result_file_path, "w") as json_file:
        json.dump(simulation_probabilities, json_file, cls=ExperimentEncoder)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Quantum volume ref data generation.")
    parser.add_argument(
        "--folder",
        required=False,
        default="test/quantum_volume/refdata",
        type=str,
    )
    parser.add_argument(
        "--seed",
        required=False,
        default=42,
        type=int,
    )
    args = parser.parse_args()

    print(f"Generating data for {os.path.basename(__file__)}...")

    _create_qv_ideal_probabilities(args.folder, args.seed)

    print("Completed")
