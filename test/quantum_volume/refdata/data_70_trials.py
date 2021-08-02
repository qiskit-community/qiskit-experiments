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
70 trials data generation.
"""
import argparse
import json
import os

from qiskit import Aer

from qiskit_experiments.database_service.json import ExperimentEncoder
from qiskit_experiments.library import QuantumVolume


def _create_qv_data_70_trials(dir_path: str, seed_val: int):
    """
    create quantum volume experiment_data using seed, and save it as a json

    Args:
        dir_path(str): The directory which the data will be saved to.
        seed_val (int): Seed value.
    """
    num_of_qubits = 3
    backend = Aer.get_backend("aer_simulator")

    qv_exp = QuantumVolume(num_of_qubits, seed=seed_val)
    qv_exp.set_experiment_options(trials=70)
    qv_data = qv_exp.run(backend)
    qv_data.block_for_results()

    result_file_path = os.path.join(dir_path, "qv_data_70_trials.json")
    with open(result_file_path, "w") as json_file:
        json.dump(qv_data.data(), json_file, cls=ExperimentEncoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantum volume ref data generation.')
    parser.add_argument(
        '--folder',
        required=False,
        default="test/quantum_volume/refdata",
        type=str,
    )
    parser.add_argument(
        '--seed',
        required=False,
        default=42,
        type=int,
    )
    args = parser.parse_args()

    print(f"Generating data for {os.path.basename(__file__)}...")

    _create_qv_data_70_trials(args.folder, args.seed)

    print(f"Completed")
