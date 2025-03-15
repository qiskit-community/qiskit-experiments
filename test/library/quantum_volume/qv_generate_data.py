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
Code for generating data for the Quantum Volume experiment for testing.
"""

import os
import sys
import json
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors.standard_errors import (
    depolarizing_error,
    thermal_relaxation_error,
)
from qiskit_aer.noise.errors import readout_error
from qiskit_experiments.library import QuantumVolume
from qiskit_experiments.framework import ExperimentEncoder

SEED = 42


def create_qv_ideal_probabilities(dir_path: str):
    """
    create quantum volume circuits using seed, and save their ideal probabilities vector in a json
    Args:
        dir_path(str): The directory which the data will be saved to.
    """
    num_of_qubits = 3
    qv_exp = QuantumVolume(range(num_of_qubits), seed=SEED)
    qv_exp.set_experiment_options(trials=20)
    qv_circs = qv_exp.circuits()
    simulation_probabilities = [
        list(qv_circ.metadata["ideal_probabilities"]) for qv_circ in qv_circs
    ]

    result_file_path = os.path.join(dir_path, "qv_ideal_probabilities.json")
    with open(result_file_path, "w", encoding="utf-8") as json_file:
        json.dump(simulation_probabilities, json_file, cls=ExperimentEncoder)


def create_qv_data_70_trials(dir_path: str):
    """
    create quantum volume experiment_data using seed, and save it as a json
    Args:
        dir_path(str): The directory which the data will be saved to.
    """
    num_of_qubits = 3
    backend = AerSimulator(seed_simulator=SEED)

    qv_exp = QuantumVolume(range(num_of_qubits), seed=SEED)
    qv_exp.set_experiment_options(trials=70)
    qv_data = qv_exp.run(backend).block_for_results()

    result_file_path = os.path.join(dir_path, "qv_data_70_trials.json")
    with open(result_file_path, "w", encoding="utf-8") as json_file:
        json.dump(qv_data.data(), json_file, cls=ExperimentEncoder)


def create_qv_data_low_hop(dir_path: str):
    """
    create quantum volume experiment_data using seed, and save it as a json
    the circuit is generated with high noise, so the mean hop is below 2/3
    Args:
        dir_path(str): The directory which the data will be saved to.
    """
    num_of_qubits = 4
    backend = AerSimulator(seed_simulator=SEED)
    basis_gates = ["id", "rz", "sx", "x", "cx", "reset"]
    noise = create_high_noise_model()

    qv_exp = QuantumVolume(range(num_of_qubits), seed=SEED)
    qv_exp.set_transpile_options(basis_gates=basis_gates)
    qv_data = qv_exp.run(backend, noise_model=noise, basis_gates=basis_gates).block_for_results()

    result_file_path = os.path.join(dir_path, "qv_data_high_noise.json")
    with open(result_file_path, "w", encoding="utf-8") as json_file:
        json.dump(qv_data.data(), json_file, cls=ExperimentEncoder)


def create_qv_data_low_confidence(dir_path: str):
    """
    create quantum volume experiment_data using seed, and save it as a json
    the circuit is generated with moderate noise, so the mean hop is above 2/3,
    but there are not enough trials, so the confidence is below threshold
    Args:
        dir_path(str): The directory which the data will be saved to.
    """
    num_of_qubits = 4
    backend = AerSimulator(seed_simulator=SEED)
    basis_gates = ["id", "rz", "sx", "x", "cx", "reset"]
    noise = create_noise_model()

    qv_exp = QuantumVolume(range(num_of_qubits), seed=SEED)
    qv_exp.set_transpile_options(basis_gates=basis_gates)
    qv_data = qv_exp.run(backend, noise_model=noise, basis_gates=basis_gates).block_for_results()

    result_file_path = os.path.join(dir_path, "qv_data_moderate_noise_100_trials.json")
    with open(result_file_path, "w", encoding="utf-8") as json_file:
        json.dump(qv_data.data(), json_file, cls=ExperimentEncoder)


def create_qv_data_high_confidence(dir_path: str):
    """
    create quantum volume experiment_data using seed, and save it as a json
    the circuit is generated with moderate noise, so the mean hop is above 2/3,
    and also with enough trials, so the confidence is above threshold
    Args:
        dir_path(str): The directory which the data will be saved to.
    """
    num_of_qubits = 4
    backend = AerSimulator(seed_simulator=SEED)
    basis_gates = ["id", "rz", "sx", "x", "cx", "reset"]
    noise = create_noise_model()

    qv_exp = QuantumVolume(range(num_of_qubits), seed=SEED)
    qv_exp.set_experiment_options(trials=300)
    qv_exp.set_transpile_options(basis_gates=basis_gates)
    qv_data = qv_exp.run(backend, noise_model=noise, basis_gates=basis_gates).block_for_results()

    result_file_path = os.path.join(dir_path, "qv_data_moderate_noise_300_trials.json")
    with open(result_file_path, "w", encoding="utf-8") as json_file:
        json.dump(qv_data.data(), json_file, cls=ExperimentEncoder)

    result_dicts = [t._asdict() for t in qv_data.analysis_results(dataframe=True).itertuples()]
    result_file_path = os.path.join(dir_path, "qv_result_moderate_noise_300_trials.json")
    with open(result_file_path, "w", encoding="utf-8") as json_file:
        json.dump(result_dicts, json_file, cls=ExperimentEncoder)


def create_noise_model():
    """
    create noise model with depolarizing error, thermal error and readout error
    Returns:
        NoiseModel: noise model
    """
    noise_model = NoiseModel()
    p1q = 0.0004
    p2q = 0.01
    depol_sx = depolarizing_error(p1q, 1)
    depol_x = depolarizing_error(p1q, 1)
    depol_cx = depolarizing_error(p2q, 2)

    # Add T1/T2 noise to the simulation
    t_1 = 110e3
    t_2 = 120e3
    gate1q = 50
    gate2q = 100
    termal_sx = thermal_relaxation_error(t_1, t_2, gate1q)
    termal_x = thermal_relaxation_error(t_1, t_2, gate1q)
    termal_cx = thermal_relaxation_error(t_1, t_2, gate2q).tensor(
        thermal_relaxation_error(t_1, t_2, gate2q)
    )

    noise_model.add_all_qubit_quantum_error(depol_sx.compose(termal_sx), "sx")
    noise_model.add_all_qubit_quantum_error(depol_x.compose(termal_x), "x")
    noise_model.add_all_qubit_quantum_error(depol_cx.compose(termal_cx), "cx")

    read_err = readout_error.ReadoutError([[0.98, 0.02], [0.04, 0.96]])
    noise_model.add_all_qubit_readout_error(read_err)
    return noise_model


def create_high_noise_model():
    """
    create high noise model with depolarizing error, thermal error and readout error
    Returns:
        NoiseModel: noise model
    """
    noise_model = NoiseModel()
    p1q = 0.004
    p2q = 0.05
    depol_sx = depolarizing_error(p1q, 1)
    depol_x = depolarizing_error(p1q, 1)
    depol_cx = depolarizing_error(p2q, 2)

    # Add T1/T2 noise to the simulation
    t_1 = 110e2
    t_2 = 120e2
    gate1q = 50
    gate2q = 100
    termal_sx = thermal_relaxation_error(t_1, t_2, gate1q)
    termal_x = thermal_relaxation_error(t_1, t_2, gate1q)
    termal_cx = thermal_relaxation_error(t_1, t_2, gate2q).tensor(
        thermal_relaxation_error(t_1, t_2, gate2q)
    )

    noise_model.add_all_qubit_quantum_error(depol_sx.compose(termal_sx), "sx")
    noise_model.add_all_qubit_quantum_error(depol_x.compose(termal_x), "x")
    noise_model.add_all_qubit_quantum_error(depol_cx.compose(termal_cx), "cx")

    read_err = readout_error.ReadoutError([[0.98, 0.02], [0.04, 0.96]])
    noise_model.add_all_qubit_readout_error(read_err)
    return noise_model


def main(argv):
    """Run data generation script"""
    directory = os.path.dirname(os.path.abspath(__file__))
    for generation_type in argv:
        if generation_type == "circuits":
            create_qv_ideal_probabilities(directory)
        elif generation_type == "analysis":
            create_qv_data_70_trials(directory)
            create_qv_data_low_hop(directory)
            create_qv_data_low_confidence(directory)
            create_qv_data_high_confidence(directory)
        else:
            print("Skipping unknown argument " + generation_type)


if __name__ == "__main__":
    main(sys.argv[1:])
