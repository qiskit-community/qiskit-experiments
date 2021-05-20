# -*- coding: utf-8 -*-

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


import qiskit_experiments as qe
from qiskit.quantum_info import Clifford
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeParis
import numpy as np


"""
A Tester for the RB experiment
"""


class TestRB(QiskitTestCase):
    """
    A simple and primitive backend, to be run by the RB tests
    """

    @staticmethod
    def rb_parameters_2_qubit():
        """
        Initialize data for a RB experiment with specific parameters
        Returns:
            exp_attributes (Dictionary): A dictionary with the experiment setup attributes.
            rb_exp (RBExperiment): The instance for the experiment object.
            exp_data (RBExperiment): The experiment data and results after it had run.
        """
        backend = FakeParis()
        exp_attributes = {"qubits": [0, 1], "lengths": [1, 3, 5, 7, 9], "num_samples": 1, "seed": 100}
        rb = qe.randomized_benchmarking
        rb_exp = rb.RBExperiment(
            exp_attributes["qubits"],
            exp_attributes["lengths"],
            num_samples=exp_attributes["num_samples"],
            seed=exp_attributes["seed"],
        )
        exp_data = rb_exp.run(backend)
        return exp_attributes, rb_exp, exp_data

    def is_identity(self, circuits: list):
        """Standard randomized benchmarking test - Identity check
            (assuming all the operator are spanned by clifford group)
        Args:
            circuits: list of the circuits which we want to check
        """
        for qc in circuits:
            num_qubits = qc.num_qubits
            qc.remove_final_measurements()
            # Checking if the matrix representation is the identity matrix
            self.assertEqual(
                np.allclose(Clifford(qc).to_matrix(), np.identity(2 ** num_qubits)),
                True,
                "Clifford sequence doesn't result in the identity matrix.",
            )

    def validate_metadata(self, circuits: list, exp_attributes: dict):
        """
        Validate the fields in "metadata" for the experiment.
        Args:
            circuits (list): A list containing quantum circuits
            exp_attributes (dict): A dictionary with the experiment variable ands values
        """
        for ind, qc in enumerate(circuits):
            self.assertEqual(
                qc.metadata["xval"],
                exp_attributes["lengths"][ind],
                "The number of gates in the experiment metadata doesn't match to the one provided.",
            )
            self.assertEqual(
                qc.metadata["qubits"],
                tuple(exp_attributes["qubits"]),
                "The qubits indices in the experiment metadata doesn't match to the one provided.",
            )

    def validate_circuit_data(self, experiment: qe.experiment_data.ExperimentData,
                              exp_attributes: dict):
        """
        Validate that the metadata of the experiment after it had run matches the one provided.
        Args:
            experiment(qiskit_experiments.experiment_data.ExperimentData): The experiment 
            data and results after it run.
            exp_attributes (dict): A dictionary with the experiment variable ands values

        Returns:

        """
        for ind, data in enumerate(experiment.data):
            experiment_information = data['metadata']
            self.assertEqual(
                experiment_information["xdata"],
                exp_attributes["lengths"][ind],
                "The number of gates in the experiment doesn't match to the one in the metadata.",
            )
            self.assertEqual(
                experiment_information["qubits"],
                exp_attributes["qubits"],
                "The qubits indices in the experiment doesn't match to the one in the metadata.",
            )

    def test_RB_circuits(self):
        """
        Run the RB test for the circuits (checking the metadata, parameters and functionallity
        of the experiment.
        """
        exp_2_qubit_metadata_attributes_dict, exp_2_qubit_exp, exp_2_quibit_exp_data = self.rb_parameters_2_qubit()
        exp_2_qubit_circuit = exp_2_qubit_exp.circuits()
        self.is_identity(exp_2_qubit_circuit)
        self.validate_metadata(exp_2_qubit_circuit, exp_2_qubit_metadata_attributes_dict)
        self.validate_circuit_data(exp_2_quibit_exp_data, exp_2_qubit_metadata_attributes_dict)

