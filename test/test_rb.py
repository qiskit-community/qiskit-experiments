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
            exp_data (Dictionary): A dictionary with the experiment setup.
            rb_exp (RBExperiment): The instance for the experiment object.
        """
        exp_data = {"qubits": [0, 1], "lengths": [1, 3, 5, 7, 9], "num_samples": 1, "seed": 100}
        rb = qe.randomized_benchmarking
        rb_exp = rb.RBExperiment(
            exp_data["qubits"],
            exp_data["lengths"],
            num_samples=exp_data["num_samples"],
            seed=exp_data["seed"],
        )
        return exp_data, rb_exp

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

    def validate_metadata(self, circuits: list, exp_data: dict):
        """
        Validate the fields in "metadata" for the experiment.
        Args:
            circuits (list): A list containing quantum circuits
            exp_data (dict): A dictionary with the experiment variable ands values
        """
        for ind, qc in enumerate(circuits):
            self.assertEqual(
                qc.metadata["xdata"],
                exp_data["lengths"][ind],
                "The length of the experiment doen't match to the one provided.",
            )
            self.assertEqual(
                qc.metadata["qubits"],
                tuple(exp_data["qubits"]),
                "The qubits indices doesn't match the ran qubit indices.",
            )

    def test_RB_circuits(self):
        """
        Run the RB test for the circuits (checking the metadata, parameters and functionallity
        of the experiment.
        """
        exp_2_qubit_data_dict, exp_2_qubit_exp = self.rb_parameters_2_qubit()
        exp_2_qubit_circ = exp_2_qubit_exp.circuits()
        self.is_identity(exp_2_qubit_circ)
        self.validate_metadata(exp_2_qubit_circ, exp_2_qubit_data_dict)
