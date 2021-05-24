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

"""
A Tester for the RB experiment
"""

from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info import Clifford
from qiskit.quantum_info.operators.measures import process_fidelity
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeParis
from qiskit.exceptions import QiskitError
import numpy as np
import qiskit_experiments as qe



class TestRB(QiskitTestCase):
    """
    A simple and primitive backend, to be run by the RB tests
    """

    def rb_parameters_one_qubit(self):
        """
        Initialize data and executing RB experiment on one qubits with specific parameters
        Returns:
            dict: A dictionary with the experiment setup attributes.
            RBExperiment: The instance for the experiment object.
            ExperimentData: The experiment data and results after it had run.
        """
        backend = FakeParis()
        exp_attributes = {
            "qubits": [1],
            "lengths": [1, 3, 5, 7, 9],
            "num_samples": 1,
            "seed": 100,
        }
        rb = qe.randomized_benchmarking
        rb_exp = rb.RBExperiment(
            exp_attributes["qubits"],
            exp_attributes["lengths"],
            num_samples=exp_attributes["num_samples"],
            seed=exp_attributes["seed"],
        )
        exp_data = rb_exp.run(backend)
        exp_circuit = rb_exp.circuits()
        exp_transpiled_circuit = rb_exp.transpiled_circuits()
        self.validate_metadata(exp_circuit, exp_attributes)
        self.validate_circuit_data(exp_data, exp_attributes)
        self.is_identity_transpiled(exp_transpiled_circuit)
        self.is_identity(exp_circuit)

    def rb_parameters_two_qubit(self):
        """
        Initialize data and executing RB experiment on two qubits with specific parameters
        Returns:
            dict: A dictionary with the experiment setup attributes.
            RBExperiment: The instance for the experiment object.
            ExperimentData: The experiment data and results after it had run.
        """
        backend = FakeParis()
        exp_attributes = {
            "qubits": [0, 1],
            "lengths": [1, 3, 5, 7, 9],
            "num_samples": 1,
            "seed": 100,
        }
        rb = qe.randomized_benchmarking
        rb_exp = rb.RBExperiment(
            exp_attributes["qubits"],
            exp_attributes["lengths"],
            num_samples=exp_attributes["num_samples"],
            seed=exp_attributes["seed"],
        )
        exp_data = rb_exp.run(backend)
        exp_circuit = rb_exp.circuits()
        exp_transpiled_circuit = rb_exp.transpiled_circuits()
        self.validate_metadata(exp_circuit, exp_attributes)
        self.validate_circuit_data(exp_data, exp_attributes)
        self.is_identity_transpiled(exp_transpiled_circuit)
        self.is_identity(exp_circuit)

    def rb_parameters_three_qubit(self):
        """
        Initialize data and executing RB experiment on three qubits with specific parameters
        Returns:
            dict: A dictionary with the experiment setup attributes.
            RBExperiment: The instance for the experiment object.
            ExperimentData: The experiment data and results after it had run.
        """
        backend = FakeParis()
        exp_attributes = {
            "qubits": [0, 1, 2],
            "lengths": [1, 3, 5, 7, 9],
            "num_samples": 1,
            "seed": 100,
        }
        rb = qe.randomized_benchmarking
        rb_exp = rb.RBExperiment(
            exp_attributes["qubits"],
            exp_attributes["lengths"],
            num_samples=exp_attributes["num_samples"],
            seed=exp_attributes["seed"],
        )
        exp_data = rb_exp.run(backend)
        exp_circuit = rb_exp.circuits()
        exp_transpiled_circuit = rb_exp.transpiled_circuits()
        self.validate_metadata(exp_circuit, exp_attributes)
        self.validate_circuit_data(exp_data, exp_attributes)
        self.is_identity_transpiled(exp_transpiled_circuit)
        self.is_identity(exp_circuit)

    def is_identity(self, circuits: list):
        """Standard randomized benchmarking test - Identity check.
            (assuming all the operator are spanned by clifford group)
        Args:
            circuits (list): list of the circuits which we want to check
        """
        for qc in circuits:
            num_qubits = qc.num_qubits
            qc.remove_final_measurements()
            # Checking if the matrix representation is the identity matrix
            self.assertEqual(
                matrix_equal(Clifford(qc).to_matrix(), np.identity(2 ** num_qubits)),
                True,
                "Clifford sequence doesn't result in the identity matrix.",
            )

    def is_identity_transpiled(self, transpiled_circuits: list):
        """Standard randomized benchmarking test - Identity check for the transpiled circuits.
            Using
        Args:
            transpiled_circuits (list): list of the circuits which we want to check
        """
        for qc in transpiled_circuits:
            num_qubits = qc.num_qubits
            qc.remove_final_measurements()
            # Checking if the matrix representation is the identity matrix
            self.assertAlmostEqual(
                process_fidelity(Clifford(qc).to_matrix(), np.identity(2 ** num_qubits)),
                1,
                "Transpiled circuit doesn't result in the identity operator.",
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

    def validate_circuit_data(
        self, experiment: qe.experiment_data.ExperimentData, exp_attributes: dict
    ):
        """
        Validate that the metadata of the experiment after it had run matches the one provided.
        Args:

            experiment(qiskit_experiments.experiment_data.ExperimentData): The experiment
            data and results after it run.
            exp_attributes (dict): A dictionary with the experiment variable ands values
        """
        for ind, data in enumerate(experiment.data):
            experiment_information = data["metadata"]
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

    def _exp_data_prpeties(self):
        """
        Return a list of dictionaries that contains invalid experiment propeties to check errors.
        """
        exp_data_list = [
            {"qubits": [3, 3], "lengths": [1, 3, 5, 7, 9], "num_samples": 0, "seed": 100},
            {"qubits": [-1], "lengths": [1, 3, 5, 7, 9], "num_samples": 1, "seed": 100},
            {"qubits": [0, 1], "lengths": [1, 3, 5, -7, 9], "num_samples": 1, "seed": 100},
            {"qubits": [0, 1], "lengths": [1, 3, 5, 7, 9], "num_samples": -4, "seed": 100},
            {"qubits": [0, 1], "lengths": [1, 3, 5, 7, 9], "num_samples": 0, "seed": 100},
            {"qubits": [0, 1], "lengths": [1, 5, 5, 5, 9], "num_samples": 0, "seed": 100},
        ]
        return exp_data_list

    def _test_input(self):
        """
        Check that errors emerge when invalid input is given to the RB experiment.
        """
        exp_data_list = self._exp_data_prpeties()
        rb = qe.randomized_benchmarking
        for exp_data in exp_data_list:
            self.assertRaises(
                QiskitError,
                rb.RBExperiment,
                exp_data["qubits"],
                exp_data["lengths"],
                num_samples=exp_data["num_samples"],
                seed=exp_data["seed"],
            )

    def test_RB_circuits(self):
        """
        Run the RB test for the circuits (checking the metadata, parameters and functionallity
        of the experiment.
        """
        # self._test_input()
        self.rb_parameters_one_qubit()
        self.rb_parameters_two_qubit()
        self.rb_parameters_three_qubit()
