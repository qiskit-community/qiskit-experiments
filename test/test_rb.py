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

import numpy as np
from ddt import ddt, data, unpack
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info import Clifford
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeParis
from qiskit.circuit.library import XGate, CXGate
import qiskit_experiments as qe


@ddt
class TestRB(QiskitTestCase):
    """
    A test class for the RB Experiment to check that the RBExperiment class is working correctly.
    """

    @data([[3]], [[4, 7]], [[0, 5, 3]])
    @unpack
    def test_rb_experiment(self, qubits: list):
        """
        Initializes data and executes an RB experiment with specific parameters.
        Args:
            qubits (list): A list containing qubit indices for the experiment
        """
        backend = FakeParis()
        exp_attributes = {
            "qubits": qubits,
            "lengths": [1, 4, 6, 9, 13, 16],
            "num_samples": 2,
            "seed": 100,
        }
        rb = qe.randomized_benchmarking
        rb_exp = rb.RBExperiment(
            exp_attributes["qubits"],
            exp_attributes["lengths"],
            num_samples=exp_attributes["num_samples"],
            seed=exp_attributes["seed"],
        )
        experiment_obj = rb_exp.run(backend)
        exp_data = experiment_obj.experiment
        exp_circuits = rb_exp.circuits()
        self.validate_metadata(exp_circuits, exp_attributes)
        self.validate_circuit_data(exp_data, exp_attributes)
        self.is_identity(exp_circuits)

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
            self.assertTrue(
                matrix_equal(Clifford(qc).to_matrix(), np.identity(2 ** num_qubits)),
                "Clifford sequence doesn't result in the identity matrix.",
            )

    def validate_metadata(self, circuits: list, exp_attributes: dict):
        """
        Validate the fields in "metadata" for the experiment.
        Args:
            circuits (list): A list containing quantum circuits
            exp_attributes (dict): A dictionary with the experiment variable and values
        """
        for qc in circuits:
            self.assertTrue(
                qc.metadata["xval"] in exp_attributes["lengths"],
                "The number of gates in the experiment metadata doesn't match "
                "any of the provided lengths",
            )
            self.assertTrue(
                qc.metadata["qubits"] == tuple(exp_attributes["qubits"]),
                "The qubits indices in the experiment metadata doesn't match to the one provided.",
            )

    def validate_circuit_data(
        self,
        experiment: qe.randomized_benchmarking.rb_experiment.RBExperiment,
        exp_attributes: dict,
    ):
        """
        Validate that the metadata of the experiment after it had run matches the one provided.
        Args:
            experiment: The experiment data and results after it run
            exp_attributes (dict): A dictionary with the experiment variable and values
        """
        self.assertTrue(
            exp_attributes["lengths"] == experiment.experiment_options.lengths,
            "The number of gates in the experiment doesn't match to the one in the metadata.",
        )
        self.assertTrue(
            exp_attributes["num_samples"] == experiment.experiment_options.num_samples,
            "The number of samples in the experiment doesn't match to the one in the metadata.",
        )
        self.assertTrue(
            tuple(exp_attributes["qubits"]) == experiment.physical_qubits,
            "The qubits indices in the experiment doesn't match to the one in the metadata.",
        )


@ddt
class TestInterleavedRB(TestRB):
    """
    A test class for the interleaved RB Experiment to check that the
    InterleavedRBExperiment class is working correctly.
    """

    @data([XGate(), [3]], [CXGate(), [4, 7]])
    @unpack
    def test_interleaved_rb_experiment(self, interleaved_element: "Gate", qubits: list):
        """
        Initializes data and executes an interleaved RB experiment with specific parameters.
        Args:
            interleaved_element: The Clifford element to interleave
            qubits (list): A list containing qubit indices for the experiment
        """
        backend = FakeParis()
        exp_attributes = {
            "interleaved_element": interleaved_element,
            "qubits": qubits,
            "lengths": [1, 4, 6, 9, 13, 16],
            "num_samples": 2,
            "seed": 100,
        }
        rb = qe.randomized_benchmarking
        rb_exp = rb.InterleavedRBExperiment(
            exp_attributes["interleaved_element"],
            exp_attributes["qubits"],
            exp_attributes["lengths"],
            num_samples=exp_attributes["num_samples"],
            seed=exp_attributes["seed"],
        )
        experiment_obj = rb_exp.run(backend)
        exp_data = experiment_obj.experiment
        exp_circuits = rb_exp.circuits()
        self.validate_metadata(exp_circuits, exp_attributes)
        self.validate_circuit_data(exp_data, exp_attributes)
        self.is_identity(exp_circuits)
