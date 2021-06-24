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
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    IGate,
    XGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    SdgGate,
    CXGate,
    CZGate,
    SwapGate,
)
from qiskit.providers.aer import AerSimulator
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
        backend = AerSimulator.from_backend(FakeParis())
        exp_attributes = {
            "qubits": qubits,
            "lengths": [1, 4, 6, 9, 13, 16],
            "num_samples": 2,
            "seed": 100,
        }
        rb = qe.randomized_benchmarking
        rb_exp = rb.StandardRB(
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
        experiment: qe.randomized_benchmarking.rb_experiment.StandardRB,
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
    InterleavedRB class is working correctly.
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
        backend = AerSimulator.from_backend(FakeParis())
        exp_attributes = {
            "interleaved_element": interleaved_element,
            "qubits": qubits,
            "lengths": [1, 4, 6, 9, 13, 16],
            "num_samples": 2,
            "seed": 100,
        }
        rb = qe.randomized_benchmarking
        rb_exp = rb.InterleavedRB(
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


@ddt
class TestRBUtilities(QiskitTestCase):
    """
    A test class for additional functionality provided by the RBExperiment
    class.
    """

    instructions = {
        "i": IGate(),
        "x": XGate(),
        "y": YGate(),
        "z": ZGate(),
        "h": HGate(),
        "s": SGate(),
        "sdg": SdgGate(),
        "cx": CXGate(),
        "cz": CZGate(),
        "swap": SwapGate(),
    }
    seed = 42

    @data(
        [1, {((0,), "x"): 3, ((0,), "y"): 2, ((0,), "h"): 1}],
        [5, {((1,), "x"): 3, ((4,), "y"): 2, ((1,), "h"): 1, ((1, 4), "cx"): 7}],
    )
    @unpack
    def test_count_ops(self, num_qubits, expected_counts):
        """Testing the count_ops utility function
        this function receives a circuit and counts the number of gates
        in it, counting gates for different qubits separately"""
        circuit = QuantumCircuit(num_qubits)
        gates_to_add = []
        for gate, count in expected_counts.items():
            gates_to_add += [gate for _ in range(count)]
        rng = np.random.default_rng(self.seed)
        rng.shuffle(gates_to_add)
        for qubits, gate in gates_to_add:
            circuit.append(self.instructions[gate], qubits)
        counts = qe.randomized_benchmarking.RBUtils.count_ops(circuit)
        self.assertDictEqual(expected_counts, counts)

    def test_calculate_1q_epg(self):
        """Testing the calculation of 1 qubit error per gate
        The EPG is computed based on the error per clifford determined
        in the RB experiment, the gate counts, and an estimate about the
        relations between the errors of different gate types
        """
        epc_1_qubit = 0.0037
        qubits = [0]
        gate_error_ratio = {((0,), "id"): 1, ((0,), "rz"): 0, ((0,), "sx"): 1, ((0,), "x"): 1}
        gates_per_clifford = {((0,), "rz"): 10.5, ((0,), "sx"): 8.15, ((0,), "x"): 0.25}
        epg = qe.randomized_benchmarking.RBUtils.calculate_1q_epg(
            epc_1_qubit, qubits, gate_error_ratio, gates_per_clifford
        )
        error_dict = {
            ((0,), "rz"): 0,
            ((0,), "sx"): 0.0004432101747785104,
            ((0,), "x"): 0.0004432101747785104,
        }

        for gate in ["x", "sx", "rz"]:
            expected_epg = error_dict[((0,), gate)]
            actual_epg = epg[0][gate]
            self.assertTrue(np.allclose(expected_epg, actual_epg, rtol=1.0e-2))
