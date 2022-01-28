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
from test.base import QiskitExperimentsTestCase
import numpy as np
from ddt import ddt, data, unpack
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info import Clifford

from qiskit.test.mock import FakeParis
from qiskit.providers.aer import AerSimulator
from qiskit.exceptions import QiskitError
from qiskit.circuit.library import (
    XGate,
    CXGate,
    TGate,
)
from qiskit_experiments.library import StandardRB, InterleavedRB


@ddt
class TestStandardRB(QiskitExperimentsTestCase):
    """
    A test class for the RB Experiment to check that the StandardRB class is working correctly.
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
            "physical_qubits": qubits,
            "lengths": [1, 4, 6, 9, 13, 16],
            "num_samples": 3,
            "seed": 100,
        }
        rb_exp = StandardRB(
            exp_attributes["physical_qubits"],
            exp_attributes["lengths"],
            num_samples=exp_attributes["num_samples"],
            seed=exp_attributes["seed"],
        )
        exp_data = rb_exp.run(backend)
        self.assertExperimentDone(exp_data)
        exp = exp_data.experiment
        exp_circuits = rb_exp.circuits()
        self.validate_metadata(exp_circuits, exp_attributes)
        self.validate_circuit_data(exp, exp_attributes)
        self.is_identity(exp_circuits)

    def is_identity(self, circuits: list):
        """Standard randomized benchmarking test - Identity check.
            (assuming all the operator are spanned by clifford group)
        Args:
            circuits (list): list of the circuits which we want to check
        """
        for circ in circuits:
            num_qubits = circ.num_qubits
            circ.remove_final_measurements()
            # Checking if the matrix representation is the identity matrix
            self.assertTrue(
                matrix_equal(Clifford(circ).to_matrix(), np.identity(2 ** num_qubits)),
                "Clifford sequence doesn't result in the identity matrix.",
            )

    def validate_metadata(self, circuits: list, exp_attributes: dict):
        """
        Validate the fields in "metadata" for the experiment.
        Args:
            circuits (list): A list containing quantum circuits
            exp_attributes (dict): A dictionary with the experiment variable and values
        """
        for circ in circuits:
            self.assertTrue(
                circ.metadata["xval"] in exp_attributes["lengths"],
                "The number of gates in the experiment metadata doesn't match "
                "any of the provided lengths",
            )
            self.assertTrue(
                circ.metadata["physical_qubits"] == tuple(exp_attributes["physical_qubits"]),
                "The qubits indices in the experiment metadata doesn't match to the one provided.",
            )

    def validate_circuit_data(
        self,
        experiment: StandardRB,
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
            tuple(exp_attributes["physical_qubits"]) == experiment.physical_qubits,
            "The qubits indices in the experiment doesn't match to the one in the metadata.",
        )

    @staticmethod
    def _exp_data_properties():
        """
        Creates a list of dictionaries that contains invalid experiment properties to check errors.
        The dict invalid data is as following:
            exp_data_list[1]: same index of qubit.
            exp_data_list[2]: the length of the sequence has negative number.
            exp_data_list[3]: num of samples is negative.
            exp_data_list[4]: num of samples is 0.
            exp_data_list[5]: the length of the sequence list has duplicates.
        Returns:
            list[dict]: list of dictionaries with experiment properties.
        """
        exp_data_list = [
            {"physical_qubits": [3, 3], "lengths": [1, 3, 5, 7, 9], "num_samples": 1, "seed": 100},
            {"physical_qubits": [0, 1], "lengths": [1, 3, 5, -7, 9], "num_samples": 1, "seed": 100},
            {"physical_qubits": [0, 1], "lengths": [1, 3, 5, 7, 9], "num_samples": -4, "seed": 100},
            {"physical_qubits": [0, 1], "lengths": [1, 3, 5, 7, 9], "num_samples": 0, "seed": 100},
            {"physical_qubits": [0, 1], "lengths": [1, 5, 5, 5, 9], "num_samples": 2, "seed": 100},
        ]
        return exp_data_list

    def test_input(self):
        """
        Check that errors emerge when invalid input is given to the RB experiment.
        """
        exp_data_list = self._exp_data_properties()
        for exp_data in exp_data_list:
            self.assertRaises(
                QiskitError,
                StandardRB,
                exp_data["physical_qubits"],
                exp_data["lengths"],
                num_samples=exp_data["num_samples"],
                seed=exp_data["seed"],
            )

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = StandardRB([0, 1], lengths=[10, 20, 30, 40], num_samples=10)
        loaded_exp = StandardRB.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = StandardRB([0, 1], lengths=[10, 20, 30, 40], num_samples=10)
        self.assertRoundTripSerializable(exp, self.json_equiv)


@ddt
class TestInterleavedRB(TestStandardRB):
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
            "physical_qubits": qubits,
            "lengths": [1, 4, 6, 9, 13, 16],
            "num_samples": 3,
            "seed": 100,
        }
        rb_exp = InterleavedRB(
            exp_attributes["interleaved_element"],
            exp_attributes["physical_qubits"],
            exp_attributes["lengths"],
            num_samples=exp_attributes["num_samples"],
            seed=exp_attributes["seed"],
        )
        experiment_obj = rb_exp.run(backend)
        self.assertExperimentDone(experiment_obj)
        exp_data = experiment_obj.experiment
        exp_circuits = rb_exp.circuits()
        self.validate_metadata(exp_circuits, exp_attributes)
        self.validate_circuit_data(exp_data, exp_attributes)
        self.is_identity(exp_circuits)

    @data([XGate(), [3], 4], [CXGate(), [4, 7], 5])
    @unpack
    def test_interleaved_structure(self, interleaved_element: "Gate", qubits: list, length: int):
        """
        Verifies that when generating an interleaved circuit, it will be
        identical to the original circuit up to additions of
        barrier and interleaved element between any two cliffords
        Args:
            interleaved_element: The clifford element to interleave
            qubits: A list containing qubit indices for the experiment
            length: The number of cliffords in the tested circuit
        """
        exp = InterleavedRB(
            qubits=qubits, interleaved_element=interleaved_element, lengths=[length], num_samples=1
        )

        circuits = exp.circuits()
        c_std = circuits[0]
        c_int = circuits[1]
        if c_std.metadata["interleaved"]:
            c_std, c_int = c_int, c_std
        num_cliffords = c_std.metadata["xval"]
        std_idx = 0
        int_idx = 0
        for _ in range(num_cliffords):
            # barrier
            self.assertEqual(c_std[std_idx][0].name, "barrier")
            self.assertEqual(c_int[int_idx][0].name, "barrier")
            # clifford
            self.assertEqual(c_std[std_idx + 1], c_int[int_idx + 1])
            # for interleaved circuit: barrier + interleaved element
            self.assertEqual(c_int[int_idx + 2][0].name, "barrier")
            self.assertEqual(c_int[int_idx + 3][0].name, interleaved_element.name)
            std_idx += 2
            int_idx += 4

    def test_non_clifford_interleaved_element(self):
        """Verifies trying to run interleaved RB with non Clifford element throws an exception"""
        qubits = 1
        lengths = [1, 4, 6, 9, 13, 16]
        interleaved_element = TGate()  # T gate is not Clifford, this should fail
        self.assertRaises(
            QiskitError,
            InterleavedRB,
            interleaved_element,
            qubits,
            lengths,
        )

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = InterleavedRB(CXGate(), [0, 1], lengths=[10, 20, 30, 40], num_samples=10)
        loaded_exp = InterleavedRB.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = InterleavedRB(CXGate(), [0, 1], lengths=[10, 20, 30, 40], num_samples=10)
        self.assertRoundTripSerializable(exp, self.json_equiv)
