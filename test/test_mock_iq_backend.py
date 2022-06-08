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

"""MockIQBackend tests."""

from test.base import QiskitExperimentsTestCase
from typing import Dict, List
from qiskit import QuantumCircuit

from qiskit.qobj.utils import MeasLevel

from qiskit_experiments.test.mock_iq_backend import MockIQBackend
from qiskit_experiments.test.mock_iq_helpers import MockIQExperimentHelper


def compute_probabilities_output(prob_list_output: List[Dict]) -> Dict[str, float]:
    """
    A function to compute the probability for parallel experiment on two qubits.
    Args:
        prob_list_output(List[Dict]): List of probability dictionaries.

    Returns:
        Dict: A dictionary for output strings and their probability.
    """
    output_dict = {}
    for i in range(2):
        for j in range(2):
            output_str = str(i) + str(j)
            output_dict[output_str] = prob_list_output[0][str(i)] * prob_list_output[1][str(j)]
    return output_dict


class MockIQReadoutAngleParallelHelper(MockIQExperimentHelper):
    """functions needed for Ramsey XY experiment on mock IQ backend"""

    def compute_probabilities(self, circuits: List[QuantumCircuit]) -> List[Dict[str, float]]:
        """Return the probability of being in the excited state."""
        num_qubits = 2
        output_dict_list = []
        for circuit in circuits:
            prob_help_list = [{}] * num_qubits
            for qubit_idx in range(num_qubits):
                probability_output_dict = {"1": 1 - circuit.metadata["xval"][qubit_idx]}
                probability_output_dict["0"] = 1 - probability_output_dict["1"]
                prob_help_list[qubit_idx] = probability_output_dict
            output_dict_list.append(compute_probabilities_output(prob_help_list))

        return output_dict_list


class MockIQBellStateHelper(MockIQExperimentHelper):
    """functions needed for Ramsey XY experiment on mock IQ backend"""

    def compute_probabilities(self, circuits: List[QuantumCircuit]) -> List[Dict[str, float]]:
        """Return the probability of being in the excited state."""
        output_dict_list = []
        for _ in circuits:
            prob_dict = {"00": 0.5, "11": 0.5}
            output_dict_list.append(prob_dict)

        return output_dict_list


class TestMockIQBackend(QiskitExperimentsTestCase):
    """Test the mock IQ backend"""

    def test_data_generation_2qubits(self):
        """
        Test readout angle experiment using a simulator.
        """
        num_qubits = 2
        num_shot = 10000
        backend = MockIQBackend(
            MockIQReadoutAngleParallelHelper(),
            iq_cluster_centers=[((-5.0, 4.0), (-5.0, -4.0)), ((3.0, 1.0), (5.0, -3.0))],
            iq_cluster_width=[1.0, 2.0],
        )
        circ0 = QuantumCircuit(num_qubits, num_qubits)
        circ0.x(1)
        circ0.measure(0, 0)
        circ0.measure(1, 1)

        circ0.metadata = {"qubits": [0, 1], "xval": [0, 1]}

        # Here meas_return is 'avg' so it will average on the results for each qubit.
        job = backend.run([circ0], shots=num_shot, meas_level=MeasLevel.KERNELED, meas_return="avg")
        res = job.result()
        self.assertEqual(len(res.results[0].data.memory), num_qubits)

        # Circuit to create Bell state
        backend.experiment_helper = MockIQBellStateHelper()
        circ1 = QuantumCircuit(num_qubits, num_qubits)
        circ1.h(0)
        circ1.cx(0, 1)
        circ1.measure(0, 0)
        circ1.measure(1, 1)

        # Now meas_return will be 'single' so it is expected that the number of results will be as
        # the number of shots.
        job = backend.run(
            [circ1], shots=num_shot, meas_level=MeasLevel.KERNELED, meas_return="single"
        )
        res = job.result()
        self.assertEqual(len(res.results[0].data.memory), num_shot)
        for data in res.results[0].data.memory:
            self.assertEqual(len(data), num_qubits)
