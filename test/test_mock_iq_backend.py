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
from typing import Dict, List, Optional, Tuple
from itertools import product
import numpy as np
from qiskit import QuantumCircuit
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit_experiments.test.mock_iq_backend import MockIQBackend
from qiskit_experiments.test.mock_iq_helpers import MockIQExperimentHelper

# Define an IQ point typing class.
IQPoint = Tuple[float, float]


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


class MockIQReadoutAmplitudeHelper(MockIQExperimentHelper):
    """functions needed for readout-amplitude experiment on mock IQ Backend"""

    def __init__(
        self,
        alter_centers: Optional[bool] = True,
        alter_widths: Optional[bool] = True,
        iq_cluster_centers: Optional[List[Tuple[IQPoint, IQPoint]]] = None,
        iq_cluster_width: Optional[List[float]] = None,
    ) -> None:
        """Construct a MockIQReadoutAmplitudeHelper instance.

        Args:
            alter_centers: Whether to alter the centers of the IQ clusters based on
                the circuit 'xval'. Defaults to True.
            alter_widths: Whether to alter the widths of the IQ clusters based on
                the circuit 'xval'. Defaults to True.
            iq_cluster_centers: A list of tuples containing the clusters' centers in the IQ plane. There
                are different centers for different logical values of the qubit.
            iq_cluster_width: A list of standard deviation values for the sampling of each qubit.
        """
        super().__init__(iq_cluster_centers, iq_cluster_width)
        self.alter_centers = alter_centers
        self.alter_widths = alter_widths

    def compute_probabilities(self, circuits: List[QuantumCircuit]) -> List[Dict[str, float]]:
        probabilities = []
        for circ in circuits:
            probabilities.append({circ.metadata["prep"]: 1.0})
        return probabilities

    def iq_clusters(
        self,
        circuits: List[QuantumCircuit],
    ) -> Tuple[List[Tuple[IQPoint, IQPoint]], List[float]]:
        """Multiplies the cluster centers by the circuits' 'xval' values."""
        output = []
        for circuit in circuits:
            if hasattr(circuit, "metadata") and "xval" in circuit.metadata.keys():
                xval = circuit.metadata["xval"]
            else:
                xval = 1.0

            new_centers = np.array(self.iq_cluster_centers)
            new_widths = np.array(self.iq_cluster_width)
            if self.alter_centers:
                new_centers = new_centers * xval
            if self.alter_widths:
                new_widths = new_widths * xval
            output.append((new_centers, new_widths))
        return output


class TestMockIQBackend(QiskitExperimentsTestCase):
    """Test the mock IQ backend"""

    def test_data_generation_2qubits(self):
        """
        Test readout angle experiment using a simulator.
        """
        num_qubits = 2
        num_shot = 10000
        iq_cluster_centers = [((-5.0, 4.0), (-5.0, -4.0)), ((3.0, 1.0), (5.0, -3.0))]
        iq_cluster_width = [1.0, 2.0]
        backend = MockIQBackend(
            MockIQReadoutAngleParallelHelper(
                iq_cluster_centers=iq_cluster_centers,
                iq_cluster_width=iq_cluster_width,
            ),
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
        backend.experiment_helper = MockIQBellStateHelper(
            iq_cluster_centers=iq_cluster_centers,
            iq_cluster_width=iq_cluster_width,
        )
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

    def test_iq_helper_iq_centers(self):
        """
        Test IQ centroid centers and widths are modified by MockIQExperimentHelper.
        """
        delta = 0.005

        num_shots = 10000
        bare_centers = [[(0, -1), (1, 1)]]
        backend = MockIQBackend(
            MockIQReadoutAmplitudeHelper(
                alter_widths=False,
                iq_cluster_centers=bare_centers,
                iq_cluster_width=[0.1, 0.1],
            ),
        )

        # Create template circuits
        circ_prep_0 = QuantumCircuit(1, 1)
        circ_prep_0.measure(0, 0)
        circ_prep_0.metadata = {"prep": "0"}
        circ_prep_1 = QuantumCircuit(1, 1)
        circ_prep_1.x(0)
        circ_prep_1.measure(0, 0)
        circ_prep_1.metadata = {"prep": "1"}

        # Create list of circuits with different xvals and preps
        circuits = []
        xvals = [0.0, 0.5, 1.0]
        for xval, _circ in product(xvals, [circ_prep_0, circ_prep_1]):
            circ = _circ.copy()
            circ.metadata["xval"] = xval
            circuits.append(circ)

        # Run circuits on mock backend
        job = backend.run(
            circuits,
            shots=num_shots,
            meas_level=MeasLevel.KERNELED,
            meas_return=MeasReturnType.SINGLE,
        )
        res = job.result()

        # Verify the number of circuits
        self.assertEqual(
            len(res.results),
            len(xvals) * 2,
            msg=f"Number of circuits does not match 2 * # xvals: {len(res.results)} instead of"
            f"{len(xvals)*2}.",
        )

        # Verify the number of shots
        for i_xval, i_prep in product(range(len(xvals)), range(2)):
            num_shots_for_i = len(res.results[i_xval * 2 + i_prep].data.memory)
            self.assertEqual(
                num_shots_for_i,
                num_shots,
                msg=f"Number of shots for xval={xvals[i_xval]} and prep={i_prep} is not as expected:"
                f"{num_shots_for_i} vs {num_shots}.",
            )

        # Verify the centers of the clusters
        expected_centers_per_xval = []
        for xval in xvals:
            # Add qubit 0 center for prepared state 0
            expected_centers_per_xval.append(np.array(bare_centers[0][0]) * xval)
            # Add qubit 0 center for prepared state 1
            expected_centers_per_xval.append(np.array(bare_centers[0][1]) * xval)

        for i, (expected_centers, data) in enumerate(zip(expected_centers_per_xval, res.results)):
            memory = np.array(data.data.memory)
            centers = np.squeeze(np.mean(memory, axis=0))
            # For I and Q values
            for axis, x, expected_x in zip(
                ["Real", "Imag"], centers.flatten(), expected_centers.flatten()
            ):
                self.assertAlmostEqual(
                    x,
                    expected_x,
                    msg=f"{axis} value was not correct for i={i} and xval={xvals[int(i/2)]}: {x} instead"
                    f" of {expected_x}.\nExpected center = {expected_centers}\nActual center ="
                    f"{centers}",
                    delta=delta,
                )

    def test_iq_helper_iq_widths(self):
        """
        Test IQ centroid centers and widths are modified by MockIQExperimentHelper.
        """
        delta = 0.005

        num_shots = 10000
        bare_widths = [0.1, 0.2]
        backend = MockIQBackend(
            MockIQReadoutAmplitudeHelper(
                alter_centers=False,
                iq_cluster_centers=[[(0, -1), (1, 1)]],
                iq_cluster_width=bare_widths,
            ),
        )

        # Create template circuits
        circ_prep_0 = QuantumCircuit(1, 1)
        circ_prep_0.measure(0, 0)
        circ_prep_0.metadata = {"prep": "0"}
        circ_prep_1 = QuantumCircuit(1, 1)
        circ_prep_1.x(0)
        circ_prep_1.measure(0, 0)
        circ_prep_1.metadata = {"prep": "1"}

        # Create list of circuits with different xvals and preps
        circuits = []
        xvals = [0.0, 0.5, 1.0]
        for xval, _circ in product(xvals, [circ_prep_0, circ_prep_1]):
            circ = _circ.copy()
            circ.metadata["xval"] = xval
            circuits.append(circ)

        # Run circuits on mock backend
        job = backend.run(
            circuits,
            shots=num_shots,
            meas_level=MeasLevel.KERNELED,
            meas_return=MeasReturnType.SINGLE,
        )
        res = job.result()

        # Verify the number of circuits
        self.assertEqual(
            len(res.results),
            len(xvals) * 2,
            msg=f"Number of circuits does not match 2 * # xvals: {len(res.results)} instead of"
            f"{len(xvals)*2}.",
        )

        # Verify the number of shots
        for i_xval, i_prep in product(range(len(xvals)), range(2)):
            num_shots_for_i = len(res.results[i_xval * 2 + i_prep].data.memory)
            self.assertEqual(
                num_shots_for_i,
                num_shots,
                msg=f"Number of shots for xval={xvals[i_xval]} and prep={i_prep} is not as expected:"
                f"{num_shots_for_i} vs {num_shots}.",
            )

        # Create list of expected widths per xval and prepared state
        expected_widths_per_xval = []
        for xval, _ in product(xvals, [0, 1]):
            # Add qubit 0 width for prepared state prep and xval
            expected_widths_per_xval.append(bare_widths[0] * xval)

        # Check the width for each circuit run (in data)
        for i, (expected_width, data) in enumerate(zip(expected_widths_per_xval, res.results)):
            # Get the actual width (I and Q)
            memory = np.array(data.data.memory)
            actual_widths = np.squeeze(np.std(memory, axis=0))

            # For the width in each dimension (I & Q)
            for axis, width in zip(["real", "imag"], actual_widths):
                self.assertAlmostEqual(
                    width,
                    expected_width,
                    msg=f"Width (std) along {axis} axis was not correct for i={i} and"
                    f"xval={xvals[int(i/2)]}: {width} instead of {expected_width}.",
                    delta=delta,
                )
