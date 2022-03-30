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

"""An mock IQ backend for testing."""

from abc import abstractmethod
from typing import List, Tuple, Dict, Union, Any, Optional, Callable
import numpy as np

from qiskit import QuantumCircuit
from qiskit.result import Result
from qiskit.test.mock import FakeOpenPulse2Q

from qiskit.qobj.utils import MeasLevel
from qiskit_experiments.framework import Options
from qiskit_experiments.test.utils import FakeJob
from qiskit_experiments.data_processing.exceptions import DataProcessorError


class MockRestlessBackend(FakeOpenPulse2Q):
    """An abstract backend for testing that can mock restless data."""

    def __init__(self, rng_seed: int = 0):
        """
        Initialize the backend.
        """
        self._rng = np.random.default_rng(rng_seed)
        self._precomputed_probabilities = None
        super().__init__()

    def _default_options(self):
        """Default options of the test backend."""
        return Options(
            shots=1024,
            meas_level=MeasLevel.CLASSIFIED,
            meas_return="single",
        )

    @staticmethod
    def _get_state_strings(n_qubits: int) -> List[str]:
        """Generate all state strings for the system."""
        format_str = "{0:0" + str(n_qubits) + "b}"
        return list(format_str.format(state_num) for state_num in range(2**n_qubits))

    @abstractmethod
    def _compute_outcome_probabilities(self, circuits: List[QuantumCircuit]):
        """Compute the probabilities of measuring 0 or 1 for each of the given
         circuits based on the previous measurement shot.

        This methods computes the dictionary self._precomputed_probabilities where
        the keys are a tuple consisting of the circuit index and the previous outcome,
        e.g. "0" or "1" for a single qubit. The values are the corresponding probabilities.

        Args:
            circuits: The circuits from which to compute the probabilities.
        """

    def run(self, run_input, **options):
        """Run the restless backend."""

        self.options.update_options(**options)
        shots = self.options.get("shots")
        meas_level = self.options.get("meas_level")

        result = {
            "backend_name": f"{self.__class__.__name__}",
            "backend_version": "0",
            "qobj_id": 0,
            "job_id": 0,
            "success": True,
            "results": [],
        }

        self._compute_outcome_probabilities(run_input)

        if run_input[0].num_qubits != 2:
            raise DataProcessorError(f"{self.__class__.__name__} is a two qubit mock device.")

        prev_outcome, state_strings = "00", self._get_state_strings(2)

        # Setup the list of dicts where each dict corresponds to a circuit.
        sorted_memory = [{"memory": [], "metadata": circ.metadata} for circ in run_input]

        for _ in range(shots):
            for circ_idx, _ in enumerate(run_input):
                probs = self._precomputed_probabilities[(circ_idx, prev_outcome)]
                # Generate the next shot dependent on the pre-computed probabilities.
                outcome = self._rng.choice(state_strings, p=probs)
                # Append the single shot to the memory of the corresponding circuit.
                sorted_memory[circ_idx]["memory"].append(hex(int(outcome, 2)))

                prev_outcome = outcome

        for idx, circ in enumerate(run_input):
            counts = {}
            for key1, key2 in zip(["00", "01", "10", "11"], ["0x0", "0x1", "0x2", "0x3"]):
                counts[key1] = sorted_memory[idx]["memory"].count(key2)
            run_result = {
                "shots": shots,
                "success": True,
                "header": {"metadata": circ.metadata},
                "meas_level": meas_level,
                "data": {
                    "counts": counts,
                    "memory": sorted_memory[idx]["memory"],
                },
            }

            result["results"].append(run_result)

        return FakeJob(self, Result.from_dict(result))


class MockRestlessFineAmp(MockRestlessBackend):
    """A mock backend for restless single-qubit fine amplitude experiments."""

    def __init__(
        self, angle_error: float, angle_per_gate: float, gate_name: str, rng_seed: int = 0
    ):
        """Setup a mock backend to test the restless fine amplitude calibration.

        Args:
            angle_error: The rotation error per gate.
            angle_per_gate: The angle per gate.
            gate_name: The name of the gate to find in the circuit.
            rng_seed: The random bit generator seed.
        """
        self.angle_error = angle_error
        self._gate_name = gate_name
        self._angle_per_gate = angle_per_gate
        super().__init__(rng_seed=rng_seed)

        self.configuration().basis_gates.extend(["sx", "x"])

    def _compute_outcome_probabilities(self, circuits: List[QuantumCircuit]):
        """Compute the probabilities of being in the excited state or
        ground state for all circuits."""

        self._precomputed_probabilities = {}

        for idx, circuit in enumerate(circuits):

            n_ops = circuit.count_ops().get(self._gate_name, 0)
            angle = n_ops * (self._angle_per_gate + self.angle_error)

            if self._gate_name != "sx":
                angle += np.pi / 2 * circuit.count_ops().get("sx", 0)

            if self._gate_name != "x":
                angle += np.pi * circuit.count_ops().get("x", 0)

            prob_1 = np.sin(angle / 2) ** 2
            prob_0 = 1 - prob_1

            self._precomputed_probabilities[(idx, "00")] = [prob_0, prob_1, 0, 0]
            self._precomputed_probabilities[(idx, "01")] = [prob_1, prob_0, 0, 0]


class MockIQBackend(FakeOpenPulse2Q):
    """An abstract backend for testing that can mock IQ data."""

    def __init__(
        self,
        compute_probabilities: Callable[[List[QuantumCircuit], ...], List[Dict[str, float]]],
        rng_seed: int = 0,
        iq_cluster_centers: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
        iq_cluster_width: Optional[List[float]] = None,
        calculation_parameters: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize the backend.
        Args:
            compute_probabilities(Callable): A function that the user provides to calculate the
            probability of each output of the circuit. The user passes the list of circuits and any
            variable that he needs to calculatethe probability. The function return a list of
            dictionaries. Each dictionary in the output list
            corresponds to the probabilities of output vectors for the circuit with the same index in the
            circuit list.
            rng_seed(int): The random seed value.
            iq_cluster_centers(Optional[List]): A list of tuples containing the clusters' centers in the
            IQ plane.
            There are different centers for different logical values of the qubit.
            iq_cluster_width(Optional[List]): A list of standard deviation values for the sampling of
            each qubit.
            calculation_parameters(Optional[List]): A dictionary with parameters that are needed for the
            function 'compute_probabilities'.
        """

        self._iq_cluster_centers = iq_cluster_centers or [((-1.0, -1.0), (1.0, 1.0))]
        self._iq_cluster_width = iq_cluster_width or [1.0]
        self._calculation_parameters = calculation_parameters
        self._compute_probabilities = compute_probabilities
        self._rng = np.random.default_rng(rng_seed)

        super().__init__()

    def _default_options(self):
        """Default options of the test backend."""
        return Options(
            shots=1024,
            meas_level=MeasLevel.KERNELED,
            meas_return="single",
        )

    def set_calculation_parameters(self, calculation_parameters: List[Dict[str, Any]]):
        """
        A function to set calculation parameters for the compute_probabilities function.
        Args:
            calculation_parameters(Dict): A dictionary with parameters that will replace the existing
            parameters of the backend.
        """
        self._calculation_parameters = calculation_parameters

    @staticmethod
    def _verify_parameters(num_qubits: int, prob_dict: Dict[str, float]):
        if num_qubits < 1:
            raise ValueError(f"The number of qubits {num_qubits} is smaller than 1.")

        if not np.allclose(1, sum(prob_dict.values())):
            raise ValueError("The probabilities given don't sum up to 1.")
        for key in prob_dict.keys():
            if num_qubits is not len(key):
                raise ValueError(
                    "The number of qubits and the length of the output string don't match."
                )
        # TODO:
        # check that the length of attributes matches the number of qubits.
        # check that probability is 1.

    def _get_normal_samples_for_shot(self, num_qubits: int):
        """
        Produce a list in the size of num_qubits. Each entry value is produced from normal distribution
        with expected value of '0' and standard deviation of self._iq_cluster_width.
        Args:
            num_qubits(int): The amount of qubits in the circuit.

        Returns:
            List[float]: A list with values that were produced from normal distribution.
        """
        widths = self._iq_cluster_width
        samples = [self._rng.normal(0, widths[qubit], size=1) for qubit in range(num_qubits)]
        # we squeeze the second dimension because samples is List[qubit_number][0][0\1] = I\Q
        # and we want to change it to be List[qubit_number][0\1]
        return np.squeeze(np.array(samples), axis=1)

    def _probability_dict_to_probability_array(
        self, prob_dict: Dict[str, float], num_qubits: int
    ) -> List[float]:
        prob_list = []
        for num in range(2**num_qubits):
            num_binary_str = str(format(num, "b").zfill(num_qubits))
            prob_list.append(prob_dict.get(num_binary_str, 0))
        return prob_list

    def _draw_iq_shots(
        self, prob: List[float], shots: int, num_qubits: int, phase: float = 0.0
    ) -> List[List[List[Union[float, complex]]]]:
        """
        Produce an IQ shot.
        Args:
            prob(List): A list of probabilities for each output.
            shots(int): The number of times the circuit will run.
            num_qubits(int): The number of qubits in the circuit.

        Returns:
            List[List[Tuple[float, float]]]: A list of shots. Each shot composed of a list of qubits.
            The qubits are tuples with two values [I,Q].
            The output structure is  - List[shot index][qubit index] = [I,Q]
        """
        # Randomize samples
        qubits_iq_rand = []
        for _ in range(shots):
            rand_i = self._get_normal_samples_for_shot(num_qubits)
            rand_q = self._get_normal_samples_for_shot(num_qubits)
            qubits_iq_rand.append(np.array([rand_i, rand_q], dtype="float").T)

        memory = []
        shot_num = 0
        iq_centers = self._iq_cluster_centers

        for output_number, number_of_occurrences in enumerate(
            self._rng.multinomial(shots, prob, size=1)[0]
        ):
            state_str = str(format(output_number, "b").zfill(num_qubits))
            for _ in range(number_of_occurrences):
                shot_memory = []
                # the iteration on the string variable state_str starts from the MSB. For readability,
                # we will reverse the string so the loop will run from the LSB to MSB.
                for iq_center, qubit_iq_rand_sample, char_qubit in zip(
                    iq_centers, qubits_iq_rand[shot_num], state_str[::-1]
                ):
                    # The structure of iq_centers is [qubit_number][logic_result][I/Q].
                    i_center = iq_center[int(char_qubit)][0]
                    q_center = iq_center[int(char_qubit)][1]

                    point_i = i_center + qubit_iq_rand_sample[0]
                    point_q = q_center + qubit_iq_rand_sample[1]

                    # Adding phase if not 0.0
                    if not np.allclose(phase, 0.0):
                        complex_iq = (point_i + 1.0j * point_q) * np.exp(1.0j * phase)
                        point_i, point_q = np.real(complex_iq), np.imag(complex_iq)

                    shot_memory.append([point_i, point_q])
                # We proceed to the next occurrence - meaning it's a new shot.
                memory.append(shot_memory)
                shot_num += 1

        return memory

    def _generate_data(
        self, prob_dict: Dict[str, float], num_qubits: int, circuit: QuantumCircuit
    ) -> Dict[str, Any]:
        """
        Generate data for the circuit.
        Args:
            prob_dict(dict): A dictionary whose keys are strings representing the output vectors and
            their values are the probability to get the output in this circuit.
            num_qubits(int): The number of qubits.
            circuit(QuantumCircuit): The circuit that needs to be simulated.

        Returns:
            A dictionary that filled with the simulated data. The output format is different between
            measurement level 1 and measurement level 2.
        """
        self._verify_parameters(num_qubits, prob_dict)
        prob_arr = self._probability_dict_to_probability_array(prob_dict, num_qubits)
        shots = self.options.get("shots")
        meas_level = self.options.get("meas_level")
        meas_return = self.options.get("meas_return")
        run_result = {}

        if meas_level == MeasLevel.CLASSIFIED:
            counts = {}
            results = self._rng.multinomial(shots, prob_arr, size=1)[0]
            for result, num_occurrences in enumerate(results):
                result_in_str = str(format(result, "b").zfill(num_qubits))
                counts[result_in_str] = num_occurrences
            run_result["counts"] = counts
        else:
            phase = self._iq_phase(circuit)
            memory = self._draw_iq_shots(prob_arr, shots, num_qubits, phase)
            if meas_return == "avg":
                memory = np.average(np.array(memory), axis=0).tolist()  # could have a bug here

            run_result["memory"] = memory
        return run_result

    # pylint: disable=unused-argument
    def _iq_phase(self, circuit: QuantumCircuit) -> float:
        """Sub-classes can override this method to introduce a phase in the IQ plan.

        This is needed, to test the resonator spectroscopy where the point in the IQ
        plan has a frequency-dependent phase rotation.
        """
        return 0.0

    def run(self, run_input, **options):
        """Run the IQ backend."""

        self.options.update_options(**options)
        shots = self.options.get("shots")
        meas_level = self.options.get("meas_level")

        result = {
            "backend_name": f"{self.__class__.__name__}",
            "backend_version": "0",
            "qobj_id": "0",
            "job_id": "0",
            "success": True,
            "results": [],
        }
        if self._calculation_parameters:
            prob_list = self._compute_probabilities(run_input, self._calculation_parameters)
        else:
            prob_list = self._compute_probabilities(run_input)
        for prob, circ in zip(prob_list, run_input):
            # nqubits = circ.num_qubits
            nqubits = len(circ.qregs)
            run_result = {
                "shots": shots,
                "success": True,
                "header": {"metadata": circ.metadata},
                "meas_level": meas_level,
            }

            run_result["data"] = self._generate_data(prob, nqubits, circ)
            result["results"].append(run_result)

        return FakeJob(self, Result.from_dict(result))
