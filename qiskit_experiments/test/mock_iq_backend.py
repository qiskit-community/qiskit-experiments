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
from typing import List, Tuple, Dict, Union
import numpy as np

from qiskit import QuantumCircuit
from qiskit.result import Result
# from qiskit.providers.aer import AerSimulator
from qiskit.test.mock import FakeOpenPulse2Q

from qiskit.qobj.utils import MeasLevel
from qiskit_experiments.framework import Options
from qiskit_experiments.test.utils import FakeJob


class MockIQBackend(FakeOpenPulse2Q):
    """An abstract backend for testing that can mock IQ data."""

    def __init__(
        self,
        iq_cluster_centers: List[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        iq_cluster_width: List[float] = None,
        rng_seed: int = 0,
    ):
        """
        Initialize the backend.
        """

        self._rng = np.random.default_rng(rng_seed)

        if iq_cluster_centers is None:
            self._iq_cluster_centers = [((1.0, 1.0), (-1.0, -1.0))]
        else:
            self._iq_cluster_centers = iq_cluster_centers

        if iq_cluster_width is None:
            self._iq_cluster_width = [1.0]
        else:
            self._iq_cluster_width = iq_cluster_width

        super().__init__()

    def _default_options(self):
        """Default options of the test backend."""
        return Options(
            shots=1024,
            meas_level=MeasLevel.KERNELED,
            meas_return="single",
        )

    @staticmethod
    def _verify_parameters(num_qubits):
        if num_qubits < 1:
            raise ValueError(f"The number of qubits {num_qubits} is smaller than 1.")
        # TODO:
        # check that the length of attributes matches the number of qubits.
        # check that probability is 1.


    def _get_normal_samples_for_shot(self, num_qubits: int):
        """
        Produce a list in the size of num_qubits. Each entry value produced from normal distribution with expected value
        of '0' and standard deviation of self._iq_cluster_width.
        Args:
            num_qubits(int): The amount of qubits in the circuit.

        Returns:
            List[float]: A list with values that were produced from normal distribution.
        """
        widths = self._iq_cluster_width
        samples = [self._rng.normal(0, widths[qubit], size=1) for qubit in range(num_qubits)]
        return samples

    @staticmethod
    def _values_to_string_array(num_qubits):
        """
        This function creates a dictionary in the size of num_qubits ** 2 (all possible values)
        that connects between a number and its full binary representation as string with length of
        num_qubits.
        Args:
            num_qubits(int): The number of qubits in the circuit.
        Returns:
            dict: A dictionary that connects between a value to its string representation.
        """
        max_value = num_qubits ** 2
        return_dict = {}
        for num in range(max_value):
            num_in_binary = format(num, "b")
            qubit_string_value = ""
            for _ in range(num_qubits - len(num_in_binary)):
                qubit_string_value += "0"
            qubit_string_value += str(num_in_binary)
            return_dict[num] = qubit_string_value
        return return_dict

    def _expand_probability(self, probability: Dict[str, float], num_qubits: int):
        """
        Take a dictionary of probabilities and expand it to include trivial cases with
        probability of 0.
        Args:
            probability(dict): A dictionary that it's keys are strings
            num_qubits:

        Returns:

        """
        value2str = self._values_to_string_array(num_qubits)
        for _, qubit_string_value in value2str.items():
            if qubit_string_value in probability:
                continue
            else:
                probability[qubit_string_value] = 0

    def _draw_iq_shots(
        self, prob: Dict[str, float], shots: int, num_qubits: int, phase: float = 0.0
    ) -> List[List[Tuple[float, float]]]:
        """
        Produce an IQ shot.
        Args:
            prob(dict): A dictionary that the keys are output string and the value is the probability.
            shots(int): The number of times the circuit will run.
            num_qubits(int): The number of qubit in hte circuit.

        Returns:
            List[List[Tuple[float, float]]]: return a list that each entry is a list that represent a shot. The output
            is build as following - List[shot index][qubit index] = [I,Q]
        """
        # Randomize samples
        qubits_iq_rand = []
        for shot_index in range(shots):
            rand_i = np.squeeze(np.array(self._get_normal_samples_for_shot(num_qubits)))
            rand_q = np.squeeze(np.array(self._get_normal_samples_for_shot(num_qubits)))
            qubits_iq_rand.append(np.array([rand_i, rand_q], dtype="float").T)

        # For multinomial, the probabilities is given in list for each outcome.
        # hence, np.log2(len(prob)) = num_qubits
        if np.log2(len(prob)) != num_qubits:
            raise ValueError("The probability provided doesn't match all cases possible.")

        val2str_dict = self._values_to_string_array(num_qubits)
        memory = []
        shot_num = 0
        iq_centers = self._iq_cluster_centers

        for idx, number_of_occurrences in enumerate(self._rng.multinomial(1, prob, size=shots)):
            # For multiple qubit - translate number to string
            # and then count them.
            # need to think about the structure of probability.
            state_str = val2str_dict[idx]
            for _ in range(number_of_occurrences):
                shot_memory = []
                for qubit_number, char_qubit in enumerate(state_str):
                    # the iteration on the str starts from the MSB so we will use a variable to
                    # make the code more readable.
                    current_qubit = num_qubits - qubit_number - 1
                    # The structure of iq_centers is [qubit_number][logic_result][I/Q].
                    i_center = iq_centers[current_qubit][int(char_qubit)][0]
                    q_center = iq_centers[current_qubit][int(char_qubit)][1]
                    point_i = i_center + qubits_iq_rand[shot_num][qubit_number]
                    point_q = q_center + qubits_iq_rand[shot_num][qubit_number]
                    
                    # Adding phase if not 0.0
                    if not np.allclose(phase, 0.0):
                        complex_iq = (point_i + 1.0j * point_q) * np.exp(1.0j * phase)
                        point_i, point_q = np.real(complex_iq), np.imag(complex_iq)
                    
                    shot_memory.append([point_i, point_q])
                # We proceed to the next occurrence - meaning its a new shot.
                memory.append(shot_memory)
                shot_num += 1

        return memory

    def _generate_data(self, prob: Dict[str, float], num_qubits: int, circuit: QuantumCircuit) -> Dict:
        """

        Args:
            prob(dict):
            num_qubits:

        Returns:

        """
        # Maybe I need to get as input for generalization
        shots = self.options.get("shots")
        meas_level = self.options.get("meas_level")
        meas_return = self.options.get("meas_return")
        run_result = {}

        val2str_dict = self._values_to_string_array(num_qubits)
        if meas_level == MeasLevel.CLASSIFIED:
            counts = {}
            results = self._rng.multinomial(shots, prob, size=1)
            for result, num_occurrences in enumerate(results):
                result_in_str = val2str_dict["result"]
                counts[result_in_str] = num_occurrences
            run_result["counts"] = counts
        else:
            phase = self._iq_phase(circuit)
            memory = self._draw_iq_shots(prob, shots, num_qubits, phase)
            if meas_return == "avg":
                memory = np.average(np.array(memory), axis=0).tolist()  # could have a bug here

            run_result["memory"] = memory
        return run_result

    @abstractmethod
    def _compute_probability(self, circuit: QuantumCircuit) -> Dict[str, float]:
        """Compute the probability used in the binomial distribution creating the IQ shot.

        An abstract method that subclasses will implement to create a probability of
        being in the excited state based on the received quantum circuit.

        Args:
            circuit: The circuit from which to compute the probability.

        Returns:
             The probability that the multinomial distribution will use to generate an IQ shot.
        """

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
        meas_return = self.options.get("meas_return")

        result = {
            "backend_name": f"{self.__class__.__name__}",
            "backend_version": "0",
            "qobj_id": "0",
            "job_id": "0",
            "success": True,
            "results": [],
        }

        for circ in run_input:
            nqubits = circ.num_qubits
            run_result = {
                "shots": shots,
                "success": True,
                "header": {"metadata": circ.metadata},
                "meas_level": meas_level,
            }

            prob = self._compute_probability(circ)
            run_result["data"] = self._generate_data(prob, nqubits, circ)
            result["results"].append(run_result)

        return FakeJob(self, Result.from_dict(result))


class DragBackend(MockIQBackend):
    """A simple and primitive backend, to be run by the rough drag tests."""

    def __init__(
        self,
        iq_cluster_centers: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [((1.0, 1.0), (-1.0, -1.0))],
        iq_cluster_width: List[float] = [1.0],
        error: float = 0.03,
        ideal_beta=2.0,
        gate_name: str = "Rp",
        rng_seed: int = 0,
    ):
        """Initialize the rabi backend."""
        self._error = error
        self._gate_name = gate_name
        self.ideal_beta = ideal_beta

        super().__init__(iq_cluster_centers, iq_cluster_width, rng_seed=rng_seed)

    def _compute_probability(self, circuit: QuantumCircuit) -> Dict[str, float]:
        """Returns the probability based on the beta, number of gates, and leakage."""
        probability_output_dict = {}
        # Need to change that the output will be dict. Need to see what the circuit do.
        n_gates = sum(circuit.count_ops().values())
        beta = next(iter(circuit.calibrations[self._gate_name].keys()))[1][0]

        # Dictionary of output  vectors and there probability
        probability_output_dict["1"] = np.sin(n_gates * self._error * (beta - self.ideal_beta)) ** 2
        probability_output_dict["0"] = 1 - probability_output_dict["1"]
        return probability_output_dict


class RabiBackend(MockIQBackend):
    """A simple and primitive backend, to be run by the Rabi tests."""

    def __init__(
        self,
        iq_cluster_centers: Tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0),
        iq_cluster_width: float = 1.0,
        amplitude_to_angle: float = np.pi,
    ):
        """Initialize the rabi backend."""
        self._amplitude_to_angle = amplitude_to_angle

        super().__init__(iq_cluster_centers, iq_cluster_width)

    @property
    def rabi_rate(self) -> float:
        """Returns the rabi rate."""
        return self._amplitude_to_angle / np.pi

    def _compute_probability(self, circuit: QuantumCircuit) -> float:
        """Returns the probability based on the rotation angle and amplitude_to_angle."""
        amp = next(iter(circuit.calibrations["Rabi"].keys()))[1][0]
        return np.sin(self._amplitude_to_angle * amp) ** 2


class MockFineAmp(MockIQBackend):
    """A mock backend for fine amplitude calibration."""

    def __init__(self, angle_error: float, angle_per_gate: float, gate_name: str):
        """Setup a mock backend to test the fine amplitude calibration.

        Args:
            angle_error: The rotation error per gate.
            gate_name: The name of the gate to find in the circuit.
        """
        self.angle_error = angle_error
        self._gate_name = gate_name
        self._angle_per_gate = angle_per_gate
        super().__init__()

        self.configuration().basis_gates.append("sx")
        self.configuration().basis_gates.append("x")

    def _compute_probability(self, circuit: QuantumCircuit) -> float:
        """Return the probability of being in the excited state."""

        n_ops = circuit.count_ops().get(self._gate_name, 0)
        angle = n_ops * (self._angle_per_gate + self.angle_error)

        if self._gate_name != "sx":
            angle += np.pi / 2 * circuit.count_ops().get("sx", 0)

        if self._gate_name != "x":
            angle += np.pi * circuit.count_ops().get("x", 0)

        return np.sin(angle / 2) ** 2


class MockFineFreq(MockIQBackend):
    """A mock backend for fine frequency calibration."""

    def __init__(self, freq_shift: float, sx_duration: int = 160):
        super().__init__()
        self.freq_shift = freq_shift
        self.dt = self.configuration().dt
        self.sx_duration = sx_duration
        self.simulator = AerSimulator(method="automatic")

    def _compute_probability(self, circuit: QuantumCircuit) -> float:
        """The freq shift acts as the value that will accumulate phase."""

        delay = None
        for instruction in circuit.data:
            if instruction[0].name == "delay":
                delay = instruction[0].duration

        if delay is None:
            return 1.0
        else:
            reps = delay // self.sx_duration

            qc = QuantumCircuit(1)
            qc.sx(0)
            qc.rz(np.pi * reps / 2 + 2 * np.pi * self.freq_shift * delay * self.dt, 0)
            qc.sx(0)
            qc.measure_all()

            counts = self.simulator.run(qc, seed_simulator=1).result().get_counts(0)

            return counts.get("1", 0) / sum(counts.values())


class MockRamseyXY(MockIQBackend):
    """A mock backend for the RamseyXY experiment."""

    def __init__(self, freq_shift: float):
        super().__init__()
        self.freq_shift = freq_shift

    def _compute_probability(self, circuit: QuantumCircuit) -> float:
        """Return the probability of the circuit."""

        series = circuit.metadata["series"]
        delay = circuit.metadata["xval"]

        if series == "X":
            phase_offset = 0.0
        else:
            phase_offset = np.pi / 2

        return 0.5 * np.cos(2 * np.pi * delay * self.freq_shift - phase_offset) + 0.5
