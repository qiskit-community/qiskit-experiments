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
from typing import List, Tuple
import numpy as np

from qiskit import QuantumCircuit
from qiskit.result import Result

from qiskit.providers.aer import AerSimulator
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
        iq_cluster_centers: Tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0),
        iq_cluster_width: float = 1.0,
        rng_seed: int = 0,
    ):
        """
        Initialize the backend.
        """
        self._iq_cluster_centers = iq_cluster_centers
        self._iq_cluster_width = iq_cluster_width
        self._rng = np.random.default_rng(rng_seed)

        super().__init__()

    def _default_options(self):
        """Default options of the test backend."""
        return Options(
            shots=1024,
            meas_level=MeasLevel.KERNELED,
            meas_return="single",
        )

    def _draw_iq_shots(self, prob, shots, phase: float = 0.0) -> List[List[List[float]]]:
        """Produce an IQ shot."""

        rand_i = self._rng.normal(0, self._iq_cluster_width, size=shots)
        rand_q = self._rng.normal(0, self._iq_cluster_width, size=shots)

        memory = []
        for idx, state in enumerate(self._rng.binomial(1, prob, size=shots)):

            if state > 0.5:
                point_i = self._iq_cluster_centers[0] + rand_i[idx]
                point_q = self._iq_cluster_centers[1] + rand_q[idx]
            else:
                point_i = self._iq_cluster_centers[2] + rand_i[idx]
                point_q = self._iq_cluster_centers[3] + rand_q[idx]

            if not np.allclose(phase, 0.0):
                complex_iq = (point_i + 1.0j * point_q) * np.exp(1.0j * phase)
                point_i, point_q = np.real(complex_iq), np.imag(complex_iq)

            memory.append([[point_i, point_q]])

        return memory

    @abstractmethod
    def _compute_probability(self, circuit: QuantumCircuit) -> float:
        """Compute the probability used in the binomial distribution creating the IQ shot.

        An abstract method that subclasses will implement to create a probability of
        being in the excited state based on the received quantum circuit.

        Args:
            circuit: The circuit from which to compute the probability.

        Returns:
             The probability that the binaomial distribution will use to generate an IQ shot.
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
            run_result = {
                "shots": shots,
                "success": True,
                "header": {"metadata": circ.metadata},
                "meas_level": meas_level,
            }

            prob = self._compute_probability(circ)

            if meas_level == MeasLevel.CLASSIFIED:
                ones = np.sum(self._rng.binomial(1, prob, size=shots))
                run_result["data"] = {"counts": {"1": ones, "0": shots - ones}}
            else:
                phase = self._iq_phase(circ)
                memory = self._draw_iq_shots(prob, shots, phase)

                if meas_return == "avg":
                    memory = np.average(np.array(memory), axis=0).tolist()

                run_result["data"] = {"memory": memory}

            result["results"].append(run_result)

        return FakeJob(self, Result.from_dict(result))


class DragBackend(MockIQBackend):
    """A simple and primitive backend, to be run by the rough drag tests."""

    def __init__(
        self,
        iq_cluster_centers: Tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0),
        iq_cluster_width: float = 1.0,
        freq: float = 0.02,
        ideal_beta=2.0,
        gate_name: str = "Rp",
        rng_seed: int = 0,
        max_prob: float = 1.0,
        offset_prob: float = 0.0,
    ):
        """Initialize the rabi backend."""
        self._freq = freq
        self._gate_name = gate_name
        self.ideal_beta = ideal_beta

        if max_prob + offset_prob > 1:
            raise ValueError("Probabilities need to be between 0 and 1.")

        self._max_prob = max_prob
        self._offset_prob = offset_prob

        super().__init__(iq_cluster_centers, iq_cluster_width, rng_seed=rng_seed)

    def _compute_probability(self, circuit: QuantumCircuit) -> float:
        """Returns the probability based on the beta, number of gates, and leakage."""
        n_gates = circuit.count_ops()[self._gate_name]

        beta = next(iter(circuit.calibrations[self._gate_name].keys()))[1][0]

        prob = np.sin(2 * np.pi * n_gates * self._freq * (beta - self.ideal_beta) / 4) ** 2
        rescaled_prob = self._max_prob * prob + self._offset_prob

        return rescaled_prob


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
