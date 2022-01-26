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
from typing import List, Tuple, Dict
import numpy as np
import random

from qiskit import QuantumCircuit, execute
from qiskit.result import Result
from qiskit.providers.aer import AerSimulator
from qiskit import BasicAer
from qiskit.test.mock import FakeOpenPulse2Q
from qiskit_experiments.framework.experiment_data import ExperimentData

from qiskit.qobj.utils import MeasLevel
from qiskit_experiments.framework import Options
from qiskit_experiments.test.utils import FakeJob


class RestlessSimulator:
    """A simulator of restless measurements experiments."""

    def __init__(
        self,
        shots: int = 1024,
    ):
        """
        Args:
            shots: The number of shots to simulate which defaults to 1024.
        """

        self._shots = shots
        self._sim = BasicAer.get_backend("statevector_simulator")
        self._precomputed_probabilities = None
        self._true_prev_outcome = None

        # The memory sorted according to circuit.
        self._sorted_memory = None

    @property
    def shots(self) -> int:
        """Return the shots used."""
        return self._shots

    @property
    def probabilities(self) -> Dict[Tuple[int, str], QuantumCircuit]:
        """Return the precomputed probabilities used in the simulation."""
        return self._precomputed_probabilities

    @staticmethod
    def state_strings(n_qubits: int) -> List[str]:
        """Generate all state strings for this system."""
        states, format_str = [], "{0:0" + str(n_qubits) + "b}"

        for state_num in range(2 ** n_qubits):
            states.append(format_str.format(state_num))

        return states

    def extend_circuit_set(
        self, in_circs: List[QuantumCircuit]
    ) -> Dict[Tuple[int, str], QuantumCircuit]:
        """Add all combinations of X gates.

        This function takes as input a list of quantum circuits and adds a quantum circuit
        for each possible initial state of the system.

        Args:
            in_circs: A list of input circuits to which X gates will be prepended to
                create all possible input states.
        Returns:
            circs: A list of circuits corresponding to the initial circuits with X
                gates to create the different restless_measurements input states.
        """

        circs, n_qubits = {}, in_circs[0].num_qubits

        for idx, qpt_circ in enumerate(in_circs):

            for state_str in self.state_strings(n_qubits):
                circ = QuantumCircuit(n_qubits, n_qubits)

                for bit_idx, bit in enumerate(state_str[::-1]):
                    if bit == "1":
                        circ.x(bit_idx)

                circ.compose(qpt_circ, inplace=True)
                circ.remove_final_measurements(inplace=True)

                circs[tuple([idx, state_str])] = circ

        return circs

    def compute_outcome_probabilities(self, circuits: Dict[Tuple[int, str], QuantumCircuit]):
        """Run the simulation to pre-compute the probabilities."""

        self._precomputed_probabilities = {}

        for key, circ in circuits.items():
            state_vec = execute(circ, self._sim).result().get_statevector(0)
            self._precomputed_probabilities[key] = np.abs(state_vec) ** 2

    def __call__(self, circuits: List[QuantumCircuit]) -> List[str]:
        """Restless simulation"""

        all_circs = self.extend_circuit_set(circuits)
        self.compute_outcome_probabilities(all_circs)

        n_qubits = circuits[0].num_qubits

        memory, prev_outcome = [], "0" * n_qubits
        self._true_prev_outcome = [prev_outcome]

        # Setup the list of dicts where each dict corresponds to a circuit.
        self._sorted_memory = [{"memory": [], "metadata": circ.metadata} for circ in circuits]

        for shot in range(self._shots):
            for circ_idx, _ in enumerate(circuits):

                probs = self._precomputed_probabilities[(circ_idx, prev_outcome)]

                outcome = random.choices(self.state_strings(n_qubits), probs)[0]

                memory.append(outcome)

                self._sorted_memory[circ_idx]["memory"].append(hex(int(outcome, 2)))

                prev_outcome = outcome

        return memory

    @property
    def experiment_data(self) -> ExperimentData:
        """Encapsulate the memory in the experiment data."""

        exp_data = ExperimentData()
        exp_data.add_data(self._sorted_memory)
        return exp_data


class MockRestlessBackend(FakeOpenPulse2Q):
    """An abstract backend for testing that can mock restless data."""

    def __init__(
        self,
        rng_seed: int = 0,
    ):
        """
        Initialize the backend.
        """
        self._rng = np.random.default_rng(rng_seed)
        super().__init__()

    def _default_options(self):
        """Default options of the test backend."""
        return Options(
            shots=1024,
            meas_level=MeasLevel.CLASSIFIED,
            meas_return="single",
        )

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

    def _apply_no_state_change(self, prev_shot: str, shot: str, probability: float) -> str:
        """Probabilistic modification of measurement shot state changes to no state changes.

        We choose a number uniformly from [0, 1] and if it is larger than
        a given probability and there was a state change compared to the previous shot we
        modify the shot such that it is equivalent to the previous one.

        Args:
            prev_shot: The previously measured bitstring.
            shot: The measured bitsting that can be modified.
            probability: The probability with which a state change is modified to a no
            state change.

        Returns:
            The modified or not modified shot.
        """

        if prev_shot != shot:
            if self._rng.uniform(0, 1) >= probability:
                shot = prev_shot

        return shot

    def run(self, run_input, **options):
        """Run the IQ backend."""

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

        # List of probabilities for each circuit and each shot with length
        # len(run_input) * shots.
        probs = [self._compute_probability(circ) for circ in run_input] * shots

        # Initialize the restless simulator.
        sim = RestlessSimulator(shots=shots)
        # Create time-ordered perfect restless shots for the experiment circuits.
        memory = sim(run_input)

        # Modify the restless shots in the memory dependent on the pre-computed
        # probabilities (self._compute_probability(circ)) for each circuit. This
        # will generate "no state change"s in the time-ordered memory dependent
        # on the probability. E.g., if the pre-computed probability for one circuit
        # is low (which is equivalent to that P(1) is small, i.e., errors occurred)
        # it will be very likely that the shots corresponding to this circuit will
        # be modified if there was a state change compared to the previous shot. This
        # will lead to a reduced P(no state change) which is equivalent to a small
        # P(1) in the standard setting.
        for idx in range(1, len(memory)):
            modified_shot = self._apply_no_state_change(memory[idx - 1], memory[idx], probs[idx])
            memory[idx] = modified_shot

        memory = [memory[i : i + len(run_input)] for i in range(0, len(memory), len(run_input))]

        sorted_memory = [{"memory": []} for _ in range(len(run_input))]
        # Group the restless shots in the time-ordered list by circuit.
        for circ_idx, _ in enumerate(run_input):
            for shot in range(shots):
                sorted_memory[circ_idx]["memory"].append(hex(int(memory[shot][circ_idx], 2)))

        for idx, circ in enumerate(run_input):
            ones = sorted_memory[idx]["memory"].count("0x1")
            run_result = {
                "shots": shots,
                "success": True,
                "header": {"metadata": circ.metadata},
                "meas_level": meas_level,
                "data": {
                    "counts": {"1": ones, "0": shots - ones},
                    "memory": sorted_memory[idx]["memory"],
                },
            }

            result["results"].append(run_result)

        return FakeJob(self, Result.from_dict(result))


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

    def _draw_iq_shots(self, prob, shots) -> List[List[List[float]]]:
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

    def run(self, run_input, **options):
        """Run the IQ backend."""

        self.options.update_options(**options)
        shots = self.options.get("shots")
        meas_level = self.options.get("meas_level")
        meas_return = self.options.get("meas_return")

        result = {
            "backend_name": f"{self.__class__.__name__}",
            "backend_version": "0",
            "qobj_id": 0,
            "job_id": 0,
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
                memory = self._draw_iq_shots(prob, shots)

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

    def _compute_probability(self, circuit: QuantumCircuit) -> float:
        """Returns the probability based on the beta, number of gates, and leakage."""
        n_gates = sum(circuit.count_ops().values())

        beta = next(iter(circuit.calibrations[self._gate_name].keys()))[1][0]

        return np.sin(n_gates * self._error * (beta - self.ideal_beta)) ** 2


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


class MockRestlessFineAmp(MockRestlessBackend):
    """A mock backend for restless fine amplitude calibration."""

    def __init__(self, angle_error: float, angle_per_gate: float, gate_name: str):
        """Setup a mock backend to test the restless fine amplitude calibration.

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
