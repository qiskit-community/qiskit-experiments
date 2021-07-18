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
from qiskit.test.mock import FakeOpenPulse2Q

from qiskit.qobj.utils import MeasLevel
from qiskit.providers.options import Options
from qiskit_experiments.test.mock_job import MockJob


class MockIQBackend(FakeOpenPulse2Q):
    """An abstract backend for testing that can mock IQ data."""

    def __init__(
        self,
        iq_cluster_centers: Tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0),
        iq_cluster_width: float = 1.0,
    ):
        """
        Initialize the backend.
        """
        self._iq_cluster_centers = iq_cluster_centers
        self._iq_cluster_width = iq_cluster_width
        self._rng = np.random.default_rng(0)

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

        return MockJob(self, result)


class DragBackend(MockIQBackend):
    """A simple and primitive backend, to be run by the rough drag tests."""

    def __init__(
        self,
        iq_cluster_centers: Tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0),
        iq_cluster_width: float = 1.0,
        leakage: float = 0.03,
        ideal_beta=2.0,
    ):
        """Initialize the rabi backend."""
        self._leakage = leakage
        self.ideal_beta = ideal_beta

        super().__init__(iq_cluster_centers, iq_cluster_width)

    def _compute_probability(self, circuit: QuantumCircuit) -> float:
        """Returns the probability based on the beta, number of gates, and leakage."""
        n_gates = sum(circuit.count_ops().values())

        beta = next(iter(circuit.calibrations["Rp"].keys()))[1][0]

        return np.sin(n_gates * self._leakage * (beta - self.ideal_beta)) ** 2


class MockFineAmp(MockIQBackend):
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
        n_sx_ops = circuit.count_ops().get("sx", 0)
        n_x_ops = circuit.count_ops().get("x", 0)

        angle = n_ops * (self._angle_per_gate + self.angle_error)

        angle += np.pi / 2 * n_sx_ops
        angle += np.pi * n_x_ops

        return np.sin(angle / 2) ** 2
