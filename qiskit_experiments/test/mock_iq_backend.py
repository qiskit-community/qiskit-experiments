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
from typing import Dict, List, Tuple
import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers.backend import BackendV1 as Backend
from qiskit.test.mock import FakeOpenPulse2Q
from qiskit.providers import JobV1
from qiskit.result import Result
from qiskit.qobj.utils import MeasLevel
from qiskit.providers.options import Options


class TestJob(JobV1):
    """Job for testing."""

    def __init__(self, backend: Backend, result: Dict):
        """Setup a job for testing."""
        super().__init__(backend, "test-id")
        self._result = result

    def result(self) -> Result:
        """Return a result."""
        return Result.from_dict(self._result)

    def submit(self):
        pass

    def status(self):
        pass

    def cancel(self):
        pass


class IQTestBackend(FakeOpenPulse2Q):
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

        return TestJob(self, result)
