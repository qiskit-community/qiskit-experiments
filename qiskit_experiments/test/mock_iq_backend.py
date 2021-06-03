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

from typing import Dict, List, Tuple
import numpy as np

from qiskit.providers.backend import BackendV1 as Backend
from qiskit.providers import JobV1
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result


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


class IQTestBackend(Backend):
    """An abstract backend for testing that can mock IQ data."""

    __configuration__ = {
        "backend_name": "simulator",
        "backend_version": "0",
        "n_qubits": int(1),
        "basis_gates": [],
        "gates": [],
        "local": True,
        "simulator": True,
        "conditional": False,
        "open_pulse": False,
        "memory": True,
        "max_shots": int(1e6),
        "coupling_map": [],
        "dt": 0.1,
    }

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

        super().__init__(QasmBackendConfiguration(**self.__configuration__))

    def _default_options(self):
        """Default options of the test backend."""

    def _draw_iq_shot(self, prob) -> List[List[float]]:
        """Produce an IQ shot."""

        rand_i = self._rng.normal(0, self._iq_cluster_width)
        rand_q = self._rng.normal(0, self._iq_cluster_width)

        if self._rng.binomial(1, prob) > 0.5:
            return [[self._iq_cluster_centers[0] + rand_i, self._iq_cluster_centers[1] + rand_q]]
        else:
            return [[self._iq_cluster_centers[2] + rand_i, self._iq_cluster_centers[3] + rand_q]]

    def run(self, run_input, **options) -> TestJob:
        """Subclasses will need to override this."""
