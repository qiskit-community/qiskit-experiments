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

"""Fake backend class for tests."""
import uuid
from qiskit.circuit.library import Measure
from qiskit.providers import ProviderV1
from qiskit.providers.backend import BackendV2
from qiskit.providers.options import Options
from qiskit.transpiler import Target

from qiskit.result import Result

from qiskit_experiments.test.utils import FakeJob


class FakeProvider(ProviderV1):
    """Fake provider with no backends for testing"""

    def backends(self, name=None, **kwargs):
        """List of available backends. Empty in this case"""
        return []


class FakeBackend(BackendV2):
    """
    Fake backend for test purposes only.
    """

    def __init__(
        self,
        provider=FakeProvider(),
        backend_name="fake_backend",
        num_qubits=1,
        max_experiments=100,
    ):
        self.simulator = True
        super().__init__(provider=provider, name=backend_name)
        self._target = Target(num_qubits=num_qubits)
        # Add a measure for each qubit so a simple measure circuit works
        self.target.add_instruction(Measure())
        self._max_circuits = max_experiments

    @property
    def max_circuits(self):
        """Maximum circuits to run at once"""
        return self._max_circuits

    @classmethod
    def _default_options(cls):
        return Options()

    @property
    def target(self) -> Target:
        return self._target

    def run(self, run_input, **options):
        shots = options.get("shots", 100)
        if not isinstance(run_input, list):
            run_input = [run_input]
        results = [
            {
                "data": {"0": shots},
                "shots": shots,
                "success": True,
                "header": {"metadata": circ.metadata},
                "meas_level": 2,
            }
            for circ in run_input
        ]

        result = {
            "backend_name": "fake_backend",
            "backend_version": "0",
            "qobj_id": uuid.uuid4().hex,
            "job_id": uuid.uuid4().hex,
            "success": True,
            "results": results,
        }
        return FakeJob(backend=self, result=Result.from_dict(result))
