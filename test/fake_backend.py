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
from qiskit.providers.backend import BackendV1
from qiskit.providers.models import QasmBackendConfiguration

from qiskit.result import Result

from qiskit_experiments.framework import Options
from qiskit_experiments.test.utils import FakeJob


class FakeBackend(BackendV1):
    """
    Fake backend for test purposes only.
    """

    def __init__(self, max_experiments=None):
        configuration = QasmBackendConfiguration(
            backend_name="fake_backend",
            backend_version="0",
            n_qubits=int(1e6),
            basis_gates=[],
            gates=[],
            local=True,
            simulator=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=int(1e6),
            max_experiments=max_experiments,
            coupling_map=None,
        )
        super().__init__(configuration)

    @classmethod
    def _default_options(cls):
        return Options()

    def run(self, run_input, **options):
        result = {
            "backend_name": "fake_backend",
            "backend_version": "0",
            "qobj_id": uuid.uuid4().hex,
            "job_id": uuid.uuid4().hex,
            "success": True,
            "results": [],
        }
        return FakeJob(backend=self, result=Result.from_dict(result))
