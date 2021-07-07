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

"""Class to test composite experiments."""

from typing import Optional

from qiskit.providers import Backend
from qiskit.providers.backend import BackendV1
from qiskit.providers.options import Options
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit.test import QiskitTestCase

from qiskit_experiments.test.mock_job import MockJob
from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.composite.parallel_experiment import ParallelExperiment


class DummyAnalysis(BaseAnalysis):
    """
    Dummy analysis class for test purposes only.
    """

    def _run_analysis(self, experiment_data, **options):
        return [], None


class DummyExperiment(BaseExperiment):
    """
    Dummy experiment class for test purposes only.
    """

    __analysis_class__ = DummyAnalysis

    def circuits(self, backend: Optional[Backend] = None):
        return []


class DummyBackend(BackendV1):
    """
    Dummy backend for test purposes only.
    """

    def __init__(self):
        configuration = QasmBackendConfiguration(
            backend_name="dummy_backend",
            backend_version="0",
            n_qubits=int(1e6),
            basis_gates=["barrier", "x", "delay", "measure"],
            gates=[],
            local=True,
            simulator=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=int(1e6),
            coupling_map=None,
        )
        super().__init__(configuration)

    @classmethod
    def _default_options(cls):
        return Options()

    def run(self, run_input, **options):
        result = {
            "backend_name": "Dummmy backend",
            "backend_version": "0",
            "qobj_id": 0,
            "job_id": 0,
            "success": True,
            "results": [],
        }
        return MockJob(backend=self, result=Result.from_dict(result))


class TestComposite(QiskitTestCase):
    """
    Test composite experiment behavior.
    """

    def test_parallel_options(self):
        """
        Test parallel experiments overriding sub-experiment options.
        """

        exp0 = DummyExperiment(0)
        exp2 = DummyExperiment(2)
        exp2.set_run_options(shots=2000)

        par_exp = ParallelExperiment([exp0, exp2])
        with self.assertWarnsRegex(
            Warning, "Sub-experiment run options are overridden by composite experiment settings."
        ):
            par_exp.run(DummyBackend())
