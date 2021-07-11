from qiskit.providers import Backend
from qiskit.providers.backend import BackendV1
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.providers.options import Options

from qiskit.result import Result

from qiskit_experiments.test.mock_job import MockJob


class FakeBackend(BackendV1):
    """
    Fake backend for test purposes only.
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
