# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
TphiBackend class.
Temporary backend to be used to test the Tphi experiment
"""

from qiskit.providers import BackendV1
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit_experiments.framework import Options
from qiskit_experiments.test.utils import FakeJob
from qiskit_experiments.test.t1_backend import T1Backend
from qiskit_experiments.test.t2ramsey_backend import T2RamseyBackend

# Fix seed for simulations
SEED = 9000


class TphiBackend(BackendV1):
    """
    A simple and primitive backend, to be run by the Tphi tests
    """

    def __init__(
        self,
        t1=None,
        t2ramsey=None,
        freq=None,
        initial_prob1=None,
        readout0to1=None,
        readout1to0=None,
    ):
        """
        Initialize the Tphi backend
        """
        self._configuration = QasmBackendConfiguration(
            backend_name="tphi_simulator",
            backend_version="0",
            n_qubits=int(1e6),
            basis_gates=["barrier", "h", "p", "delay", "measure", "x"],
            gates=[],
            local=True,
            simulator=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=int(1e6),
            coupling_map=None,
        )
        self._t1 = [t1]
        self._t2ramsey = t2ramsey
        self._freq = freq
        self._initial_prob1 = initial_prob1
        self._readout0to1 = readout0to1
        self._readout1to0 = readout1to0
        self._internal_backends = {}
        super().__init__(self._configuration)

        self._internal_backends["T1"] = T1Backend(
            self._t1, self._initial_prob1, self._readout0to1, self._readout1to0
        )
        if self._initial_prob1 is None:
            self._initial_prob_plus = None
        else:
            self._initial_prob_plus = 0.5  # temporary
        self._p0 = {
            "A": [0.5],
            "T2star": [t2ramsey],
            "f": [freq],
            "phi": [0.0],
            "B": [0.5],
        }
        self._internal_backends["T2*"] = T2RamseyBackend(
            self._p0, self._initial_prob_plus, self._readout0to1, self._readout1to0
        )

    def configuration(self):
        """Return the backend configuration.

        Returns:
            BackendConfiguration: the configuration for the backend.
        """
        return self._configuration

    @classmethod
    def _default_options(cls):
        """Default options of the test backend."""
        return Options()

    def run(self, run_input, **options):
        """
        Run the Tphi backend
        """
        self.options.update_options(**options)
        shots = 1000
        t1_circuits = []
        t2ramsey_circuits = []
        for circ in run_input:
            if circ.metadata["composite_metadata"][0]["experiment_type"] == "T1":
                t1_circuits.append(circ)
            elif circ.metadata["composite_metadata"][0]["experiment_type"] == "T2Ramsey":
                t2ramsey_circuits.append(circ)
            else:
                raise ValueError("Illegal name for circuit in Tphi")

        job_t1 = self._internal_backends["T1"].run(run_input=t1_circuits, shots=shots)
        job_t2ramsey = self._internal_backends["T2*"].run(run_input=t2ramsey_circuits, shots=shots)
        final_results = job_t1.result().results + job_t2ramsey.result().results
        result_for_fake = Result(
            backend_name="Tphi backend",
            backend_version="0",
            qobj_id="0",
            job_id="0",
            success=True,
            results=final_results,
            status="JobStatus.DONE",
        )
        return FakeJob(self, result_for_fake)
