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
"""
T2Ramsey Backend class.
Temporary backend to be used for t2ramsey experiment
"""

import numpy as np

from qiskit.providers import BackendV1
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit_experiments.framework import Options
from qiskit_experiments.test.utils import FakeJob

# Fix seed for simulations
SEED = 9000


class T2RamseyBackend(BackendV1):
    """
    A simple and primitive backend, to be run by the T2Ramsey tests
    """

    def __init__(
        self,
        p0=None,
        initial_prob_plus=None,
        readout0to1=None,
        readout1to0=None,
    ):
        """
        Initialize the T2Ramsey backend
        """
        configuration = QasmBackendConfiguration(
            backend_name="T2Ramsey_simulator",
            backend_version="0",
            n_qubits=int(1e6),
            basis_gates=["barrier", "h", "p", "delay", "measure"],
            gates=[],
            local=True,
            simulator=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=int(1e6),
            coupling_map=None,
        )

        self._t2ramsey = p0["T2star"]
        self._a_param = p0["A"]
        self._freq = p0["f"]
        self._phi = p0["phi"]
        self._b_param = p0["B"]
        self._initial_prob_plus = initial_prob_plus
        self._readout0to1 = readout0to1
        self._readout1to0 = readout1to0
        self._rng = np.random.default_rng(0)
        super().__init__(configuration)

    @classmethod
    def _default_options(cls):
        """Default options of the test backend."""
        return Options(shots=1024)

    # pylint: disable = arguments-differ
    def run(self, run_input, **options):
        """
        Run the T2Ramsey backend
        """
        self.options.update_options(**options)
        shots = self.options.get("shots")
        result = {
            "backend_name": "T2Ramsey backend",
            "backend_version": "0",
            "qobj_id": "0",
            "job_id": "0",
            "success": True,
            "results": [],
        }
        for circ in run_input:
            nqubits = circ.num_qubits
            qubit_indices = {bit: idx for idx, bit in enumerate(circ.qubits)}
            clbit_indices = {bit: idx for idx, bit in enumerate(circ.clbits)}
            counts = dict()
            if self._readout0to1 is None:
                ro01 = np.zeros(nqubits)
            else:
                ro01 = self._readout0to1
            if self._readout1to0 is None:
                ro10 = np.zeros(nqubits)
            else:
                ro10 = self._readout1to0
            for _ in range(shots):
                if self._initial_prob_plus is None:
                    prob_plus = np.ones(nqubits)
                else:
                    prob_plus = self._initial_prob_plus.copy()

                clbits = np.zeros(circ.num_clbits, dtype=int)
                for op, qargs, cargs in circ.data:
                    qubit = qubit_indices[qargs[0]]

                    if op.name == "delay":
                        delay = op.params[0]
                        t2ramsey = self._t2ramsey[qubit]
                        freq = self._freq[qubit]

                        prob_plus[qubit] = (
                            self._a_param[qubit]
                            * np.exp(-delay / t2ramsey)
                            * np.cos(2 * np.pi * freq * delay + self._phi[qubit])
                            + self._b_param[qubit]
                        )

                    if op.name == "measure":
                        # we measure in |+> basis which is the same as measuring |0>
                        meas_res = self._rng.binomial(
                            1,
                            (1 - prob_plus[qubit]) * (1 - ro10[qubit])
                            + prob_plus[qubit] * ro01[qubit],
                        )
                        clbit = clbit_indices[cargs[0]]
                        clbits[clbit] = meas_res

                clstr = ""
                for clbit in clbits[::-1]:
                    clstr = clstr + str(clbit)

                if clstr in counts:
                    counts[clstr] += 1
                else:
                    counts[clstr] = 1
            result["results"].append(
                {
                    "shots": shots,
                    "success": True,
                    "header": {"metadata": circ.metadata},
                    "data": {"counts": counts},
                }
            )
        return FakeJob(self, Result.from_dict(result))
