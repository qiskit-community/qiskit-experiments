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
T2HahnBackend class.
Temporary backend to be used for t2hahn experiment
"""

import numpy as np

from qiskit.providers import BackendV1
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit_experiments.framework import Options
from qiskit_experiments.test.utils import FakeJob

# Fix seed for simulations
SEED = 9000


class T2HahnBackend(BackendV1):
    """
    A simple and primitive backend, to be run by the T2Hahn tests
    """

    def __init__(
        self,
        p0=None,
        initial_prob_plus=None,
        readout0to1=None,
        readout1to0=None,
        conversion_factor=1,
    ):
        """
        Initialize the T2Hahn backend
        """
        conversion_factor_in_ns = conversion_factor * 1e9 if conversion_factor is not None else None
        configuration = QasmBackendConfiguration(
            backend_name="T2Hahn_simulator",
            backend_version="0",
            n_qubits=int(1e6),
            basis_gates=["barrier", "ry", "rx", "delay", "measure"],
            gates=[],
            local=True,
            simulator=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=int(1e6),
            coupling_map=None,
            dt=conversion_factor_in_ns,
        )

        self._t2hahn = p0["T2"]
        self._a_param = p0["A"]
        self._phi = p0["phi"]
        self._b_param = p0["B"]
        self._initial_prob_plus = None
        self._readout0to1 = readout0to1
        self._readout1to0 = readout1to0
        self._conversion_factor = conversion_factor
        self._rng = np.random.default_rng(seed=SEED)
        super().__init__(configuration)

    @classmethod
    def _default_options(cls):
        """Default options of the test backend."""
        return Options(shots=1024)

    # pylint: disable = arguments-differ
    def run(self, run_input, **options):
        """
        Run the T2Hahn backend
        """
        self.options.update_options(**options)
        shots = self.options.get("shots")
        result = {
            "backend_name": "T2Hahn backend",
            "backend_version": "0",
            "qobj_id": 0,
            "job_id": 0,
            "success": True,
            "results": [],
        }
        for circ in run_input:
            nqubits = circ.num_qubits
            qubit_indices = {bit: idx for idx, bit in enumerate(circ.qubits)}
            clbit_indices = {bit: idx for idx, bit in enumerate(circ.clbits)}
            counts = dict()

            for _ in range(shots):
                qubit_state = {"qubit state": 0, "XY plain": False, "Theta": 0}
                delayCheck = True
                clbits = np.zeros(circ.num_clbits, dtype=int)
                for op, qargs, cargs in circ.data:
                    qubit = qubit_indices[qargs[0]]

                    # The noise will only be applied if we are in the XY plain.
                    if op.name == "delay" and delayCheck:
                        delay = op.params[0]
                        t2hahn = self._t2hahn[qubit] * self._conversion_factor
                        if qubit_state["XY plain"]:
                            prob_noise = 1 - (np.exp(-delay / t2hahn))
                            if self._rng.random() < prob_noise:
                                if self._rng.random() < 0.5:
                                    qubit_state = {"qubit state": 0, "XY plain": False, "Theta": 0}
                                else:
                                    qubit_state = {"qubit state": 1, "XY plain": False, "Theta": 0}

                    if op.name == "rx":
                        if qubit_state["XY plain"]:
                            qubit_state = {"qubit state": 0, "XY plain": True, "Theta": np.pi}
                        elif qubit_state["qubit state"] == 0:
                            qubit_state = {"qubit state": 1, "XY plain": False, "Theta": 0}
                        else:
                            qubit_state = {"qubit state": 0, "XY plain": False, "Theta": 0}

                    if op.name == "ry":
                        if qubit_state["XY plain"]:
                            if qubit_state["Theta"] == 0:
                                qubit_state = {"qubit state": 1, "XY plain": False, "Theta": 0}
                            else:
                                qubit_state = {"qubit state": 0, "XY plain": False, "Theta": 0}
                        elif qubit_state["qubit state"] == 0:
                            qubit_state = {"qubit state": 1, "XY plain": True, "Theta": 0}
                        else:
                            qubit_state = {"qubit state": 0, "XY plain": True, "Theta": np.pi}

                    if op.name == "measure":
                        # we measure in |+> basis which is the same as measuring |0>
                        if qubit_state["XY plain"]:
                            meas_res = self._rng.random() < 0.5
                        else:
                            meas_res = qubit_state["qubit state"]
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
