# -*- coding: utf-8 -*-

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
A simulator for T1 experiment for testing and documentation
"""

import numpy as np
from qiskit.providers import BaseBackend
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result


class T1Backend(BaseBackend):
    """
    A simple and primitive backend, to be run by the T1 tests
    """

    def __init__(self, t1, initial_prob1=None, readout0to1=None, readout1to0=None, dt_factor=None):
        """
        Initialize the T1 backend
        """

        dt_factor_in_ns = dt_factor * 1e9 if dt_factor is not None else None

        configuration = QasmBackendConfiguration(
            backend_name="t1_simulator",
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
            dt=dt_factor_in_ns,
        )

        self._t1 = t1
        self._initial_prob1 = initial_prob1
        self._readout0to1 = readout0to1
        self._readout1to0 = readout1to0
        self._dt_factor = dt_factor
        super().__init__(configuration)

    # pylint: disable = arguments-differ
    def run(self, qobj):
        """
        Run the T1 backend
        """

        shots = qobj.config.shots

        result = {
            "backend_name": "T1 backend",
            "backend_version": "0",
            "qobj_id": 0,
            "job_id": 0,
            "success": True,
            "results": [],
        }

        for circ in qobj.experiments:
            nqubits = circ.config.n_qubits
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
                if self._initial_prob1 is None:
                    prob1 = np.zeros(nqubits)
                else:
                    prob1 = self._initial_prob1.copy()

                clbits = np.zeros(circ.config.memory_slots, dtype=int)

                for op in circ.instructions:
                    qubit = op.qubits[0]
                    if op.name == "x":
                        prob1[qubit] = 1 - prob1[qubit]
                    elif op.name == "delay":
                        delay = op.params[0]
                        prob1[qubit] = prob1[qubit] * np.exp(-delay / self._t1[qubit])
                    elif op.name == "measure":
                        meas_res = np.random.binomial(
                            1, prob1[qubit] * (1 - ro10[qubit]) + (1 - prob1[qubit]) * ro01[qubit]
                        )
                        clbits[op.memory[0]] = meas_res
                        prob1[qubit] = meas_res

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
                    "header": {"metadata": circ.header.metadata},
                    "data": {"counts": counts},
                }
            )

        return Result.from_dict(result)
