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
from numpy import isclose
from qiskit import QiskitError
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
        t2hahn=None,
        frequency=None,
        initialization_error=None,
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

        self._t2hahn = t2hahn
        self._frequency = frequency
        self._initialization_error = initialization_error
        self._readout0to1 = readout0to1
        self._readout1to0 = readout1to0
        self._conversion_factor = conversion_factor
        self._rng = np.random.default_rng(seed=SEED)
        self._measurement_error = 0.05
        super().__init__(configuration)

    @classmethod
    def _default_options(cls):
        """Default options of the test backend."""
        return Options(shots=1024)

    def _qubit_initialization(self) -> dict:
        if self._initialization_error is not None and (self._rng.random() < self._initialization_error[0]):
            return {"XY plain": False, "ZX plain": True, "Theta": np.pi}
        else:
            return {
                "XY plain": False,
                "ZX plain": True,
                "Theta": 0,
            }

    def _delay_gate(self, qubit_state: dict, delay: float, t2hahn: float) -> dict:
        """
        Apply delay gate to the qubit. From the delay time we can calculate the probability
        that an error has accrued.
        Args:
            qubit_state(dict): The state of the qubit before operating the gate.
            delay(float): The time in which there are no operation on the qubit.
            t2hahn(float): The T2 parameter of the backhand for probability calculation.

        Returns:
            dict: The state of the qubit after operating the gate.
        """
        if qubit_state["XY plain"]:
            prob_noise = 1 - (np.exp(-delay / t2hahn))
            if self._rng.random() < prob_noise:
                if self._rng.random() < 0.5:
                    new_qubit_state = {
                        "XY plain": False,
                        "ZX plain": True,
                        "Theta": 0,
                    }
                else:
                    new_qubit_state = {
                        "XY plain": False,
                        "ZX plain": True,
                        "Theta": np.pi,
                    }
            else:
                phase = self._frequency[0] * delay
                new_theta = qubit_state["Theta"] + phase
                new_theta = new_theta % (2 * np.pi)
                new_qubit_state = {
                    "XY plain": True,
                    "ZX plain": False,
                    "Theta": new_theta
                }
        else:
            new_qubit_state = qubit_state
        # new_qubit_state = qubit_state
        return new_qubit_state

    def _rx_gate(self, qubit_state: dict, angle: float) -> dict:
        """
        Apply Rx gate.
        Args:
            qubit_state(dict): The state of the qubit before operating the gate.
            angle(float): The angle of the rotation.

        Returns:
                dict: The state of the qubit after operating the gate.
        """
        if qubit_state["XY plain"]:
            if isclose(angle, np.pi):
                new_theta = - qubit_state["Theta"]
                new_theta = new_theta % (2 * np.pi)
                new_qubit_state = {
                    "XY plain": True,
                    "ZX plain": False,
                    "Theta": new_theta,
                }
            elif isclose(angle, np.pi/2):
                new_theta = angle - qubit_state["Theta"]
                new_theta = new_theta % (2 * np.pi)
                new_qubit_state = {
                    "XY plain": False,
                    "ZX plain": True,
                    "Theta": new_theta,
                }
            else:
                print("Error - This angle isn't supported. We only support multipication of pi/2")
        else:
            if isclose(angle, np.pi/2):
                new_theta = qubit_state["Theta"] + 3 * np.pi/2  # its theta -pi/2 but we added 2*pi
                new_theta = new_theta % (2 * np.pi)
                new_qubit_state = {
                    "XY plain": True,
                    "ZX plain": False,
                    "Theta": new_theta,
                }
            elif isclose(angle, np.pi):
                new_theta = qubit_state["Theta"] + np.pi
                new_theta = new_theta % (2 * np.pi)
                new_qubit_state = {
                    "XY plain": False,
                    "ZX plain": True,
                    "Theta": new_theta,
                }
            else:
                print("Error - This angle isn't supported. We only support multiplication of pi/2")
        return new_qubit_state


    def _measurement_gate(self, qubit_state: dict) -> int:
        """
        implementing measurement on qubit with read-out error.
        Args:
            qubit_state(dict): The state of the qubit at the end of the circuit.

        Returns:
            int: The result of the measurement after applying read-out error.
        """
        if qubit_state["XY plain"]:
            meas_res = self._rng.random() < 0.5
        else:
            z_projection = np.cos(qubit_state["Theta"])
            probability = abs(z_projection) ** 2
            if self._rng.random() > probability:
                meas_res = self._rng.random() < 0.5
            else:
                meas_res = z_projection < 0

        # Measurement error implementation
        if meas_res and self._readout1to0 is not None:
            if self._rng.random() < self._readout1to0[0]:
                meas_res = 0
        elif self._readout0to1 is not None:
            if self._rng.random() < self._readout0to1[0]:
                meas_res = 1

        return meas_res

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
            qubit_indices = {bit: idx for idx, bit in enumerate(circ.qubits)}
            clbit_indices = {bit: idx for idx, bit in enumerate(circ.clbits)}
            counts = dict()

            for _ in range(shots):
                qubit_state = self._qubit_initialization()  # for parrallel need to make an array
                clbits = np.zeros(circ.num_clbits, dtype=int)
                for op, qargs, cargs in circ.data:
                    qubit = qubit_indices[qargs[0]]

                    # The noise will only be applied if we are in the XY plain.
                    if op.name == "delay":
                        delay = op.params[0]
                        t2hahn = self._t2hahn[qubit] * self._conversion_factor
                        qubit_state = self._delay_gate(qubit_state, delay, t2hahn)
                    elif op.name == "rx":
                        qubit_state = self._rx_gate(qubit_state, op.params[0])
                    elif op.name == "measure":
                        # we measure in |+> basis which is the same as measuring |0>
                        meas_res = self._measurement_gate(qubit_state)
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
