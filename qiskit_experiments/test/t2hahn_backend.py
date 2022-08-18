# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
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

from typing import List
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
    ):
        """
        Initialize the T2Hahn backend
        """
        configuration = QasmBackendConfiguration(
            backend_name="T2Hahn_simulator",
            backend_version="0",
            n_qubits=int(1e6),
            basis_gates=["barrier", "rx", "delay", "measure"],
            gates=[],
            local=True,
            simulator=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=int(1e6),
            coupling_map=None,
        )

        self._t2hahn = t2hahn
        self._frequency = frequency
        self._initialization_error = initialization_error
        self._readout0to1 = readout0to1
        self._readout1to0 = readout1to0
        self._rng = np.random.default_rng(seed=SEED)
        super().__init__(configuration)

    @classmethod
    def _default_options(cls):
        """Default options of the test backend."""
        return Options(shots=1024)

    def _qubit_initialization(self, nqubits: int) -> List[dict]:
        """
        Initialize the list of qubits state. If initialization error is provided to the backend it will
        use it to determine the initialized state.

        Args:
            nqubits(int): the number of qubits in the circuit.

        Returns:
            List[dict]: A list of dictionary which each dictionary contain the qubit state in the format
                        {"XY plane": (bool), "ZX plane": (bool), "Theta": float}

        Raises:
            QiskitError: Raised if initialization_error type isn't 'None'', 'float' or a list of 'float'
                         with length of number of the qubits.
            ValueError: Raised if the initialization error is negative.
        """
        qubits_sates = [{} for _ in range(nqubits)]
        # Making an array with the initialization error for each qubit.
        initialization_error = self._initialization_error
        if isinstance(initialization_error, float) or initialization_error is None:
            initialization_error_arr = [initialization_error for _ in range(nqubits)]
        elif isinstance(initialization_error, list):
            if len(initialization_error) == 1:
                initialization_error_arr = [initialization_error[0] for _ in range(nqubits)]
            elif len(initialization_error) == nqubits:
                initialization_error_arr = initialization_error
            else:
                raise QiskitError(
                    f"The length of the list {initialization_error} isn't the same as the number "
                    "of qubits."
                )
        else:
            raise QiskitError("Initialization error type isn't a list or float")

        for err in initialization_error_arr:
            if not isinstance(err, float):
                raise QiskitError("Initialization error type isn't a list or float")
            if err < 0:
                raise ValueError("Initialization error value can't be negative.")

        for qubit in range(nqubits):
            if initialization_error_arr[qubit] is not None and (
                self._rng.random() < initialization_error_arr[qubit]
            ):
                qubits_sates[qubit] = {"XY plane": False, "ZX plane": True, "Theta": np.pi}
            else:
                qubits_sates[qubit] = {
                    "XY plane": False,
                    "ZX plane": True,
                    "Theta": 0,
                }
        return qubits_sates

    def _delay_gate(self, qubit_state: dict, delay: float, t2hahn: float, frequency: float) -> dict:
        """
        Apply delay gate to the qubit. From the delay time we can calculate the probability
        that an error has accrued.

        Args:
            qubit_state(dict): The state of the qubit before operating the gate.
            delay(float): The time in which there are no operation on the qubit.
            t2hahn(float): The T2 parameter of the backhand for probability calculation.
            frequency(float): The frequency of the qubit for phase calculation.

        Returns:
            dict: The state of the qubit after operating the gate.

         Raises:
            QiskitError: Raised if the frequency is 'None' or if the qubit isn't in the XY plane.
        """
        if frequency is None:
            raise QiskitError("Delay gate supported only if the qubit is on the XY plane.")
        new_qubit_state = qubit_state
        if qubit_state["XY plane"]:
            prob_noise = 1 - (np.exp(-delay / t2hahn))
            if self._rng.random() < prob_noise:
                if self._rng.random() < 0.5:
                    new_qubit_state = {
                        "XY plane": False,
                        "ZX plane": True,
                        "Theta": 0,
                    }
                else:
                    new_qubit_state = {
                        "XY plane": False,
                        "ZX plane": True,
                        "Theta": np.pi,
                    }
            else:
                phase = frequency * delay
                new_theta = qubit_state["Theta"] + phase
                new_theta = new_theta % (2 * np.pi)
                new_qubit_state = {"XY plane": True, "ZX plane": False, "Theta": new_theta}
        else:
            if not isclose(qubit_state["Theta"], np.pi) and not isclose(qubit_state["Theta"], 0):
                raise QiskitError("Delay gate supported only if the qubit is on the XY plane.")
        return new_qubit_state

    def _rx_gate(self, qubit_state: dict, angle: float) -> dict:
        """
        Apply Rx gate.

        Args:
            qubit_state(dict): The state of the qubit before operating the gate.
            angle(float): The angle of the rotation.

        Returns:
            dict: The state of the qubit after operating the gate.

        Raises:
            QiskitError: if angle is not ±π/2 or ±π. Those are the only supported angles.
        """

        if qubit_state["XY plane"]:
            if isclose(angle, np.pi):
                new_theta = -qubit_state["Theta"]
                new_theta = new_theta % (2 * np.pi)
                new_qubit_state = {
                    "XY plane": True,
                    "ZX plane": False,
                    "Theta": new_theta,
                }
            elif isclose(angle, np.pi / 2):
                new_theta = (np.pi / 2) - qubit_state["Theta"]
                new_theta = new_theta % (2 * np.pi)
                new_qubit_state = {
                    "XY plane": False,
                    "ZX plane": True,
                    "Theta": new_theta,
                }
            elif isclose(angle, -np.pi / 2):
                new_theta = np.abs((-np.pi / 2) - qubit_state["Theta"])
                new_theta = new_theta % (2 * np.pi)
                new_qubit_state = {
                    "XY plane": False,
                    "ZX plane": True,
                    "Theta": new_theta,
                }
            else:
                raise QiskitError(
                    f"Error - the angle {angle} isn't supported. We only support multiplications of pi/2"
                )
        else:
            if isclose(angle, np.pi):
                new_theta = qubit_state["Theta"] + np.pi
                new_theta = new_theta % (2 * np.pi)
                new_qubit_state = {
                    "XY plane": False,
                    "ZX plane": True,
                    "Theta": new_theta,
                }
            elif isclose(angle, np.pi / 2):
                new_theta = (
                    qubit_state["Theta"] + 3 * np.pi / 2
                )  # its theta -pi/2 but we added 2*pi
                new_theta = new_theta % (2 * np.pi)
                new_qubit_state = {
                    "XY plane": True,
                    "ZX plane": False,
                    "Theta": new_theta,
                }
            elif isclose(angle, -np.pi / 2):
                new_theta = np.pi / 2 - qubit_state["Theta"]
                new_theta = new_theta % (2 * np.pi)
                new_qubit_state = {
                    "XY plane": True,
                    "ZX plane": False,
                    "Theta": new_theta,
                }
            else:
                raise QiskitError(
                    f"Error - The angle {angle} isn't supported. We only support multiplication of pi/2"
                )
        return new_qubit_state

    def _measurement_gate(self, qubit_state: dict) -> int:
        """
        Implementing measurement on qubit with read-out error.

        Args:
            qubit_state(dict): The state of the qubit at the end of the circuit.

        Returns:
            int: The result of the measurement after applying read-out error.
        """
        # Here we are calculating the probability for measurement result depending on the
        # location of the qubit on the Bloch sphere.
        if qubit_state["XY plane"]:
            meas_res = self._rng.random() < 0.5
        else:
            # Since we are not in the XY plane, we need to calculate the probability for
            # measuring output. First, we calculate the probability and later we are
            # tossing to see if the event did happen.
            z_projection = np.cos(qubit_state["Theta"])
            probability = z_projection**2
            if self._rng.random() > probability:
                meas_res = self._rng.random() < 0.5
            else:
                meas_res = z_projection < 0

        # Measurement error implementation
        if meas_res and self._readout1to0 is not None:
            if self._rng.random() < self._readout1to0[0]:
                meas_res = 0
        elif not meas_res and self._readout0to1 is not None:
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

            for _ in range(shots):
                qubit_state = self._qubit_initialization(
                    nqubits=nqubits
                )  # for parallel need to make an array
                clbits = np.zeros(circ.num_clbits, dtype=int)
                for op, qargs, cargs in circ.data:
                    qubit = qubit_indices[qargs[0]]

                    # The noise will only be applied if we are in the XY plane.
                    if op.name == "delay":
                        delay = op.params[0]
                        t2hahn = self._t2hahn[qubit]
                        freq = self._frequency[qubit]
                        qubit_state[qubit] = self._delay_gate(
                            qubit_state=qubit_state[qubit],
                            delay=delay,
                            t2hahn=t2hahn,
                            frequency=freq,
                        )
                    elif op.name == "rx":
                        qubit_state[qubit] = self._rx_gate(qubit_state[qubit], op.params[0])
                    elif op.name == "measure":
                        meas_res = self._measurement_gate(qubit_state[qubit])
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
