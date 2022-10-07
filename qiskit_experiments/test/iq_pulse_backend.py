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

"""A Pulse simulation backend based on Qiskit-Dynamics"""
import copy
import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.measure import Measure
from qiskit.providers import BackendV2, QubitProperties
from qiskit.providers.models import PulseDefaults
from qiskit.providers.options import Options
from qiskit.pulse import ScheduleBlock
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.quantum_info.states import Statevector
from qiskit.result import Result
from qiskit.transpiler import InstructionProperties, Target

from qiskit_dynamics import Solver
from qiskit_dynamics.pulse import InstructionToSignals

from qiskit_experiments.exceptions import QiskitError
from qiskit_experiments.test.utils import FakeJob


class IQPulseBackend(BackendV2):
    """Pulse Simulator abstract class"""

    def __init__(
        self,
        static_hamiltonian: np.ndarray,
        hamiltonian_operators: np.ndarray,
        dt: Optional[float] = 0.1 * 1e-9,
        **kwargs,
    ):
        """Hamiltonian and operators is the Qiskit Dynamics object"""
        super().__init__(
            None,
            name="PulseBackendV2",
            description="A PulseBackend simulator",
            online_date=datetime.datetime.utcnow(),
            backend_version="0.0.1",
        )
        self._defaults = PulseDefaults.from_dict(
            {
                "qubit_freq_est": [0],
                "meas_freq_est": [0],
                "buffer": 0,
                "pulse_library": [],
                "cmd_def": [],
            }
        )

        self.converter = None

        self.static_hamiltonian = static_hamiltonian
        self.hamiltonian_operators = hamiltonian_operators
        self.solver = Solver(self.static_hamiltonian, self.hamiltonian_operators, **kwargs)
        self._target = Target(dt=dt, granularity=16)
        self.gound_state = np.zeros(self.solver.model.dim)
        self.gound_state[0] = 1
        self.y_0 = np.eye(self.solver.model.dim)

    @property
    def default_pulse_unitaries(self) -> Dict[Tuple, np.array]:
        """Return the default unitary matrices of the backend."""
        return self._simulated_pulse_unitaries

    @default_pulse_unitaries.setter
    def default_pulse_unitaries(self, unitaries: Dict[Tuple, np.array]):
        """Set the default unitary pulses this allows the tests to simulate the pulses only once."""
        self._simulated_pulse_unitaries = unitaries

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    def defaults(self):
        """return backend defaults"""
        return self._defaults

    @classmethod
    def _default_options(cls):
        return Options(shots=1024)

    @staticmethod
    def _get_info(instruction: CircuitInstruction) -> Tuple[Tuple(int), Tuple(float), str]:
        p_dict = instruction.operation
        qubit = tuple(int(str(val)[-2]) for val in instruction.qubits)
        params = tuple(float(val) for val in p_dict.params)
        return qubit, params, p_dict.name

    @staticmethod
    def _state_vector_to_result(
        state: Union[Statevector, np.ndarray],
        shots: Optional[int] = 1024,
        meas_return: Optional[MeasReturnType] = 0,
        meas_level: Optional[MeasLevel] = 0,
    ) -> Union[Dict[str, int], complex]:
        """Convert the state vector to IQ data or counts."""

        if meas_level == MeasLevel.CLASSIFIED:
            measurement = Statevector(state).sample_counts(shots)
        elif meas_level == MeasLevel.KERNELED:
            raise QiskitError("TODO: generate IQ data")
            # measurement = iq_data = ... #create IQ data.

        if meas_return == "avg":
            return np.average(list(measurement.keys()), weights=list(measurement.values()))
        else:
            return measurement

    @lru_cache
    def solve(self, schedule_blocks: ScheduleBlock, qubits: Tuple(int)) -> np.ndarray:
        """Soleves a single schdule block and returns the unitary"""
        if len(qubits) > 1:
            QiskitError("TODO multi qubit gates")

        signal = self.converter.get_signals(schedule_blocks)
        time_f = schedule_blocks.duration * self.dt
        result = self.solver.solve(
            t_span=[0.0, time_f],
            y0=self.y_0,
            t_eval=[time_f],
            signals=signal,
            method="RK23",
        ).y[0]

        return result

    def run(self, run_input: Union[QuantumCircuit, List[QuantumCircuit]], **options) -> FakeJob:
        """run method takes circuits as input and returns FakeJob object with results"""

        self.options.update_options(**options)
        shots = self.options.get("shots")
        meas_level = self.options.get("meas_level")

        result = {
            "backend_name": f"{self.__class__.__name__}",
            "backend_version": self.backend_version,
            "qobj_id": 0,
            "job_id": 0,
            "success": True,
            "results": [],
        }

        if isinstance(run_input, QuantumCircuit):
            run_input = [run_input]

        experiment_unitaries = copy.deepcopy(self.default_pulse_unitaries)

        for circuit in run_input:
            for name, schedule in circuit.calibrations.items():
                for (qubits, params), schedule_block in schedule.items():
                    if (name, qubits, params) not in experiment_unitaries:
                        experiment_unitaries[(name, qubits, params)] = self.solve(
                            schedule_block, qubits
                        )

            psi = self.gound_state.copy()
            for instruction in circuit.data:
                qubits, params, inst_name = self._get_info(instruction)
                if inst_name in ["barrier", "measure"]:
                    continue
                unitary = experiment_unitaries[(inst_name, qubits, params)]
                psi = unitary @ psi

            counts = self._state_vector_to_result(psi / np.linalg.norm(psi), **options)
            # counts = dict(zip(*np.unique(memory, return_counts=True)))
            run_result = {
                "shots": shots,
                "success": True,
                "header": {"metadata": circuit.metadata},
                "meas_level": meas_level,
                "data": {
                    "counts": counts,
                    # "memory": memory,
                },
            }

            result["results"].append(run_result)
        return FakeJob(self, Result.from_dict(result))


class SingleTransmonTestBackend(IQPulseBackend):
    """Three level aharmonic transmon qubit"""

    def __init__(self, omega_01: float, delta: float, lambda_0: float, lambda_1: float):

        omega_02 = 2 * omega_01 + delta
        ket0 = np.array([[1, 0, 0]]).T
        ket1 = np.array([[0, 1, 0]]).T
        ket2 = np.array([[0, 0, 1]]).T

        sigma_m1 = ket0 @ ket1.T.conj()
        sigma_m2 = ket1 @ ket2.T.conj()

        sigma_p1 = sigma_m1.T.conj()
        sigma_p2 = sigma_m2.T.conj()

        p1 = ket1 @ ket1.T.conj()
        p2 = ket2 @ ket2.T.conj()

        drift = 2 * np.pi * (omega_01 * p1 + omega_02 * p2)
        control = [
            2 * np.pi * (lambda_0 * (sigma_p1 + sigma_m1) + lambda_1 * (sigma_p2 + sigma_m2))
        ]
        r_frame = 2 * np.pi * (omega_01 * p1 + 2 * omega_01 * p2)

        super().__init__(
            static_hamiltonian=drift,
            hamiltonian_operators=control,
            rotating_frame=r_frame,
            rwa_cutoff_freq=1.9 * omega_01,
            rwa_carrier_freqs=[omega_01],
        )

        self._defaults = PulseDefaults.from_dict(
            {
                "qubit_freq_est": [omega_01 / 1e9],
                "meas_freq_est": [0],
                "buffer": 0,
                "pulse_library": [],
                "cmd_def": [],
            }
        )
        self._target = Target(
            qubit_properties=[QubitProperties(frequency=omega_01)], dt=self.dt, granularity=16
        )
        measure_props = {
            (0,): InstructionProperties(duration=0, error=0),
        }
        self._target.add_instruction(Measure(), measure_props)
        self.converter = InstructionToSignals(self.dt, carriers={"d0": omega_01})
