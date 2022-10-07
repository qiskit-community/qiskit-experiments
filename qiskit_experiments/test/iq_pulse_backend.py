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

"""A Pulse simulation backend based on Qiskit-Dynamics"""
import datetime
from typing import Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2, QubitProperties
from qiskit.providers.models import PulseDefaults
from qiskit.providers.options import Options
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.quantum_info.states import Statevector
from qiskit.result import Result
from qiskit.transpiler import Target, InstructionProperties

from qiskit_dynamics import Solver
from qiskit_dynamics.pulse import InstructionToSignals

from qiskit_experiments.exceptions import QiskitError
from qiskit_experiments.test.utils import FakeJob

from qiskit.circuit.measure import Measure


class IQPulseBackend(BackendV2):
    """Pulse Simulator abstract class"""

    def __init__(self, static_hamiltonian, hamiltonian_operators, dt=0.1 * 1e-9, **kwargs):
        """Hamiltonian and operators is the Qiskit Dynamics object"""
        super().__init__(
            None,
            name="PulseBackendV2",
            description="A PulseBackend simulator",
            online_date=datetime.datetime.utcnow(),
            backend_version="0.0.1",
        )
        self._defaults = PulseDefaults.from_dict({
                "qubit_freq_est": [0],
                "meas_freq_est": [0],
                "buffer": 0,
                "pulse_library": [],
                "cmd_def": [],
            })
        
        self.converter = None

        self.static_hamiltonian = static_hamiltonian
        self.hamiltonian_operators = hamiltonian_operators
        self.solver = Solver(self.static_hamiltonian, self.hamiltonian_operators, **kwargs)
        self._target = Target(dt=dt, granularity=16)
        self.gound_state = np.zeros(self.solver.model.dim)
        self.gound_state[0] = 1
        self.y_0 = np.eye(self.solver.model.dim)

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
    def _get_info(inst):
        p_dict = inst.operation
        qubit = tuple(int(str(val)[-2]) for val in inst.qubits)
        params = tuple(float(val) for val in p_dict.params)
        return qubit, params, p_dict.name

    @staticmethod
    def _state_vector_to_result(
        state: Union[Statevector, np.ndarray],
        shots: int = 1024,
        meas_return: MeasReturnType = 0,
        meas_level: MeasLevel = 0,
    ):
        """Convert the state vector to IQ data or counts."""

        # if meas_level == MeasLevel.CLASSIFIED:
        # sample from the state vector. There might already be functions in Qiskit to do this.
        #     counts = ...
        # if meas_level == MeasLevel.KERNELED:
        #     iq_data = ... create IQ data.
        pass

    def solve(self, schedule_blocks, qubits):
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

    # Parameters differ from overridden 'run' method
    def run(self, run_input, **options) -> FakeJob:
        """This should be able to return both counts and IQ data depending on the run options.
        Loop through each circuit, extract the pulses from the circuit.calibraitons
        and give them to the pulse solver.
        Multiply the unitaries of the gates together to get a state vector
        that we convert to counts or IQ Data"""

        if isinstance(run_input, QuantumCircuit):
            run_input = [run_input]

        results = []
        experiment_unitaries = {}

        for circuit in run_input:
            if circuit.calibrations.__len__ == 0:
                raise QiskitError("TODO get schedule using pulse.InstructionScheduleMap")
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
            
            memory = self._state_vector_to_result(psi/np.linalg.norm(psi), **options)
            counts = dict(zip(*np.unique(memory, return_counts=True)))
            run_result = {
                "shots": shots,
                "success": True,
                "header": {"metadata": circuit.metadata},
                "meas_level": meas_level,
                "data": {
                    "counts": counts,
                    "memory": memory,
                },
            }

            if meas_level == MeasLevel.CLASSIFIED:
                counts = {}
                results = self._rng.multinomial(shots, prob_arr, size=1)[0]
                for result, num_occurrences in enumerate(results):
                    result_in_str = str(format(result, "b").zfill(output_length))
                    counts[result_in_str] = num_occurrences
                run_result["counts"] = counts
            else:
                # Phase has meaning only for IQ shot, so we calculate it here
                phase = self.experiment_helper.iq_phase([circuit])[0]
                iq_cluster_centers, iq_cluster_width = self.experiment_helper.iq_clusters([circuit])[0]

                # 'circ_qubits' get a list of all the qubits
                memory = self._draw_iq_shots(
                    prob_arr,
                    shots,
                    list(range(output_length)),
                    iq_cluster_centers,
                    iq_cluster_width,
                    phase,
                )
                if meas_return == "avg":
                    memory = np.average(np.array(memory), axis=0).tolist()
                run_result["memory"] = memory

            result["results"].append(run_result)
        return FakeJob(self, Result.from_dict(result))


class SingleTransmonTestBackend(IQPulseBackend):
    """Construct H in the init"""

    def __init__(self, omega_01, delta, lambda_0, lambda_1):

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

        self._target = Target(qubit_properties=[QubitProperties(frequency=omega_01)], dt=self.dt, granularity=16)

        measure_props = {
            (0,): InstructionProperties(duration=0, error=0),
        }
        self._target.add_instruction(Measure(), measure_props)

        self.converter = InstructionToSignals(self.dt, carriers={"d0": omega_01})
