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
Temporary backend to be used to apply T2 and T1 noise.
"""
from typing import List, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.providers.backend import Backend
from qiskit.providers.aer.noise.passes import RelaxationNoisePass
from qiskit.circuit import Delay


class NoisyAerBackend(AerSimulator):
    """Backend for T1 and T2Ramsey experiments."""

    def __init__(self, t1: List[float] = None, t2: List[float] = None, dt: float = None, backend=None,
                 op_types: List = None, instruction_durations: List[Tuple] = None,
                 basis_gates: List[str] = None,
                 **backend_options):
        """configure backend noise and gates durations"""

        super().__init__(**backend_options)
        self._t2 = t2 or [1e-4]
        self._t1 = t1 or [2e-4]

        if backend and hasattr(backend.configuration(), "dt"):
            self._dt_unit = True
            self._dt_factor = backend.configuration().dt
        else:
            self._dt_unit = False
            self._dt_factor = dt or 1e-9

        if backend and hasattr(backend.configuration(), "basis_gates"):
            self._basis_gates_controller = True
            self._basis_gates = backend.configuration().basis_gates
        else:
            self._basis_gates_controller = False
            self._basis_gates = basis_gates or ["x", "sx", "rz", "id", "delay"]

        self._op_types = op_types or [Delay]

        # time_unit = 'dt' if self._dt_unit else 's'
        # time_unit = 'dt'
        time_unit = 's'

        us = 1e-9

        self._instruction_durations = instruction_durations or [
            ('x', [0], 50*us, time_unit),
            ('sx', [0], 25*us, time_unit),
            ('id', [0], 25*us, time_unit),
            ('rz', [0], 0, time_unit),
            ('measure', [0], 0, time_unit)
        ]

    def run(self, run_input, **options):
        """
        Add noise pass to all circuits and then run the circuits.
        Args:
            run_input: List of circuit to run.
            **options: Running options.

        Returns:

        """
        optimization_level = options.get("optimization_level") or 0
        options["optimization_level"] = 0
        self.options.update_options(**options)

        relax_pass = RelaxationNoisePass(self._t1, self._t2, dt=self._dt_factor, op_types=self._op_types)

        noisy_circuits = []
        for circ in run_input:
            sched_qc = transpile(circ, basis_gates=self._basis_gates,
                scheduling_method='asap', instruction_durations=self._instruction_durations, optimization_level=0)
            noisy_circuits.append(relax_pass(sched_qc))

        # Return "options" to its previous state
        if optimization_level != 0:
            options["optimization_level"] = optimization_level
            self.options.update_options(**options)

        return super().run(noisy_circuits, **options)
