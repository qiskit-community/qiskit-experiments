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
from typing import List

from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.jobs.aerjob import AerJob
from qiskit.providers.aer.noise.passes import RelaxationNoisePass
from qiskit.circuit import Delay
from qiskit_experiments.framework import BackendData


class NoisyDelayAerBackend(AerSimulator):
    """Backend for T1 and T2Ramsey experiments."""

    def __init__(
        self,
        t1: List[float] = None,
        t2: List[float] = None,
        dt: float = None,
        backend=None,
        **backend_options,
    ):
        """configure backend noise"""

        super().__init__(**backend_options)
        self._t2 = t2 or [1e-4]
        self._t1 = t1 or [2e-4]
        if backend:
            dt = BackendData(backend).dt
        if dt is not None:
            self._dt_unit = True
            self._dt_factor = dt
        else:
            self._dt_unit = False
            self._dt_factor = dt or 1e-9

        self._op_types = [Delay]

    # pylint: disable=arguments-differ
    def run(self, run_input: List[QuantumCircuit], **run_options) -> AerJob:
        """
        Add noise pass to all circuits and then run the circuits.
        Args:
            run_input: List of circuit to run.
            run_options (kwargs): additional run time backend options.

        Returns:
            AerJob: A job that contains the simulated data.

        """
        relax_pass = RelaxationNoisePass(
            self._t1, self._t2, dt=self._dt_factor, op_types=self._op_types
        )

        noisy_circuits = []
        for circ in run_input:
            noisy_circuits.append(relax_pass(circ))

        return super().run(noisy_circuits, **run_options)
