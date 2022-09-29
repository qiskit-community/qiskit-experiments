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
T2Ramsey Experiment class.

"""

from typing import List, Union, Optional
import numpy as np

import qiskit
from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.library.characterization.analysis.t2ramsey_analysis import T2RamseyAnalysis


class T2Ramsey(BaseExperiment):
    r"""T2 Ramsey Experiment.

    # section: overview

        This experiment is used to estimate two properties for a single qubit:
        T2* and Ramsey frequency.

        See `Qiskit Textbook <https://qiskit.org/textbook/ch-quantum-hardware/\
        calibrating-qubits-pulse.html>`_  for a more detailed explanation on
        these properties.

        This experiment consists of a series of circuits of the form

        .. parsed-literal::

                 ┌───┐┌──────────────┐┌──────┐ ░ ┌───┐ ░ ┌─┐
            q_0: ┤ H ├┤   DELAY(t)   ├┤ P(λ) ├─░─┤ H ├─░─┤M├
                 └───┘└──────────────┘└──────┘ ░ └───┘ ░ └╥┘
            c: 1/═════════════════════════════════════════╩═
                                                        0

        for each *t* from the specified delay times, where
        :math:`\lambda =2 \pi \times {osc\_freq}`,
        and the delays are specified by the user.
        The circuits are run on the device or on a simulator backend.

    # section: tutorial
        :doc:`/tutorials/t2ramsey_characterization`

    # section: analysis_ref
        :py:class:`T2RamseyAnalysis`
    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            delays (Iterable[float]): Delay times of the experiments in seconds.
            osc_freq (float): Oscillation frequency offset in Hz.
        """
        options = super()._default_experiment_options()

        options.delays = None
        options.osc_freq = 0.0

        return options

    def __init__(
        self,
        qubit: int,
        delays: Union[List[float], np.array],
        backend: Optional[Backend] = None,
        osc_freq: float = 0.0,
    ):
        """
        Initialize the T2Ramsey class.

        Args:
            qubit: the qubit under test.
            delays: delay times of the experiments in seconds.
            backend: Optional, the backend to run the experiment on.
            osc_freq: the oscillation frequency induced by the user.
                The frequency is given in Hz.

        """
        super().__init__([qubit], analysis=T2RamseyAnalysis(), backend=backend)
        self.set_experiment_options(delays=delays, osc_freq=osc_freq)

    def _set_backend(self, backend: Backend):
        super()._set_backend(backend)

        # Scheduling parameters
        if not self._backend_data.is_simulator:
            scheduling_method = getattr(self.transpile_options, "scheduling_method", "alap")
            self.set_transpile_options(scheduling_method=scheduling_method)

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits.

        Each circuit consists of a Hadamard gate, followed by a fixed delay,
        a phase gate (with a linear phase), and an additional Hadamard gate.

        Returns:
            The experiment circuits
        """
        dt_unit = False
        if self.backend:
            dt_factor = self._backend_data.dt
            dt_unit = dt_factor is not None

        circuits = []
        for delay in self.experiment_options.delays:
            if dt_unit:
                delay_dt = round(delay / dt_factor)
                real_delay_in_sec = delay_dt * dt_factor
            else:
                real_delay_in_sec = delay

            rotation_angle = 2 * np.pi * self.experiment_options.osc_freq * real_delay_in_sec

            circ = qiskit.QuantumCircuit(1, 1)
            circ.sx(0)  # Brings the qubit to the X Axis
            if dt_unit:
                circ.delay(delay_dt, 0, "dt")
            else:
                circ.delay(delay, 0, "s")
            circ.rz(rotation_angle, 0)
            circ.barrier(0)
            circ.sx(0)
            circ.barrier(0)
            circ.measure(0, 0)

            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits[0],
                "xval": real_delay_in_sec,
                "osc_freq": self.experiment_options.osc_freq,
                "unit": "s",
            }

            circuits.append(circ)

        return circuits

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata
