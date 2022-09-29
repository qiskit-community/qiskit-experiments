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
T2Hahn Echo Experiment class.
"""

from typing import List, Optional, Union
import numpy as np

from qiskit import QuantumCircuit, QiskitError
from qiskit.providers.backend import Backend

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.library.characterization.analysis.t2hahn_analysis import T2HahnAnalysis


class T2Hahn(BaseExperiment):
    r"""T2 Hahn Echo Experiment.

    # section: overview

        This experiment is used to estimate T2 noise of a single qubit.

        See `Qiskit Textbook <https://qiskit.org/textbook/ch-quantum-hardware/\
        calibrating-qubits-pulse.html>`_  for a more detailed explanation on
        these properties.

        This experiment consists of a series of circuits of the form


        .. parsed-literal::

                 ┌─────────┐┌──────────┐┌───────┐┌──────────┐┌─────────┐┌─┐
            q_0: ┤ Rx(π/2) ├┤ DELAY(t) ├┤ RX(π) ├┤ DELAY(t) ├┤ RX(π/2) ├┤M├
                 └─────────┘└──────────┘└───────┘└──────────┘└─────────┘└╥┘
            c: 1/════════════════════════════════════════════════════════╩═
                                                                         0

        for each *t* from the specified delay times
        and the delays are specified by the user.
        The delays that are specified are delay for each delay gate while
        the delay in the metadata is the total delay which is delay * (num_echoes +1)
        The circuits are run on the device or on a simulator backend.

    # section: tutorial
        :doc:`/tutorials/t2hahn_characterization`

    # section: analysis_ref
        :py:class:`T2HahnAnalysis`
    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            delays (Iterable[float]): Delay times of the experiments.
        """
        options = super()._default_experiment_options()

        options.delays = None
        options.num_echoes = 1
        return options

    def __init__(
        self,
        qubit: int,
        delays: Union[List[float], np.array],
        num_echoes: int = 1,
        backend: Optional[Backend] = None,
    ):
        """
        Initialize the T2 - Hahn Echo class

        Args:
            qubit:  the qubit whose T2 is to be estimated
            delays: Total delay times of the experiments.
                        backend: Optional, the backend to run the experiment on.
            num_echoes: The number of echoes to preform.
            backend: Optional, the backend to run the experiment on..

         Raises:
             QiskitError : Error for invalid input.
        """
        # Initialize base experiment
        super().__init__([qubit], analysis=T2HahnAnalysis(), backend=backend)

        # Set experiment options
        self.set_experiment_options(delays=delays, num_echoes=num_echoes)
        self._verify_parameters()

    def _verify_parameters(self):
        """
        Verify input correctness, raise QiskitError if needed.

        Raises:
            QiskitError : Error for invalid input.
        """
        if any(delay < 0 for delay in self.experiment_options.delays):
            raise QiskitError(
                f"The lengths list {self.experiment_options.delays} should only contain "
                "non-negative elements."
            )

    def _set_backend(self, backend: Backend):
        super()._set_backend(backend)

        # Scheduling parameters
        if not self._backend_data.is_simulator:
            scheduling_method = getattr(self.transpile_options, "scheduling_method", "alap")
            self.set_transpile_options(scheduling_method=scheduling_method)

    def circuits(self) -> List[QuantumCircuit]:
        """
        Return a list of experiment circuits.

        Each circuit consists of RX(π/2) followed by a sequence of delay gate,
        RX(π) for echo and delay gate again.
        The sequence repeats for the number of echoes and terminates with RX(±π/2).

        Returns:
            The experiment circuits.
        """

        dt_unit = False
        if self.backend:
            dt_factor = self._backend_data.dt
            dt_unit = dt_factor is not None

        circuits = []
        for delay_gate in np.asarray(self.experiment_options.delays, dtype=float):
            if dt_unit:
                delay_dt = round(delay_gate / dt_factor)
                real_delay_in_sec = delay_dt * dt_factor
            else:
                real_delay_in_sec = delay_gate

            total_delay = real_delay_in_sec * (self.experiment_options.num_echoes * 2)

            circ = QuantumCircuit(1, 1)

            # First X rotation in 90 degrees
            circ.rx(np.pi / 2, 0)  # Brings the qubit to the X Axis
            for _ in range(self.experiment_options.num_echoes):
                if dt_unit:
                    circ.delay(delay_dt, 0, "dt")
                    circ.rx(np.pi, 0)
                    circ.delay(delay_dt, 0, "dt")
                else:
                    circ.delay(delay_gate, 0, "s")
                    circ.rx(np.pi, 0)
                    circ.delay(delay_gate, 0, "s")

            # if number of echoes is 0 then just apply the delay gate
            if self.experiment_options.num_echoes == 0:
                if dt_unit:
                    total_delay = real_delay_in_sec
                    circ.delay(delay_dt, 0, "dt")
                else:
                    total_delay = real_delay_in_sec
                    circ.delay(delay_gate, 0, "s")

            if self.experiment_options.num_echoes % 2 == 1:
                circ.rx(np.pi / 2, 0)  # X90 again since the num of echoes is odd
            else:
                circ.rx(-np.pi / 2, 0)  # X(-90) again since the num of echoes is even
            circ.measure(0, 0)  # measure
            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits[0],
                "xval": total_delay,
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
