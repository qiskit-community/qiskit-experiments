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
T2Hahn Echo Experiment class.

"""

from typing import List, Optional, Union
import numpy as np

from qiskit.utils import apply_prefix
from qiskit import QuantumCircuit, QiskitError
from qiskit.providers.backend import Backend
from qiskit.test.mock import FakeBackend

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

        """
    __analysis_class__ = T2HahnAnalysis

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            delays (Iterable[float]): Delay times of the experiments.
            unit (str): Unit of the delay times. Supported units are
                's', 'ms', 'us', 'ns', 'ps', 'dt'.
        """
        options = super()._default_experiment_options()

        options.delays = None
        options.unit = "s"
        options.conversion_factor = None
        options.osc_freq = 0.0
        options.num_echoes = 1
        return options

    def __init__(
            self,
            qubit: int,
            delays: Union[List[float], np.array],
            backend: Optional[Backend] = None,
            unit: str = "s",
    ):
        """
        Initialize the T2 - Hahn Echo class

        Args:
            qubit:  the qubit whose T2 is to be estimated
            delays: Total delay times of the experiments.
			backend: Optional, the backend to run the experiment on.
            unit: Optional, time unit of `delays`.
                Supported units: 's', 'ms', 'us', 'ns', 'ps', 'dt'.

         Raises:
             QiskitError : Error for invalid input.
        """
        # Initialize base experiment
        super().__init__([qubit], backend=backend)

        # Set experiment options
        self.set_experiment_options(delays=delays, unit=unit)
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
        if not self._backend.configuration().simulator and not isinstance(backend, FakeBackend):
            timing_constraints = getattr(self.transpile_options, "timing_constraints", {})
            if "acquire_alignment" not in timing_constraints:
                timing_constraints["acquire_alignment"] = 16
            scheduling_method = getattr(self.transpile_options, "scheduling_method", "alap")
            self.set_transpile_options(
                timing_constraints=timing_constraints, scheduling_method=scheduling_method
            )

        # Set conversion factor
        if self.experiment_options.unit == "dt":
            try:
                dt_factor = getattr(self.backend.configuration(), "dt")
                conversion_factor = dt_factor
            except AttributeError as no_dt:
                raise AttributeError("Dt parameter is missing in backend configuration") from no_dt
        elif self.experiment_options.unit != "s":
            conversion_factor = apply_prefix(1, self.experiment_options.unit)
        else:
            conversion_factor = 1
        self.set_experiment_options(conversion_factor=conversion_factor)

    def circuits(self) -> List[QuantumCircuit]:
        """
        Return a list of experiment circuits

        Returns:
            The experiment circuits

        Raises:
            AttributeError: if unit is 'dt', but 'dt' parameter is missing in the backend configuration
        """
        if self.backend:
            self._set_backend(self.backend)
        prefactor = self.experiment_options.conversion_factor

        if prefactor is None:
            raise ValueError("Conversion factor is not set.")

        circuits = []
        for delay_gate in prefactor * np.asarray(self.experiment_options.delays, dtype=float):
            total_delay = delay_gate * (self.experiment_options.num_echoes + 1)
            # delay_gate = delay

            delay_gate = np.round(delay_gate, decimals=10)

            circ = QuantumCircuit(1, 1)

            # First X rotation in 90 degrees
            circ.rx(np.pi / 2, 0)  # Bring to qubits to X Axis
            for idx in range(self.experiment_options.num_echoes):
                circ.delay(delay_gate, 0, self.experiment_options.unit)
                circ.rx(np.pi, 0)
                circ.delay(delay_gate, 0, self.experiment_options.unit)
            if self.experiment_options.num_echoes % 2 == 1:
                circ.rx(np.pi / 2, 0)  # X90 again since the num of echoes is odd
            else:
                circ.rx(-np.pi / 2, 0)  # X90 again since the num of echoes is even
            circ.measure(0, 0)  # measure
            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits[0],
                "xval": total_delay,
                "unit": "s",
            }

            circuits.append(circ)

        return circuits
