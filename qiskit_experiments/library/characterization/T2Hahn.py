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

from typing import Union, Iterable, List, Optional

import numpy as np
from qiskit import QuantumCircuit, QiskitError
from qiskit.circuit import Measure
from qiskit.providers.options import Options
from qiskit.providers import Backend

from qiskit_experiments.framework import BaseExperiment


class T2Hahn(BaseExperiment):
    r"""T2 Ramsey Experiment.

        # section: overview

            This experiment is used to estimate T2 noise of a single qubit.

            See `Qiskit Textbook <https://qiskit.org/textbook/ch-quantum-hardware/\
            calibrating-qubits-pulse.html>`_  for a more detailed explanation on
            these properties.

            This experiment consists of a series of circuits of the form

            .. parsed-literal::

                 ┌─────────┐┌──────────┐┌───────┐┌──────────┐┌──────────┐┌─┐
            q_0: ┤ RY(π/2) ├┤ DELAY(t) ├┤ RX(π) ├┤ DELAY(t) ├┤ RY(-π/2) ├┤M├
                 └─────────┘└──────────┘└───────┘└──────────┘└──────────┘└╥┘
            c: 1/═════════════════════════════════════════════════════════╩═
                                                                         0

            for each *t* from the specified delay times, where
            :math:`\lambda =2 \pi \times {osc\_freq}`,
            and the delays are specified by the user.
            The circuits are run on the device or on a simulator backend.

        # section: tutorial
            :doc:`/tutorials/t2ramsey_characterization`

        """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            delays (Iterable[float]): Delay times of the experiments.
            unit (str): Unit of the delay times. Supported units are
                's'.
            n_echos (int); Number of echoes to preform.
            phase_alt_echo (bool): If to use alternate echoes (must have n_echoes greater than 1)
        """
        options = super()._default_experiment_options()

        options.qubit = []
        options.delays = None
        options.unit = "s"
        options.n_echos = 1
        options.phase_alt_echo = False

        return options

    def __init__(
        self,
        qubit: Union[int, Iterable[int]],
        delays: Union[List[float], np.array],  # need to change name?
        n_echos: int = 1,
        phase_alt_echo: bool = False,
    ):
        """
        **T2 - Hahn Echo class**
        Initialize the T2 - Hahn Echo class
         Args:
             qubit: the qubit under test.
             delays (List[float)): delay times of the experiments.
             n_echos (int): Amount of Echoes to preform.
             phase_alt_echo (bool): if to use alternate echo methods

         Raises:
             Error for invalid input.
        """
        # Initialize base experiment
        super().__init__(qubit)
        # Set configurable options
        self.set_experiment_options(
            delays=delays, n_echos=n_echos, phase_alt_echo=phase_alt_echo, qubit=qubit
        )
        self._verify_parameters()

    def _verify_parameters(self):
        """
        Verify input correctness, raise QiskitError if needed.
        Args:
            qubit: the qubit under test.

        Raises:
            QiskitError : Error for invalid input.
        """
        if any(delay <= 0 for delay in self.experiment_options.delays):
            raise QiskitError(
                f"The lengths list {self.experiment_options.delays} should only contain "
                "positive elements."
            )
        if len(set(self.experiment_options.delays)) != len(self.experiment_options.delays):
            raise QiskitError(
                f"The lengths list {self.experiment_options.delays} should not contain "
                "duplicate elements."
            )

        if any(
            self.experiment_options.delays[idx - 1] >= self.experiment_options.delays[idx]
            for idx in range(1, len(self.experiment_options.delays))
        ):
            raise QiskitError(
                f"The number of identity gates {self.experiment_options.delays} should "
                "be increasing."
            )

        if isinstance(self.experiment_options.qubit, list):
            if len(self.experiment_options.qubit) != 1:
                raise QiskitError(
                    "The experiment if for 1 qubit. For multiple qubits,"
                    " please use parallel experiments."
                )
            if self.experiment_options.qubit[0] < 0:
                raise QiskitError(
                    f"The index of the qubit {self.experiment_options.qubit[0]} should "
                    "be non-negative."
                )
        else:
            if self.experiment_options.qubit < 0:
                raise QiskitError(
                    f"The index of the qubit {self.experiment_options.qubit} should "
                    "be non-negative."
                )

        if self.experiment_options.n_echos < 1:
            raise QiskitError(
                f"The number of echoes {self.experiment_options.n_echos} should " "be at least 1."
            )

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """
        Args:
            backend: Optional, a backend object.

        Returns:
            The experiment circuits.

        """

        circuits = []
        qubit = self.experiment_options.qubit
        for circ_index, delay in enumerate(self.experiment_options.delays):
            circ = QuantumCircuit(max(qubit) + 1, len(qubit))
            circ.name = "t2circuit_" + str(circ_index) + "_0"
            # First Y rotation in 90 degrees
            circ.ry(np.pi / 2, qubit)  # Bring to qubits to X Axis
            circ.delay(delay, qubit, self.experiment_options.unit)
            circ.rx(np.pi, qubit)
            circ.delay(delay, qubit, self.experiment_options.unit)
            circ.ry(-np.pi / 2, qubit)  # Y90
            circ.append(Measure(), qubit, [0])  # measure
            circ.metadata = {
                "experiment_type": self._type,
                "qubit": self.physical_qubits,
                "xval": delay,
                "unit": self.experiment_options.unit,
            }
            circuits.append(circ)

        return circuits
