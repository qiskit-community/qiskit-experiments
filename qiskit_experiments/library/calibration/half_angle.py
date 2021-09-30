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

"""Half angle calibration."""

from typing import Optional
import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers import Backend

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.library.calibration.analysis.fine_amplitude_analysis import (
    FineAmplitudeAnalysis,
)


class HalfAngle(BaseExperiment):
    """A calibration experiment class to perform half angle calibration.

    This experiment runs circuits that repeat blocks of :code:`Rx(π/2) - Rx(π/2) - Ry(π)`
    circuits. This gate sequence is designed to amplify errors in the rotation axis of the
    X and Y rotations. Such errors can occur due to ...

    .. parsed-literal::

                ┌─────────┐┌─────────┐┌─────────┐┌───────┐   ┌─────────┐┌─────────┐»
           q_0: ┤ Ry(π/2) ├┤ Rx(π/2) ├┤ Rx(π/2) ├┤ Ry(π) ├...┤ Rx(π/2) ├┤ Rx(π/2) ├»
                └─────────┘└─────────┘└─────────┘└───────┘   └─────────┘└─────────┘»
        meas: 1/══════════════════════════════════════════...══════════════════════»
                                                                                   »
        «        ┌───────┐┌─────────┐ ░ ┌─┐
        «   q_0: ┤ Ry(π) ├┤ Rx(π/2) ├─░─┤M├
        «        └───────┘└─────────┘ ░ └╥┘
        «meas: 1/════════════════════════╩═
        «                                0


    """

    __analysis_class__ = FineAmplitudeAnalysis

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine amplitude experiment.

        Experiment Options:
            repetitions (List[int]): A list of the number of times that the gate
                sequence :code:`[sx sx y]` is repeated.
            normalization (bool): If set to True the DataProcessor will normalized the
                measured signal to the interval [0, 1]. Defaults to True.
        """
        options = super()._default_experiment_options()
        options.repetitions = list(range(15))
        options.normalization = True

        return options

    def __init__(self, qubit: int):
        """Setup a fine amplitude experiment on the given qubit.

        Args:
            qubit: The qubit on which to run the fine amplitude calibration experiment.
        """
        super().__init__([qubit])

    @staticmethod
    def _pre_circuit() -> QuantumCircuit:
        """Return the preparation circuit for the experiment."""
        return QuantumCircuit(1)

    def circuits(self, backend: Optional[Backend] = None):
        """Create the circuits for the half angle calibration experiment."""

        circuits = []

        for repetition in self.experiment_options.repetitions:
            circuit = self._pre_circuit()

            circuit.ry(np.pi / 2, 0)

            for _ in range(repetition):
                circuit.rx(np.pi / 2, 0)
                circuit.rx(np.pi / 2, 0)
                circuit.ry(np.pi, 0)

            circuit.rx(np.pi / 2, 0)
            circuit.measure_all()

            circuits.append(circuit)

        return circuits
