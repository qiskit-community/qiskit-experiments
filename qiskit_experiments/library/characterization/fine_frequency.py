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

"""Fine frequency characterization experiment."""

from typing import List, Optional
import numpy as np

from qiskit import QuantumCircuit, schedule
from qiskit.providers.backend import Backend

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.library.characterization.analysis import FineAmplitudeAnalysis


class FineFrequency(BaseExperiment):
    r"""An experiment to make a fine measurement of the qubit frequency.

    The fine frequency characterization experiment seeks to measure the qubit frequency
    by moving to the equator of the Bloch sphere with a sx gate and idling for a time
    :math:`n\cdot \tau` where :math:`\tau` is the duration of the single-qubit gate and
    :math:`n` is an integer which ranges from zero to a maximal value in integer steps.
    The idle time is followed by rz rotation with an angle :math:`n\pi/2` and a final
    sx gate. If the frequency of the pulses matches the frequency of the qubit then the
    sequence :math:`n\in[0,1,2,3,4,...]` will result in the sequence of measured qubit
    populations :math:`[1, 0.5, 0, 0.5, 1, ...]` due to the rz rotation. If the frequency
    of the pulses does not match the qubit frequency then the qubit will precess in the
    drive frame during the idle time and phase error will build-up. By fitting the measured
    points we can extract the value of the error in frequency. The circuit that are run are

    .. parsed-literal::

                ┌────┐┌────────────────┐┌──────────┐┌────┐ ░ ┌─┐
             q: ┤ √X ├┤ Delay(n*τ[dt]) ├┤ Rz(nπ/2) ├┤ √X ├─░─┤M├
                └────┘└────────────────┘└──────────┘└────┘ ░ └╥┘
        meas: 1/══════════════════════════════════════════════╩═
                                                              0

    """

    __analysis_class__ = FineAmplitudeAnalysis

    def __init__(self, qubit, backend: Optional[Backend] = None):
        """Setup a fine frequency experiment on the given qubit.

        Args:
            qubit: The qubit on which to run the fine amplitude calibration experiment.
            gate: The gate that will be repeated.
            backend: Optional, the backend to run the experiment on.
        """
        super().__init__([qubit], backend=backend)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine amplitude experiment.

        Experiment Options:
            repetitions (List[int]): A list of the number of times that the identity is repeated.
        """
        options = super()._default_experiment_options()
        options.repetitions = list(range(40))

        return options

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()
        options.angle_per_gate = np.pi
        options.phase_offset = np.pi / 2

        return options

    def _pre_circuit(self) -> QuantumCircuit:
        """A method that subclasses can override to perform gates before the main sequence."""
        return QuantumCircuit(1)

    def circuits(self) -> List[QuantumCircuit]:
        """Return the list of quantum circuits to run."""

        # Find out the duration of the sx gate on the backend
        circ = QuantumCircuit(1)
        circ.sx(0)
        duration = schedule(circ, self.backend).duration

        circuits = []

        # Measure 0
        qc0 = self._pre_circuit()
        qc0.measure_all()

        # Measure 1
        qc1 = self._pre_circuit()
        qc1.x(0)
        qc1.measure_all()
        circuits.extend([qc0, qc1])

        # The main sequence
        for repetition in self.experiment_options.repetitions:
            circuit = self._pre_circuit()
            circuit.sx(0)

            circuit.delay(duration=duration*repetition)

            circuit.rz(np.pi * repetition / 2, 0)
            circuit.sx(0)
            circuit.measure_all()

            circuits.append(circuit)

        return circuits
