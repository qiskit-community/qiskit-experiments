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

from qiskit import QuantumCircuit, schedule, transpile
from qiskit.providers.backend import Backend

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.library.characterization.analysis import FineAmplitudeAnalysis


class FineFrequency(BaseExperiment):
    r"""An experiment to make a fine measurement of the qubit frequency.

    # section: overview

        The fine frequency characterization experiment measures the qubit frequency by moving
        to the equator of the Bloch sphere with a sx gate and idling for a time
        :math:`n\cdot \tau` where :math:`\tau` is the duration of the single-qubit gate and
        :math:`n` is an integer which ranges from zero to a maximal value in integer steps.
        The idle time is followed by a rz rotation with an angle :math:`n\pi/2` and a final
        sx gate. If the frequency of the pulses match the frequency of the qubit then the
        sequence :math:`n\in[0,1,2,3,4,...]` will result in the sequence of measured qubit
        populations :math:`[1, 0.5, 0, 0.5, 1, ...]` due to the rz rotation. If the frequency
        of the pulses do not match the qubit frequency then the qubit will precess in the
        drive frame during the idle time and phase errors will build-up. By fitting the measured
        points we can extract the error in the qubit frequency. The circuit that are run are

        .. parsed-literal::

                    ┌────┐┌────────────────┐┌──────────┐┌────┐ ░ ┌─┐
                 q: ┤ √X ├┤ Delay(n*τ[dt]) ├┤ Rz(nπ/2) ├┤ √X ├─░─┤M├
                    └────┘└────────────────┘└──────────┘└────┘ ░ └╥┘
            meas: 1/══════════════════════════════════════════════╩═
                                                                  0

    """

    def __init__(
        self, qubit: int, backend: Optional[Backend] = None, repetitions: Optional[List[int]] = None
    ):
        """Setup a fine frequency experiment on the given qubit.

        Args:
            qubit: The qubit on which to run the fine frequency characterization experiment.
            backend: Optional, the backend to run the experiment on.
            repetitions: The number of repetitions, if not given then the default value
                from the experiment default options will be used.
        """
        super().__init__([qubit], analysis=FineAmplitudeAnalysis(), backend=backend)

        # Set default analysis options
        self.analysis.set_options(angle_per_gate=np.pi / 2, phase_offset=0)

        if repetitions is not None:
            self.set_experiment_options(repetitions=repetitions)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine frequency experiment.

        Experiment Options:
            repetitions (List[int]): A list of the number of times that the identity is repeated.
            sq_gate_duration (int): The duration of the single-qubit gate as the number of arbitrary
                waveform generator samples it contains.
        """
        options = super()._default_experiment_options()
        options.repetitions = list(range(40))
        options.sq_gate_duration = None

        return options

    @classmethod
    def _default_transpile_options(cls) -> Options:
        """Default transpiler options."""
        options = super()._default_transpile_options()
        options.inst_map = None
        return options

    def _pre_circuit(self) -> QuantumCircuit:
        """A method that subclasses can override to perform gates before the main sequence."""
        return QuantumCircuit(1)

    def circuits(self) -> List[QuantumCircuit]:
        """Return the list of quantum circuits to run."""

        # Find out the duration of the sx gate from instructions map or the backend if missing.
        if self.experiment_options.sq_gate_duration is None:
            circuit = QuantumCircuit(1)
            circuit.sx(0)
            if self.transpile_options.inst_map is not None:
                inst_map = self.transpile_options.inst_map
                circuit = transpile(circuit, initial_layout=self.physical_qubits, inst_map=inst_map)
            duration = schedule(circuit, self.backend).duration
            self.set_experiment_options(sq_gate_duration=duration)

        circuits = []

        # The main sequence
        for repetition in self.experiment_options.repetitions:
            circuit = self._pre_circuit()
            circuit.sx(0)

            circuit.delay(duration=self.experiment_options.sq_gate_duration * repetition)

            circuit.rz(np.pi * repetition / 2, 0)
            circuit.sx(0)
            circuit.measure_all()

            circuit.metadata = {
                "experiment_type": self._type,
                "qubits": self.physical_qubits,
                "xval": repetition,
                "unit": "Id gate number",
            }

            circuits.append(circuit)

        return circuits
