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

    # section: overview

        This experiment runs circuits that repeat blocks of :code:`sx - sx - y`
        circuits inserted in a Ramsey type experiment, i.e. the full gate sequence is thus
        :code:`Ry(π/2) - [sx - sx - y] ^ n - sx` where :code:`n` is varied, see [1]. This
        gate sequence is designed to amplify X-Y axis errors. Such errors can occur due to
        phase errors.

        .. parsed-literal::

                    ┌─────────┐┌────┐┌────┐┌───┐   ┌────┐┌────┐┌───┐┌────┐ ░ ┌─┐
               q_0: ┤ Ry(π/2) ├┤ sx ├┤ sx ├┤ y ├...┤ sx ├┤ sx ├┤ y ├┤ sx ├─░─┤M├
                    └─────────┘└────┘└────┘└───┘   └────┘└────┘└───┘└────┘ ░ └╥┘
            meas: 1/════════════════════════════...═══════════════════════════╩═
                                                                              0

    # section: reference
        .. ref_arxiv:: 1 1504.06597
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

        return options

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()
        options.normalization = True
        options.angle_per_gate = 0.0
        options.phase_offset = 0.0

        return options

    @classmethod
    def _default_transpile_options(cls) -> Options:
        """Default transpile options.

        Experiment Options:
            inst_map (InstructionScheduleMap): An instance of an instruction schedule map
                to bring in the pulses for the rx, and ry rotations.
        """
        options = super()._default_transpile_options()
        options.inst_map = None
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

            # First ry gate
            circuit.rz(np.pi / 2, 0)
            circuit.sx(0)
            circuit.rz(-np.pi / 2, 0)

            # Error amplifying sequence
            for _ in range(repetition):
                circuit.sx(0)
                circuit.sx(0)
                circuit.y(0)

            circuit.sx(0)
            circuit.measure_all()

            circuit.metadata = {
                "experiment_type": self._type,
                "qubits": self.physical_qubits,
                "xval": repetition,
                "unit": "repetition number",
            }

            circuits.append(circuit)

        return circuits
