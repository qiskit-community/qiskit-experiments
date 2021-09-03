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

"""Fine DRAG calibration experiment."""

from typing import Optional, List
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.qobj.utils import MeasLevel
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
import qiskit.pulse as pulse

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.library.calibration.analysis.fine_amplitude_analysis import (
    FineAmplitudeAnalysis,
)


class FineDrag(BaseExperiment):
    r"""Fine DRAG Calibration experiment.

    # section: overview

        The class :class:`FineDrag` runs fine DRAG calibration experiments (see :class:`DragCal`
        for the definition of DRAG pulses). Fine DRAG calibration proceeds by iterating the
        gate sequence Rp - Rm where Rp is a rotation around an axis and Rm is the same rotation
        but in the opposite direction. The circuits that are executed are of the form

        .. parsed-literal::

                    ┌─────┐┌────┐┌────┐     ┌────┐┌────┐┌──────┐ ░ ┌─┐
               q_0: ┤ Pre ├┤ Rp ├┤ Rm ├ ... ┤ Rp ├┤ Rm ├┤ Post ├─░─┤M├
                    └─────┘└────┘└────┘     └────┘└────┘└──────┘ ░ └╥┘
            meas: 1/═══════════════════ ... ════════════════════════╩═
                                                                    0

        Here, Pre and Post designate gates that may be pre-appended and and post-appended,
        respectively, to the repeated sequence of Rp and Rm gates. When calibrating a pulse
        with a target rotation angle of π the Pre and Post gates are Id and RYGate(π/2),
        respectively. When calibrating a pulse with a target rotation angle of π/2 the Pre and
        Post gates are RXGate(π/2) and RYGate(π/2), respectively.
    """

    __analysis_class__ = FineAmplitudeAnalysis

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default option values for the experiment :meth:`run` method."""
        options = super()._default_run_options()
        options.meas_level = MeasLevel.CLASSIFIED
        options.meas_return = "avg"

        return options

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine amplitude experiment.

        Experiment Options:
            repetitions (List[int]): A list of the number of times that Rp - Rm gate sequence
                is repeated.
            schedule (ScheduleBlock): The schedule for the plus rotation.
            normalization (bool): If set to True the DataProcessor will normalized the
                measured signal to the interval [0, 1]. Defaults to True.
            sx_schedule (ScheduleBlock): The schedule to attache to the SX gate.
        """
        options = super()._default_experiment_options()
        options.repetitions = list(range(20))
        options.schedule = None
        options.normalization = True
        options.sx_schedule = None

        return options

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()
        options.angle_per_gate = 0.0
        options.phase_offset = np.pi / 2

        return options

    def __init__(self, qubit: int):
        """Setup a fine amplitude experiment on the given qubit.

        Args:
            qubit: The qubit on which to run the fine amplitude calibration experiment.
        """
        super().__init__([qubit])

    @staticmethod
    def _pre_circuit() -> QuantumCircuit:
        """Return the quantum circuit to apply before repeating the Rp and Rm gates."""
        return QuantumCircuit(1)

    @staticmethod
    def _post_circuit() -> QuantumCircuit:
        """Return the quantum circuit to apply after repeating the Rp and Rm gates."""
        circ = QuantumCircuit(1)
        circ.ry(np.pi / 2, 0)
        return circ

    # TODO Remove once #251 gets merged.
    def _set_anti_schedule(self, schedule) -> ScheduleBlock:
        """A DRAG specific method that sets the rm schedule based on rp.
        The rm schedule, i.e. the anti-schedule, is the rp schedule sandwiched
        between two virtual phase gates with angle pi.
        """
        with pulse.build(name="rm") as minus_sched:
            pulse.shift_phase(np.pi, pulse.DriveChannel(self._physical_qubits[0]))
            pulse.call(schedule)
            pulse.shift_phase(-np.pi, pulse.DriveChannel(self._physical_qubits[0]))

        return minus_sched

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Create the circuits for the fine DRAG calibration experiment.

        Args:
            backend: A backend object.

        Returns:
            A list of circuits with a variable number of gates. Each gate has the same
            pulse schedule.
        """

        schedule, qubits = self.experiment_options.schedule, self._physical_qubits

        # Prepare the circuits
        repetitions = self.experiment_options.get("repetitions")

        circuits = []

        for repetition in repetitions:
            circuit = self._pre_circuit()

            for _ in range(repetition):
                circuit.append(Gate(name="Rp", num_qubits=1, params=[]), (0,))
                circuit.append(Gate(name="Rm", num_qubits=1, params=[]), (0,))

            circuit.compose(self._post_circuit(), inplace=True)

            circuit.measure_all()
            circuit.add_calibration("Rp", qubits, schedule, params=[])
            circuit.add_calibration("Rm", qubits, self._set_anti_schedule(schedule), params=[])

            # TODO update with metadata function after #251
            circuit.metadata = {
                "experiment_type": self._type,
                "qubits": (self.physical_qubits[0],),
                "xval": repetition,
                "unit": "gate number",
            }

            circuits.append(circuit)

        return circuits
