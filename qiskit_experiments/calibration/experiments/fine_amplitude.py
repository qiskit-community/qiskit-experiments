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

"""Fine amplitude calibration experiment."""

from typing import List, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.qobj.utils import MeasLevel
from qiskit.providers import Backend
from qiskit.providers.options import Options
from qiskit.pulse.schedule import ScheduleBlock

from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.data_processing.processor_library import get_to_signal_processor
from qiskit_experiments.calibration.experiments.fine_amplitude_analysis import FineAmplitudeAnalysis
from qiskit_experiments.calibration.exceptions import CalibrationError


class FineAmplitude(BaseExperiment):
    """Error amplifying fine amplitude calibration experiment.

    The :class:`FineAmplitude` calibration experiment repeats N times a gate with a pulse
    to amplify the under-/over-rotations in the gate to determine the optimal amplitude.
    The circuits that are run have a custom gate with the pulse schedule attached to it
    through the calibrations. The circuits are therefore of the form:

    .. parsed-literal::

                   ┌─────┐       ┌─────┐ ░ ┌─┐
              q_0: ┤ Cal ├─ ... ─┤ Cal ├─░─┤M├
                   └─────┘       └─────┘ ░ └╥┘
        measure: 1/════════ ... ════════════╩═
                                            0

    Here, Cal is the name of the gate which will be taken from the name of the schedule.
    The user can optionally add a square-root of X pulse before the Cal gates are repeated.
    This square-root of X pulse allows the analysis to differentiate between over rotations
    and under rotations in the case of pi-pulses. Importantly, the resulting data is analyzed
    by a fit to a cosine function in which we try and determine the over/under rotation given
    an intended rotation angle per gate which must also be specified by the user. The steps
    to run a fine amplitude calibration experiment are therefore

    .. code-block:

        qubit = 3
        amp_cal = FineAmplitude(qubit)
        amp_cal.set_schedule(schedule=x45p, angle_per_gate=np.pi/4)
        amp_cal.run(backend)

    Note that the schedule and angle_per_gate could have been set by independently calling
    :meth:`set_experiment_options` for the schedule and :meth:`set_analysis_options` for
    the angle_per_gate.
    """

    __analysis_class__ = FineAmplitudeAnalysis

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default option values for the experiment :meth:`run` method."""
        return Options(
            meas_level=MeasLevel.CLASSIFIED,
            meas_return="avg",
        )

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default values for the fine amplitude experiment.

        Users can set the schedule by doing

        .. code-block::

            amp_cal.set_experiment_options(schedule=my_x90p)

        """
        options = super()._default_experiment_options()
        options.repetitions = 15
        options.schedule = None
        options.normalization = True
        options.add_sx = False
        options.sx_schedule = None

        return options

    def __init__(self, qubit: int):
        """Setup a fine amplitude experiment on the given qubit.

        Args:
            qubit: The qubit on which to run the fine amplitude calibration experiment.
        """
        super().__init__([qubit])

    def set_schedule(self, schedule: ScheduleBlock, angle_per_gate: float, phase_offset: float):
        """Set the schedule and its corresponding intended angle per gate.

        Args:
            schedule: The schedule to attache to the gates.
            angle_per_gate: The intended angle per gate used by the analysis method.
            phase_offset: The phase offset to use in the fit.
        """
        self.set_experiment_options(schedule=schedule)
        self.set_analysis_options(angle_per_gate=angle_per_gate, phase_offset=phase_offset)

    def _pre_circuit(self) -> QuantumCircuit:
        """Return a preparation circuit.

        This method can be overridden by subclasses e.g. to calibrate schedules on
        transitions other than the 0 <-> 1 transition.
        """
        circuit = QuantumCircuit(1)

        if self.experiment_options.add_sx:
            circuit.sx(0)

        if self.experiment_options.sx_schedule is not None:
            sx_schedule = self.experiment_options.sx_schedule
            circuit.add_calibration("sx", (self.physical_qubits[0],), sx_schedule, params=[])
            circuit.barrier()

        return circuit

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Create the circuits for the fine amplitude calibration experiment.

        Args:
            backend: A backend object.

        Returns:
            A list of circuits with a variable number of gates. Each gate has the same
            pulse schedule.

        Raises:
            CalibrationError:
                - If no schedule was provided.
                - If the channel index does not correspond to the physical qubit index.
                - If the schedule contains unassigned parameters.
        """
        # TODO this is temporary logic. Need update of circuit data and processor logic.
        self.set_analysis_options(
            data_processor=get_to_signal_processor(
                meas_level=self.run_options.meas_level,
                meas_return=self.run_options.meas_return,
                normalize=self.experiment_options.normalization,
            )
        )

        # Get the schedule and check assumptions.
        schedule = self.experiment_options.get("schedule", None)

        if schedule is None:
            raise CalibrationError("No schedule set for fine amplitude calibration.")

        if self.physical_qubits[0] not in set(ch.index for ch in schedule.channels):
            raise CalibrationError(
                f"User provided schedule {schedule.name} does not contain a channel "
                "for the qubit on which to run the fine amplitude calibration."
            )

        if len(schedule.parameters) > 0:
            raise CalibrationError(
                "All parameters in a fine amplitude calibration schedule must be bound. "
                f"Unbound parameters: {schedule.parameters}"
            )

        # Prepare the circuits.
        gate = Gate(name=schedule.name, num_qubits=1, params=[])

        repetitions = self.experiment_options.get("repetitions", 15)

        circuits = []
        for repetition in range(repetitions):
            circuit = self._pre_circuit()

            for _ in range(repetition):
                circuit.append(gate, (0,))

            circuit.measure_all()
            circuit.add_calibration(gate, (self.physical_qubits[0],), schedule, params=[])

            circuit.metadata = {
                "experiment_type": self._type,
                "qubits": (self.physical_qubits[0],),
                "xval": repetition,
                "unit": "gate number",
            }

            circuits.append(circuit)

        return circuits
