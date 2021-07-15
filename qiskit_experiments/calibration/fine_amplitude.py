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
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.qobj.utils import MeasLevel
from qiskit.providers import Backend
from qiskit.providers.options import Options
from qiskit.pulse.schedule import ScheduleBlock

from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.calibration.analysis.fine_amplitude_analysis import FineAmplitudeAnalysis
from qiskit_experiments.exceptions import CalibrationError


class FineAmplitude(BaseExperiment):
    r"""Error amplifying fine amplitude calibration experiment.

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

    .. code-block:: python

        qubit = 3
        amp_cal = FineAmplitude(qubit)
        amp_cal.set_schedule(
            schedule=x90p,
            angle_per_gate=np.pi/2,
            add_xp_circuit=False,
            add_sx=False
        )
        amp_cal.run(backend)

    Note that the schedule and angle_per_gate could have been set by independently calling
    :meth:`set_experiment_options` for the schedule and :meth:`set_analysis_options` for
    the angle_per_gate.

    Error amplifying experiments are most sensitive to angle errors when we measure points along
    the equator of the Block sphere. This is why users should insert a square-root of X pulse
    before running calibrations for :math:`\pm\pi` rotations. Furthermore, when running
    calibrations for :math:`\pm\pi/2` rotations users are advised to use an odd number of
    repetitions, e.g. [1, 2, 3, 5, 7, ...] to ensure that the ideal points are on the equator
    of the Bloch sphere. Note the presence of two repetitions which allows us to prepare the
    excited state. Therefore, add_xp_circuit = True is not needed in this case.

    **Summary of experiment options**

    * repetitions: A list with the number of times the gate of interest will be repeated.
    * schedule: The schedule of the gate that will be repeated.
    * add_sx: A boolean which if set to True will add a square-root of X before the repetitions
      of the gate of interest. Set this to True if you are calibrating gates with an ideal
      rotation angle per gate of :math:`\pm\pi`. The default value is False.
    * add_xp_circuit: A boolean which if set to True will cause the experiment to run an
      additional circuit with an X gate and a measurement. This prepares the excited state
      and is typically crucial to get the correct sign for the magnitude of the error in
      the rotation angle. The default value is True.
    * sx_schedule: Allows users to set a schedule for the square-root of X gate.

    **Summary of analysis options**

    * angle_per_gate: The ideal angle per repeated gate. The user must set this options.
    * phase_offset: A phase offset for the analysis. This phase offset will be :math:`\pi/2`
      if the square-root of X gate is added before the repeated gates. This is decided for
      the user in :meth:`set_schedule` depending on whether the sx gate is included in the
      experiment.

    Users can call :meth:`set_schedule` to conveniently set the schedule and the corresponding
    experiment and analysis options.
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
        """Default values for the fine amplitude experiment."""
        options = super()._default_experiment_options()
        options.repetitions = list(range(15))
        options.schedule = None
        options.normalization = True
        options.add_sx = False
        options.add_xp_circuit = True
        options.sx_schedule = None

        return options

    def __init__(self, qubit: int):
        """Setup a fine amplitude experiment on the given qubit.

        Args:
            qubit: The qubit on which to run the fine amplitude calibration experiment.
        """
        super().__init__([qubit])

    def set_schedule(
        self,
        schedule: ScheduleBlock,
        angle_per_gate: float,
        add_xp_circuit: bool,
        add_sx: bool,
    ):
        r"""Set the schedule and its corresponding intended angle per gate.

        Args:
            schedule: The schedule to attache to the gates.
            angle_per_gate: The intended angle per gate used by the analysis method.
            add_xp_circuit: If True then a circuit preparing the excited state is also run.
            add_sx: Whether or not to add a pi-half pulse before running the calibration.

        Raises:
            CalibrationError: If the target angle is a multiple of :math:`2\pi`.
        """
        self.set_experiment_options(schedule=schedule, add_xp_circuit=add_xp_circuit, add_sx=add_sx)

        if np.isclose(angle_per_gate % (2 * np.pi), 0.0):
            raise CalibrationError(
                f"It does not make sense to use {self.__class__.__name__} on a pulse with an "
                "angle_per_gate of zero as the update rule will set the amplitude to zero "
                "angle_per_gate / (angle_per_gate + d_theta)."
            )

        phase_offset = np.pi / 2 if add_sx else 0

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
            CalibrationError: If no schedule was provided.
            CalibrationError: If the channel index does not correspond to the physical qubit index.
            CalibrationError: If the schedule contains unassigned parameters.
            CalibrationError: If the analysis options do not contain the angle_per_gate.
        """

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

        repetitions = self.experiment_options.get("repetitions")

        circuits = []

        if self.experiment_options.add_xp_circuit:
            # Note that the rotation error in this xval will be overweighted when calibrating xp
            # because it will be treated as a half pulse instead of a full pulse. However, since
            # the qubit population is first-order insensitive to rotation errors for an xp pulse
            # this point won't contribute much to inferring the angle error.
            angle_per_gate = self.analysis_options.get("angle_per_gate", None)
            phase_offset = self.analysis_options.get("phase_offset")

            if angle_per_gate is None:
                raise CalibrationError(
                    f"Unknown angle_per_gate for {self.__class__.__name__}. "
                    "Please set it in the analysis options."
                )

            circuit = QuantumCircuit(1)
            circuit.x(0)
            circuit.measure_all()

            circuit.metadata = {
                "experiment_type": self._type,
                "qubits": (self.physical_qubits[0],),
                "xval": (np.pi - phase_offset) / angle_per_gate,
                "unit": "gate number",
            }

            circuits.append(circuit)

        for repetition in repetitions:
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


class FineXAmplitude(FineAmplitude):
    r"""A fine amplitude experiment with all the options set for the :math:`\pi`-rotation."""

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default values for the fine amplitude experiment."""
        options = super()._default_experiment_options()
        options.add_sx = True
        options.add_xp_circuit = True

        return options

    def __init__(self, qubit: int):
        """Setup a fine amplitude experiment on the given qubit.

        Args:
            qubit: The qubit on which to run the fine amplitude calibration experiment.
        """
        super().__init__(qubit)
        self.set_analysis_options(angle_per_gate=np.pi, phase_offset=np.pi / 2)


class FineSXAmplitude(FineAmplitude):
    r"""A fine amplitude experiment with all the options set for the :math:`\pi/2`-rotation."""

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default values for the fine amplitude experiment."""
        options = super()._default_experiment_options()
        options.add_sx = False
        options.add_xp_circuit = False
        options.repetitions = [1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 21, 23, 25]

        return options

    def __init__(self, qubit: int):
        """Setup a fine amplitude experiment on the given qubit.

        Args:
            qubit: The qubit on which to run the fine amplitude calibration experiment.
        """
        super().__init__(qubit)
        self.set_analysis_options(angle_per_gate=np.pi / 2, phase_offset=0)
