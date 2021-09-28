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
from qiskit.providers import Backend
from qiskit.pulse.schedule import ScheduleBlock

from qiskit_experiments.framework import Options
from qiskit_experiments.library.calibration.analysis.fine_amplitude_analysis import (
    FineAmplitudeAnalysis,
)
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.framework.experiment_data import ExperimentData
from qiskit_experiments.calibration_management.update_library import Amplitude
from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.calibration_management.base_calibration_experiment import (
    BaseCalibrationExperiment,
)


class FineAmplitude(BaseCalibrationExperiment):
    r"""Error amplifying fine amplitude calibration experiment.

    # section: overview

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
        an intended rotation angle per gate which must also be specified by the user.

        Error amplifying experiments are most sensitive to angle errors when we measure points along
        the equator of the Block sphere. This is why users should insert a square-root of X pulse
        before running calibrations for :math:`\pm\pi` rotations. Furthermore, when running
        calibrations for :math:`\pm\pi/2` rotations users are advised to use an odd number of
        repetitions, e.g. [1, 2, 3, 5, 7, ...] to ensure that the ideal points are on the equator
        of the Bloch sphere. Note the presence of two repetitions which allows us to prepare the
        excited state. Therefore, add_xp_circuit = True is not needed in this case.

        Users can call :meth:`set_schedule` to conveniently set the schedule and the corresponding
        experiment and analysis options.

    # section: example


        The steps to run a fine amplitude calibration experiment are

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

        Note that the ``schedule`` and ``angle_per_gate`` could have been set by independently calling
        :meth:`set_experiment_options` for the ``schedule`` and :meth:`set_analysis_options` for
        the ``angle_per_gate``.


    # section: reference
        .. ref_arxiv:: 1 1504.06597


    # section: tutorial
        :doc:`/tutorials/fine_amplitude_calibration`

    """

    __analysis_class__ = FineAmplitudeAnalysis

    __updater__ = Amplitude

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine amplitude experiment.

        Experiment Options:
            repetitions (List[int]): A list of the number of times that the gate is repeated.
            schedule (ScheduleBlock): The schedule attached to the gate that will be repeated.
            normalization (bool): If set to True the DataProcessor will normalized the
                measured signal to the interval [0, 1]. Defaults to True.
            add_sx (bool): If True then the circuits will start with an sx gate. This is typically
                needed when calibrating pulses with a target rotation angle of :math:`\pi`. The
                default value is False.
            add_xp_circuit (bool): If set to True then a circuit with only an X gate will also be
                run. This allows the analysis class to determine the correct sign for the amplitude.
            sx_schedule (ScheduleBlock): The schedule to attache to the SX gate.
        """
        options = super()._default_experiment_options()
        options.repetitions = list(range(15))
        options.schedule = None
        options.normalization = True
        options.add_sx = False
        options.add_xp_circuit = True
        options.sx_schedule = None

        return options

    def __init__(
        self,
        qubit: int,
        cals: Optional[Calibrations] = None,
        schedule_name: Optional[str] = None,
        cal_parameter_name: Optional[str] = "amp",
        repetitions: Optional[int] = None,
    ):
        r"""Setup a fine amplitude experiment on the given qubit.

        Args:
            qubit: The qubit on which to run the fine amplitude calibration experiment.
            cals: If calibrations is given then running the experiment will update
                the values of the pulse parameters stored in calibrations.
            schedule_name: The name of the schedule to extract from the calibrations.
            cal_parameter_name: The name of the parameter in calibrations to update. This name will
                be stored in the experiment options and defaults to "amp".
            repetitions: The list of times to repeat the gate in each circuit.
        """
        super().__init__([qubit], cals, schedule_name, cal_parameter_name)

        if repetitions is not None:
            self.experiment_options.repetitions = repetitions

    def validate_schedule(self, schedule: ScheduleBlock):
        """Validate the schedule to calibrate."""
        self._validate_channels(schedule)
        self._validate_parameters(schedule, 0)

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
            CalibrationError: If the analysis options do not contain the angle_per_gate.
        """

        # Get the schedule and check assumptions.
        schedule = self.get_schedule()

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
            circuit.metadata = self._circuit_metadata(xval=(np.pi - phase_offset) / angle_per_gate)
            circuits.append(circuit)

        for repetition in repetitions:
            circuit = self._pre_circuit()

            for _ in range(repetition):
                circuit.append(gate, (0,))

            circuit.measure_all()
            circuit.add_calibration(gate, (self.physical_qubits[0],), schedule, params=[])
            circuit.metadata = self._circuit_metadata(xval=repetition)

            circuits.append(circuit)

        return circuits

    def update_calibrations(self, experiment_data: ExperimentData):
        """Update the calibrations given the experiment data.

        Args:
            experiment_data: The experiment data to use for the update.

        Raises:
            CalibrationError: If the schedule name is None in the calibration options.
        """
        angle = self.analysis_options.angle_per_gate

        if self._sched_name is None:
            raise CalibrationError(
                f"Cannot perform {self.__updater__.__class__.__name__} without a schedule name."
            )

        self.__updater__.update(
            self._cals, experiment_data, angles_schedules=[(angle, self._param_name, self._sched_name)]
        )


class FineXAmplitude(FineAmplitude):
    r"""A fine amplitude experiment with all the options set for the :math:`\pi`-rotation.

    # section: overview

        :class:`FineXAmplitude` is a subclass of :class:`FineAmplitude` and is used to set
        the appropriate values for the default options.
    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine amplitude experiment.

        Experiment Options:
            add_sx (bool): This option is True by default when calibrating gates with a target
                angle per gate of :math:`\pi` as this increases the sensitivity of the
                experiment.
            add_xp_circuit (bool): This option is True by default when calibrating gates with
                a target angle per gate of :math:`\pi`.
        """
        options = super()._default_experiment_options()
        options.add_sx = True
        options.add_xp_circuit = True

        return options

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()
        options.angle_per_gate = np.pi
        options.phase_offset = np.pi / 2

        return options

    def __init__(
        self,
        qubit: int,
        cals: Optional[Calibrations] = None,
        schedule_name: Optional[str] = "x",
        cal_parameter_name: Optional[str] = "amp",
        sx_schedule_name: Optional[str] = "sx",
        repetitions: Optional[int] = None,
    ):
        """Setup a fine amplitude experiment on the given qubit.

        Args:
            qubit: The qubit on which to run the fine amplitude calibration experiment.
            cals: An optional instance of :class:`Calibrations`. If calibrations is
                given then running the experiment will update the values of the pulse parameters
                stored in calibrations.
            schedule_name: The name of the schedule to extract from the calibrations. The default
                value is "x".
            cal_parameter_name: The name of the parameter in calibrations to update. This name will
                be stored in the experiment options and defaults to "amp".
            sx_schedule_name: The name of the schedule to extract from the calibrations for the
                "sx" pulse that will be added.
            repetitions: The list of times to repeat the gate in each circuit.
        """
        super().__init__(qubit, cals, schedule_name, cal_parameter_name, repetitions)

        if cals is not None and sx_schedule_name is not None:
            self.experiment_options.sx_schedule = cals.get_schedule(sx_schedule_name, qubit)


class FineSXAmplitude(FineAmplitude):
    r"""A fine amplitude experiment with all the options set for the :math:`\pi/2`-rotation.

    # section: overview

        :class:`FineSXAmplitude` is a subclass of :class:`FineAmplitude` and is used to set
        the appropriate values for the default options.
    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine amplitude experiment.

        Experiment Options:
            add_sx (bool): This option is False by default when calibrating gates with a target
                angle per gate of :math:`\pi/2` as this increases the sensitivity of the
                experiment.
            add_xp_circuit (bool): This option is False by default when calibrating gates with
                a target angle per gate of :math:`\pi/2`.
            repetitions (List[int]): By default the repetitions take on odd numbers for
                :math:`\pi/2` target angles as this ideally prepares states on the equator of
                the Bloch sphere. Note that the repetitions include two repetitions which
                plays the same role as including a circuit with an X gate.
        """
        options = super()._default_experiment_options()
        options.add_sx = False
        options.add_xp_circuit = False
        options.repetitions = [1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 21, 23, 25]

        return options

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()
        options.angle_per_gate = np.pi / 2
        options.phase_offset = 0

        return options
