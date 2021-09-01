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

"""Rabi amplitude experiment."""

from typing import List, Optional, Tuple
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.qobj.utils import MeasLevel
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
import qiskit.pulse as pulse

from qiskit_experiments.framework import Options
from qiskit_experiments.curve_analysis import ParameterRepr
from qiskit_experiments.framework.experiment_data import ExperimentData
from qiskit_experiments.library.calibration.analysis.oscillation_analysis import OscillationAnalysis
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management.update_library import Amplitude
from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.calibration_management.base_calibration_experiment import (
    BaseCalibrationExperiment,
)


class Rabi(BaseCalibrationExperiment):
    """An experiment that scans the amplitude of a pulse to calibrate rotations between 0 and 1.

    # section: overview

        The circuits that are run have a custom rabi gate with the pulse schedule attached to it
        through the calibrations. The circuits are of the form:

        .. parsed-literal::

                       ┌───────────┐ ░ ┌─┐
                  q_0: ┤ Rabi(amp) ├─░─┤M├
                       └───────────┘ ░ └╥┘
            measure: 1/═════════════════╩═
                                        0

        If the user provides his own schedule for the Rabi then it must have one free parameter,
        i.e. the amplitude that will be scanned, and a drive channel which matches the qubit.

    # section: tutorial
        :doc:`/tutorials/calibrating_armonk`

        See also `Qiskit Textbook <https://qiskit.org/textbook/ch-quantum-hardware/\
        calibrating-qubits-pulse.html>`_
        for the pulse level programming of Rabi experiment.

    """

    __analysis_class__ = OscillationAnalysis
    __rabi_gate_name__ = "Rabi"
    __updater__ = Amplitude

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default option values for the experiment :meth:`run` method."""
        options = super()._default_run_options()

        options.meas_level = MeasLevel.KERNELED
        options.meas_return = "single"

        return options

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default values for the pulse if no schedule is given.

        Users can set a schedule by doing

        .. code-block::

            rabi.set_experiment_options(schedule=rabi_schedule)

        Experiment Options:
            duration (int): The duration of the default Gaussian pulse.
            sigma (float): The standard deviation of the default Gaussian pulse.
            amplitudes (iterable): The list of amplitude values to scan.
            schedule (ScheduleBlock): The schedule for the Rabi pulse that overrides the default.
        """
        options = super()._default_experiment_options()

        options.duration = 160
        options.sigma = 40
        options.amplitudes = np.linspace(-0.95, 0.95, 51)
        options.schedule = None
        return options

    @classmethod
    def _default_calibration_options(cls) -> Options:
        """Default calibration options for the experiment.

        Calibration Options:
            cal_parameter_name (str): The name of the amplitude parameter in the schedule stored in
                the calibrations instance. The default value is "amp".
            angles_schedules (List): A list of tuples that is given to the :class:`Amplitude`
                updater. By default this is set to update the x and square-root X pulse, i.e. the
                default value is :code:`[(np.pi, "amp", "x"), (np.pi / 2, "amp", "sx")]`.
        """
        options = super()._default_calibration_options()
        options.cal_parameter_name = "amp"
        options.angles_schedules = [(np.pi, "amp", "x"), (np.pi / 2, "amp", "sx")]
        options.schedule_name = "x"
        return options

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()
        options.result_parameters = [ParameterRepr("freq", "rabi_rate")]
        options.normalization = True

        return options

    def __init__(
        self,
        qubit: int,
        cals: Optional[Calibrations] = None,
        schedule_name: Optional[str] = "x",
        cal_parameter_name: Optional[str] = "amp",
        amplitudes: Optional[List] = None,
        angles_schedules: Optional[List[Tuple]] = None,
    ):
        """Initialize a Rabi experiment on the given qubit.

        The parameters of the Gaussian Rabi pulse can be specified at run-time.
        The rabi pulse has the following parameters:
        - duration: The duration of the rabi pulse in samples, the default is 160 samples.
        - sigma: The standard deviation of the pulse, the default is duration 40.
        - amplitudes: The amplitude that are scanned in the experiment, default  is
        np.linspace(-0.95, 0.95, 51)

        Args:
            qubit: The qubit on which to run the Rabi experiment.
            cals: If calibrations is given then running the experiment may update the
                values of the pulse parameters stored in calibrations.
            schedule_name: The name of the schedule to extract from the calibrations. This value
                defaults to "x".
            cal_parameter_name: The name of the parameter in calibrations to update. This name will
                be stored in the experiment options and defaults to "amp".
            amplitudes: The values of the amplitudes to scan. Specify this argument to override the
                default values of the experiment.
            angles_schedules: A list of tuples that is given to the :class:`Amplitude`
                updater. See the experiment options for default values.

        Raises:
            CalibrationError: If the schedule_name or calibration parameter name are not contained
                in the list of angles to update.
        """
        super().__init__([qubit])
        self.calibration_options.calibrations = cals
        self.calibration_options.cal_parameter_name = cal_parameter_name
        self.calibration_options.schedule_name = schedule_name

        if angles_schedules is not None:
            self.calibration_options.angles_schedules = angles_schedules

        if amplitudes is not None:
            self.experiment_options.amplitudes = amplitudes

    # pylint: disable=arguments-differ
    def get_schedule_from_defaults(self, backend: Optional[Backend] = None) -> ScheduleBlock:
        """Get the schedules from the default options."""
        with pulse.build(backend=backend, name="rabi") as default_schedule:
            pulse.play(
                pulse.Gaussian(
                    duration=self.experiment_options.duration,
                    amp=Parameter("amp"),
                    sigma=self.experiment_options.sigma,
                ),
                pulse.DriveChannel(self.physical_qubits[0]),
            )

        return default_schedule

    def validate_schedule(self, schedule: ScheduleBlock):
        """Validate the Rabi schedule.

        Raises:
            CalibrationError: If the name of the schedule and its parameter are not
                in the angles and schedules tuple to update (see :class:`Amplitude`).
        """
        self._validate_channels(schedule)
        self._validate_parameters(schedule, 1)

        # consistency check between the schedule and the amplitudes to update.
        if self.calibration_options.calibrations is not None:
            param = self.calibration_options.cal_parameter_name
            for update_tuple in self.calibration_options.angles_schedules:
                if update_tuple[1] == param and update_tuple[2] == schedule.name:
                    break
            else:
                raise CalibrationError(
                    f"The schedule {schedule.name} is not in the angles to update."
                )

    def _template_circuit(self, amp_param) -> QuantumCircuit:
        """Return the template quantum circuit."""
        circuit = QuantumCircuit(1)
        circuit.append(Gate(name=self.__rabi_gate_name__, num_qubits=1, params=[amp_param]), (0,))
        circuit.measure_active()

        return circuit

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        """Create the circuits for the Rabi experiment.

        Args:
            backend: A backend object.

        Returns:
            A list of circuits with a rabi gate with an attached schedule. Each schedule
            will have a different value of the scanned amplitude.

        Raises:
            CalibrationError:
                - If the user-provided schedule does not contain a channel with an index
                  that matches the qubit on which to run the Rabi experiment.
                - If the user provided schedule has more than one free parameter.
        """
        schedule = self.get_schedule(
            assign_params={self.calibration_options.cal_parameter_name: Parameter("amp")},
        )

        param = next(iter(schedule.parameters))

        # Create template circuit
        circuit = self._template_circuit(param)
        circuit.add_calibration(
            self.__rabi_gate_name__, self.physical_qubits, schedule, params=[param]
        )

        # Create the circuits to run
        circs = []
        for amp in self.experiment_options.amplitudes:
            amp = np.round(amp, decimals=6)
            assigned_circ = circuit.assign_parameters({param: amp}, inplace=False)
            assigned_circ.metadata = self.circuit_metadata(xval=amp)

            circs.append(assigned_circ)

        return circs

    def update_calibrations(self, experiment_data: ExperimentData):
        """Update the calibrations given the experiment data.

        Args:
            experiment_data: The experiment data to use for the update.
        """
        calibrations = self.calibration_options.calibrations
        angles_schedules = self.calibration_options.angles_schedules

        self.__updater__.update(calibrations, experiment_data, angles_schedules=angles_schedules)


class EFRabi(Rabi):
    """An experiment that scans the amplitude of a pulse to calibrate rotations between 1 and 2.

    # section: overview

        This experiment is a subclass of the :class:`Rabi` experiment but takes place between
        the first and second excited state. An initial X gate used to populate the first excited
        state. The Rabi pulse is then applied on the 1 <-> 2 transition (sometimes also labeled
        the e <-> f transition) which implies that frequency shift instructions are used. The
        necessary frequency shift (typically the qubit anharmonicity) should be specified
        through the experiment options.

        The circuits are of the form:

        .. parsed-literal::

                       ┌───┐┌───────────┐ ░ ┌─┐
                  q_0: ┤ X ├┤ Rabi(amp) ├─░─┤M├
                       └───┘└───────────┘ ░ └╥┘
            measure: 1/══════════════════════╩═
                                             0

    # section: example
        Users can set a schedule by doing

        .. code-block::

            ef_rabi.set_experiment_options(schedule=rabi_schedule)

    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default values for the pulse if no schedule is given.

        Experiment Options:

            frequency_shift (float): The frequency by which the 1 to 2 transition is
                detuned from the 0 to 1 transition.
        """
        options = super()._default_experiment_options()
        options.frequency_shift = None

        return options

    @classmethod
    def _default_analysis_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_analysis_options()
        options.result_parameters = [ParameterRepr("freq", "rabi_rate_12")]

        return options

    def get_schedules_from_defaults(self, backend: Optional[Backend] = None) -> ScheduleBlock:
        """Create the default schedule for the EFRabi gate with a frequency shift to the 1-2
        transition."""

        if self.experiment_options.frequency_shift is None:
            try:
                anharm, _ = backend.properties().qubit_property(self.physical_qubits[0])[
                    "anharmonicity"
                ]
                self.set_experiment_options(frequency_shift=anharm)
            except KeyError as key_err:
                raise CalibrationError(
                    f"The backend {backend} does not provide an anharmonicity for qubit "
                    f"{self.physical_qubits[0]}. Use EFRabi.set_experiment_options(frequency_shift="
                    f"anharmonicity) to manually set the correct frequency for the 1-2 transition."
                ) from key_err
            except AttributeError as att_err:
                raise CalibrationError(
                    "When creating the default schedule without passing a backend, the frequency needs "
                    "to be set manually through EFRabi.set_experiment_options(frequency_shift=..)."
                ) from att_err

        with pulse.build(backend=backend, name=self.__rabi_gate_name__) as default_schedule:
            with pulse.frequency_offset(
                self.experiment_options.frequency_shift,
                pulse.DriveChannel(self.physical_qubits[0]),
            ):
                pulse.play(
                    pulse.Gaussian(
                        duration=self.experiment_options.duration,
                        amp=Parameter("amp"),
                        sigma=self.experiment_options.sigma,
                    ),
                    pulse.DriveChannel(self.physical_qubits[0]),
                )

        return default_schedule

    def _template_circuit(self, amp_param) -> QuantumCircuit:
        """Return the template quantum circuit."""
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.append(Gate(name=self.__rabi_gate_name__, num_qubits=1, params=[amp_param]), (0,))
        circuit.measure_active()

        return circuit
