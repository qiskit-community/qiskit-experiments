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

"""Base class for calibration-type experiments."""

from abc import ABC
from typing import Dict, Optional, Tuple, Type
import warnings

from qiskit.providers.backend import Backend
from qiskit.circuit import Parameter
from qiskit.pulse import ScheduleBlock

from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework.base_experiment import BaseExperiment
from qiskit_experiments.framework.experiment_data import ExperimentData
from qiskit_experiments.exceptions import CalibrationError


class BaseCalibrationExperiment(BaseExperiment, ABC):
    """A mixin class to create calibration experiments.

    This abstract class extends a characterization experiment by turning it into a
    calibration experiment. Such experiments allow schedule management and updating of an
    instance of :class:`Calibrations`. Furthermore, calibration experiments also specify
    an auto_update variable which, by default, is set to True. If this variable,
    is True then the run method of the experiment will call :meth:`block_for_results`
    and update the calibrations instance once the backend has returned the data.

    This mixin class inherits from the :class:`BaseExperiment` class since calibration
    experiments by default call :meth:`block_for_results`. This ensures that the next
    calibration experiment cannot proceed before the calibration parameters have been
    updated. Developers that wish to create a calibration experiment must subclass this
    base class and the characterization experiment. Therefore, developers that use this
    mixin class must pay special attention to their class definition. Indeed, the first
    class should be this mixin and the second class should be the characterization
    experiment since the run method from the mixin must be used. For example, the rough
    frequency calibration experiment is defined as

    .. code-block:: python

        RoughFrequency(BaseCalibrationExperiment, QubitSpectroscopy)

    This ensures that the :meth:`run` method of :class:`RoughFrequency` will be the
    run method of the :class:`BaseCalibrationExperiment` class. Furthermore, developers
    must explicitly call the :meth:`__init__` methods of both parent classes.

    Developers should strive to follow the convention that the first two arguments of
    a calibration experiment are the qubit(s) and the :class:`Calibration` instance.

    If the experiment uses custom schedules, which is typically the case, then
    developers may chose to use the :meth:`get_schedules` method when creating the
    circuits for the experiment. If :meth:`get_schedules` is used then the developer
    must override at least one of the following methods used by :meth:`get_schedules`
    to set the schedules:

    #. :meth:`_get_schedules_from_options`

    #. :meth:`_get_schedules_from_calibrations`

    #. :meth:`_get_schedules_from_defaults`

    These methods are called by :meth:`get_schedules`.

    The :meth:`update_calibrations` method is responsible for updating the values of the parameters
    stored in the instance of :class:`Calibrations`. Here, :class:`BaseCalibrationExperiment`
    provides a default update methodology that subclasses can override if a more elaborate behaviour
    is needed. At the minimum the developer must set the variable :code:`_updater` which
    should have an :code:`update` method and can be chosen from the library
    :mod:`qiskit_experiments.calibration_management.update_library`. See also
    :class:`qiskit_experiments.calibration_management.update_library.BaseUpdater`. If no updater
    is specified the experiment will still run but no update of the calibrations will be performed.

    In addition to the calibration specific requirements, the developer must set the analysis method
    with the class variable :code:`__analysis_class__` and any default experiment options.
    """

    def __init_subclass__(cls, **kwargs):
        """Warn if BaseCalibrationExperiment is not the first parent."""
        for mro_cls in cls.mro():
            if mro_cls is BaseCalibrationExperiment:
                break
            if issubclass(mro_cls, BaseExperiment) and not issubclass(
                mro_cls, BaseCalibrationExperiment
            ):
                warnings.warn(
                    "Calibration experiments must inherit from BaseCalibrationExperiment "
                    f"before a BaseExperiment subclass: {cls}->{mro_cls}."
                )
                break
        super().__init_subclass__(**kwargs)

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        calibrations: Calibrations,
        *args,
        schedule_name: Optional[str] = None,
        cal_parameter_name: Optional[str] = None,
        updater: Optional[Type[BaseUpdater]] = None,
        auto_update: bool = True,
        **kwargs,
    ):
        """Setup the calibration experiment object.

        Args:
            calibrations: The calibrations instance with which to initialize the experiment.
            args: Arguments for the characterization class.
            schedule_name: An optional string which specifies the name of the schedule in
                the calibrations that will be updated.
            cal_parameter_name: An optional string which specifies the name of the parameter in
                the calibrations that will be updated. If None is given then no parameter will
                be updated. Subclasses may assign default values in their init.
            updater: The updater class that updates the Calibrations instance. Different
                calibration experiments will use different updaters.
            auto_update: If set to True (the default) then the calibrations will automatically be
                updated once the experiment has run and :meth:`block_for_results()` will be called.
            kwargs: Key word arguments for the characterization class.
        """
        super().__init__(*args, **kwargs)
        self._cals = calibrations
        self._sched_name = schedule_name
        self._param_name = cal_parameter_name
        self._updater = updater
        self.auto_update = auto_update

    @property
    def calibrations(self) -> Calibrations:
        """Return the calibrations."""
        return self._cals

    def update_calibrations(self, experiment_data: ExperimentData):
        """Update parameter values in the :class:`Calibrations` instance.

        The default behaviour is to call the update method of the class variable
        :code:`__updater__` with simplistic options. Subclasses can override this
        method to update the instance of :class:`Calibrations` if they require a
        more sophisticated behaviour as is the case for the :class:`Rabi` and
        :class:`FineAmplitude` calibration experiments.
        """
        if self._updater is not None:
            self._updater.update(
                self._cals,
                experiment_data,
                parameter=self._param_name,
                schedule=self._sched_name,
            )

    def _get_schedule_from_options(self, option_name: str) -> ScheduleBlock:
        """Get a schedule from the experiment options.

        Developers can subclass this method if they need a more sophisticated
        methodology to get schedules from their experiment options.

        Args:
            option_name: The name of the option under which the schedule is stored.

        Returns:
            The schedule to use in the calibration experiment.
        """
        return self.experiment_options.get(option_name, None)

    def _get_schedule_from_calibrations(
        self,
        qubits: Optional[Tuple[int, ...]] = None,
        sched_name: Optional[str] = None,
        assign_params: Optional[Dict[str, Parameter]] = None,
    ) -> Optional[ScheduleBlock]:
        """Get the schedules from the Calibrations instance.

        This method is called if :meth:`get_schedules_from_options` did not return
        any schedules to use. Here, we get a schedule from an instance of
        :class:`Calibrations` using the :meth:`get_schedule` method. Subclasses can override
        this method if they need a different behaviour.

        Args:
            qubits: The qubits for which to fetch the schedules. If None is given this will
                default to the physical qubits of the experiment.
            sched_name: The name of the schedule to fetch from the calibrations. If None is
                gven this will default to :code:`self._sched_name`.
            assign_params: A dict to specify parameters in the schedule that are
                to be mapped to an unassigned parameter.

        Returns:
            A schedule for the corresponding arguments if there exists an instance
            :code:`self._cals`.
        """

        if sched_name is None:
            sched_name = self._sched_name

        if qubits is None:
            qubits = self.physical_qubits

        if self._cals is not None:
            return self._cals.get_schedule(sched_name, qubits=qubits, assign_params=assign_params)

        return None

    def _get_schedule_from_defaults(self, **kwargs) -> Optional[ScheduleBlock]:
        """Get the schedules based on default experiment options.

        Subclasses can override this method to define and get default schedules based on
        default experiment options such as the number of samples in a Gaussian and its
        amplitude. This function is called as a last resort in :meth:`get_schedules`
        and accommodates cases when the user provides neither calibrations nor schedules.
        For example, if the default schedule is a Gaussian then this function may return
        the schedule

        .. code-block:: python

            with pulse.build(backend=backend, name="rabi") as default_schedule:
                pulse.play(
                    pulse.Gaussian(
                        duration=self.experiment_options.duration,
                        amp=Parameter("amp"),
                        sigma=self.experiment_options.sigma,
                    ),
                    pulse.DriveChannel(self.physical_qubits[0]),
            )

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} could not find a schedule in the experiment options "
            "or the calibrations and no default schedule method was implemented."
        )

    def _validate_schedule(self, schedule: ScheduleBlock):
        """Subclass can implement this method to validate the schedule they use.

        Validating schedules may include checks on the number of parameters and
        the channels in the schedule. The functions :meth:`_validate_channels` and
        :meth:`_validate_parameters` implement such standard checks for reuse.
        """

    def _validate_channels(self, schedule: ScheduleBlock):
        """Check that the physical qubits are contained in the schedule.

        This is a helper method that experiment developers can call in their implementation
        of :meth:`validate_schedules` when checking the schedules.

        Args:
            schedule: The schedule for which to check the qubits.

        Raises:
            CalibrationError: If a physical qubit is not contained in the channels schedule.
        """
        for qubit in self.physical_qubits:
            if qubit not in set(ch.index for ch in schedule.channels):
                raise CalibrationError(
                    f"Schedule {schedule.name} does not contain a channel "
                    f"for the physical qubit {qubit}."
                )

    def _validate_parameters(self, schedule: ScheduleBlock, n_expected_parameters: int):
        """Check that the schedule has the expected number of parameters.

        This is a helper method that experiment developers can call in their implementation
        of :meth:`validate_schedules` when checking the schedules.

        Args:
            schedule: The schedule for which to check the qubits.
            n_expected_parameters: The number of free parameters the schedule must have.

        Raises:
            CalibrationError: If the schedule does not have n_expected_parameters parameters.
        """
        if len(schedule.parameters) != n_expected_parameters:
            raise CalibrationError(
                f"The schedules {schedule.name} for {self.__class__.__name__} must have "
                f"{n_expected_parameters} parameters. Found {len(schedule.parameters)}."
            )

    def get_schedule(
        self,
        qubits: Optional[Tuple[int, ...]] = None,
        sched_name: Optional[str] = None,
        option_name: str = "schedule",
        assign_params: Optional[Dict[str, Parameter]] = None,
        **kwargs,
    ) -> ScheduleBlock:
        """Get the schedules for the circuits.

        This method defines the order in which the schedules are consumed. This order is

        #. Use the schedules directly available in the experiment, i.e. those specified
           by experiment users. This is made possible in experiments by implementing the
           :meth:`get_schedules_from_options` method.

        #. Use the schedules found in the instance of :class:`Calibrations` attached to the
           experiment. This is done by implementing the :meth:`get_schedules_from_calibrations`
           method.

        #. Use any default schedules specified by the :meth:`get_schedules_from_defaults`.

        If any one step does not return a schedule then we attempt to get schedules from the next
        step. If none of these three steps have returned any schedules then an error is raised.

        Args:
            qubits: The qubits for which to get the schedule in the calibrations. If None is given
                this will default to the physical qubits of the experiment.
            sched_name: The name of the schedule to retrieve from the instance of
                :class:`Calibrations` stored as a protected variable. If this is None then
                :meth:`get_schedule_from_calibrations` will default to the :code:`self._sched_name`.
            option_name: The name of the option under which to get the schedule from the experiment
                options. This will default to "schedule" if None is given.
            assign_params: A dict that :meth:`get_schedule_from_calibrations` can use to leave
                certain parameters in the schedule unassigned. The key is the name of the parameter
                and the value should be an instance of :class:`ParameterExpression`.
            kwargs: Additional keyword arguments that can be used by implementations of
                :meth:`get_schedule_from_defaults`.

        Returns:
            schedules: The schedules (possibly with one or more free parameters) as either a
                ScheduleBlock or a list of ScheduleBlocks depending on the experiment.

        Raises:
            CalibrationError: if none of the methods above returned schedules.
        """
        schedules = self._get_schedule_from_options(option_name)

        if schedules is None:
            schedules = self._get_schedule_from_calibrations(qubits, sched_name, assign_params)

        if schedules is None:
            schedules = self._get_schedule_from_defaults(**kwargs)

        self._validate_schedule(schedules)

        return schedules

    def run(
        self,
        backend: Backend,
        analysis: bool = True,
        experiment_data: Optional[ExperimentData] = None,
        **run_options,
    ) -> ExperimentData:
        """Run an experiment, perform analysis, and update any calibrations.

        Args:
            backend: The backend to run the experiment on.
            analysis: If True run analysis on the experiment data.
            experiment_data: Optional, add results to existing experiment data.
                If None a new ExperimentData object will be returned.
            run_options: backend runtime options used for circuit execution.

        Returns:
            The experiment data object.
        """
        experiment_data = super().run(backend, analysis, experiment_data, **run_options)

        if self.auto_update and analysis:
            experiment_data.add_analysis_callback(self.update_calibrations)

        return experiment_data
