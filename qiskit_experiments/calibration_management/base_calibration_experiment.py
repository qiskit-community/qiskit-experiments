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
from typing import Any, Dict, Iterable, Optional, Tuple

from qiskit.providers.options import Options
from qiskit.providers.backend import Backend
from qiskit.circuit import Parameter
from qiskit.pulse import ScheduleBlock

from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.framework.base_experiment import BaseExperiment
from qiskit_experiments.framework.experiment_data import ExperimentData
from qiskit_experiments.exceptions import CalibrationError


class BaseCalibrationExperiment(BaseExperiment, ABC):
    """An abstract base class for calibration experiments.

    This abstract base class specifies an experiment and how to update an
    optional instance of :class:`Calibrations` specified in the calibration options
    as `calibrations`. Furthermore, the calibration options also specify
    an auto_update variable which, by default, is set to True. If this variable,
    is True then the run method of the experiment will call :meth:`block_for_results`
    and update the calibrations instance once the backend has returned the data.

    Developers that wish to create a calibration experiment must subclass this base
    class. If the experiment uses custom schedules, which is typically the case, then
    developers may chose to use the :meth:`get_schedules` method when creating the
    circuits for the experiment. If :meth:`get_schedules` is used then the developer
    must override at least one of the following methods used by :meth:`get_schedules`
    to set the schedules:

    #. :meth:`get_schedules_from_options`

    #. :meth:`get_schedules_from_calibrations`

    #. :meth:`get_schedules_from_defaults`

    These methods are called by :meth:`get_schedules`. Furthermore, developers must implement
    the :meth:`update_calibrations` which is responsible for updating the values of the
    parameters stored in an instance of :class:`Calibrations`. This may require the developer
    to set the class variable :code:`__updater__` if they wish to use the update classes
    implemented in :mod:`qiskit_experiments.calibration_management.update_library`. In addition
    to these calibration specific requirements, the developer must set the analysis method with
    the class variable :code:`__analysis_class__` and any default experiment options.
    """

    # The updater class that updates the Calibrations instance
    __updater__ = None

    def __init__(
        self,
        qubits: Iterable[int],
        calibrations: Calibrations,
        experiment_type: Optional[str] = None,
    ):
        """Initialize the experiment object.

        Args:
            qubits: the number of qubits or list of physical qubits for
                    the experiment.
            calibrations: The calibrations instance with which to initialize the experiment.
            experiment_type: Optional, the experiment type string.
        """
        super().__init__(qubits, experiment_type)

        self._calibration_options = self._default_calibration_options()
        self._cals = calibrations

    @property
    def calibrations(self) -> Calibrations:
        """Calibration management object that holds the schedule."""
        return self._cals

    def update_calibrations(self, experiment_data: ExperimentData):
        """Update parameter values in the :class:`Calibrations` instance.

        The default behaviour is to call the update method of the class variable
        :code:`__updater__` with simplistic options. Subclasses can override this
        method to update the instance of :class:`Calibrations` if they require a
        more sophisticated behaviour as is the case for the :class:`Rabi` and
        :class:`FineAmplitude` calibration experiments.
        """
        self.__updater__.update(
            self._cals,
            experiment_data,
            parameter=self.calibration_options.cal_parameter_name,
            schedule=self.calibration_options.schedule_name,
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
                gven this will default to :code:`schedule_name` in the calibration options.
            assign_params: A dict to specify parameters in the schedule that are
                to be mapped to an unassigned parameter.

        Returns:
            A schedule for the corresponding arguments if there exists an instance
            :code:`self.calibration_options.calibrations`.
        """

        if sched_name is None:
            sched_name = self.calibration_options.schedule_name

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
                :class:`Calibrations` stored under the calibration options. If this is None then
                :meth:`get_schedule_from_calibrations` will default to the :code:`schedule_name`
                in the calibration options.
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

    def _circuit_metadata(self, xval: Any, **kwargs) -> Dict[str, Any]:
        """Return the circuit metadata for the calibration experiment."""
        metadata = {"experiment_type": self._type, "qubits": self.physical_qubits, "xval": xval}
        metadata.update(kwargs)
        return metadata

    @classmethod
    def _default_calibration_options(cls) -> Options:
        """Default calibration options for the experiment.

        Calibration Options:
            calibrations (Calibrations): An optional instance of :class:`Calibrations` if this
                instance is specified then the experiment will try and update the calibrations.
            auto_update (bool): A boolean which defaults to True. If this variable is set to
                True then running the calibration experiment will block for the results and
                update the calibrations if the calibrations is not None.
            schedule_name (str): The name of the schedule to retrieve from the calibrations.
            cal_parameter_name (str): The name of the parameter to update in the calibrations.
        """
        return Options(auto_update=True, schedule_name=None, cal_parameter_name=None)

    @property
    def calibration_options(self) -> Options:
        """Return the calibration options for the experiment."""
        return self._calibration_options

    def set_calibration_options(self, **fields):
        """Set the calibration options.

        Args:
            fields: The fields to update the options

        Raises:
            AttributeError: If the field passed in is not a supported options
        """
        for field in fields:
            if not hasattr(self._calibration_options, field):
                raise AttributeError(
                    f"Options field {field} is not valid for {type(self).__name__}"
                )
        self._calibration_options.update_options(**fields)

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

        if self.calibration_options.auto_update:
            if self._cals is not None:
                experiment_data = experiment_data.block_for_results()
                self.update_calibrations(experiment_data)

        return experiment_data
