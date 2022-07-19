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
import copy
import logging
from typing import List, Optional, Type, Union
import warnings

from qiskit import QuantumCircuit, transpile
from qiskit.providers.backend import Backend
from qiskit.pulse import ScheduleBlock

from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework.base_analysis import BaseAnalysis
from qiskit_experiments.framework.base_experiment import BaseExperiment
from qiskit_experiments.framework.experiment_data import ExperimentData
from qiskit_experiments.exceptions import CalibrationError

LOG = logging.getLogger(__name__)


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

    @classmethod
    def _default_experiment_options(cls):
        """Default values for a calibration experiment.

        Experiment Options:
            result_index (int): The index of the result from which to update the calibrations.
            group (str): The calibration group to which the parameter belongs. This will default
                to the value "default".

        """
        options = super()._default_experiment_options()

        options.result_index = -1
        options.group = "default"

        return options

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

    def _validate_channels(self, schedule: ScheduleBlock, physical_qubits: List[int]):
        """Check that the physical qubits are contained in the schedule.

        This is a helper method that experiment developers can call in their implementation
        of :meth:`validate_schedules` when checking the schedules.

        Args:
            schedule: The schedule for which to check the qubits.
            physical_qubits: The qubits that should be included in the schedule.

        Raises:
            CalibrationError: If a physical qubit is not contained in the channels schedule.
        """
        for qubit in physical_qubits:
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

    def _metadata(self):
        """Add standard calibration metadata."""
        metadata = super()._metadata()

        metadata["cal_group"] = self.experiment_options.group
        metadata["cal_param_name"] = self._param_name
        metadata["cal_schedule"] = self._sched_name

        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        """Override the transpiled circuits method to bring in the inst_map.

        The calibrated schedules are transpiled into the circuits using the instruction
        schedule map. Since instances of :class:`InstructionScheduleMap` are not serializable
        they should not be in the transpile options. Only instances of :class:`Calibrations`
        need to be serialized. Here, we pass the instruction schedule map of the calibrations
        to the transpiler.

        Returns:
            A list of transpiled circuits.
        """
        transpile_opts = copy.copy(self.transpile_options.__dict__)
        if "inst_map" in transpile_opts:
            LOG.warning(
                "Instruction schedule maps should not be present in calibration "
                "experiments. Overriding with the inst. map of the calibrations."
            )

        transpile_opts["inst_map"] = self.calibrations.default_inst_map
        transpile_opts["initial_layout"] = list(self.physical_qubits)

        return transpile(self.circuits(), self.backend, **transpile_opts)

    def run(
        self,
        backend: Optional[Backend] = None,
        analysis: Optional[Union[BaseAnalysis, None]] = "default",
        timeout: Optional[float] = None,
        **run_options,
    ) -> ExperimentData:
        """Run an experiment, perform analysis, and update any calibrations.

        Args:
            backend: Optional, the backend to run the experiment on. This
                     will override any currently set backends for the single
                     execution.
            analysis: Optional, a custom analysis instance to use for performing
                      analysis. If None analysis will not be run. If ``"default"``
                      the experiments :meth:`analysis` instance will be used if
                      it contains one.
            timeout: Time to wait for experiment jobs to finish running before
                     cancelling.
            run_options: backend runtime options used for circuit execution.

        Returns:
            The experiment data object.
        """
        experiment_data = super().run(
            backend=backend, analysis=analysis, timeout=timeout, **run_options
        )

        if self.auto_update and analysis:
            experiment_data.add_analysis_callback(self.update_calibrations)

        return experiment_data
