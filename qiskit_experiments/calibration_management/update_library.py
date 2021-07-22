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

"""A library of experiment calibrations."""

from abc import ABC
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Union
import numpy as np

from qiskit.circuit import Parameter
from qiskit.pulse import ScheduleBlock

from qiskit_experiments.framework.experiment_data import ExperimentData
from qiskit_experiments.calibration_management.backend_calibrations import BackendCalibrations
from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.calibration_management.parameter_value import ParameterValue
from qiskit_experiments.calibration_management.calibration_key_types import ParameterValueType
from qiskit_experiments.exceptions import CalibrationError


class BaseUpdater(ABC):
    """A base class to update calibrations."""

    __fit_parameter__ = None

    def __init__(self):
        """Updaters are not meant to be instantiated.

        Instead of instantiating updaters use them by calling the :meth:`update` class method.
        For example, the :class:`Frequency` updater is called in the following way

         .. code-block:: python

            Frequency.update(calibrations, spectroscopy_data)

        Here, calibrations is an instance of :class:`BackendCalibrations` and spectroscopy_data
        is the result of a :class:`QubitSpectroscopy` experiment.
        """
        raise CalibrationError(
            "Calibration updaters are not meant to be instantiated. The intended usage"
            "is Updater.update(calibrations, exp_data, ...)."
        )

    @staticmethod
    def _time_stamp(exp_data: ExperimentData) -> datetime:
        """Helper method to extract the datetime."""
        all_times = exp_data.completion_times.values()
        if all_times:
            return max(all_times)

        return datetime.now(timezone.utc)

    @classmethod
    def _add_parameter_value(
        cls,
        cal: Calibrations,
        exp_data: ExperimentData,
        value: ParameterValueType,
        param: Union[Parameter, str],
        schedule: Union[ScheduleBlock, str] = None,
        group: str = "default",
    ):
        """Update the calibrations with the given value.

        Args:
            cal: The Calibrations instance to update.
            exp_data: The ExperimentData instance that contains the result and the experiment data.
            value: The value extracted by the subclasses in the :meth:`update` method.
            param: The name of the parameter, or the parameter instance, which will receive an
                updated value.
            schedule: The ScheduleBlock instance or the name of the instance to which the parameter
                is attached.
            group: The calibrations group to update.
        """

        qubits = exp_data.data(0)["metadata"]["qubits"]

        param_value = ParameterValue(
            value=value,
            date_time=cls._time_stamp(exp_data),
            group=group,
            exp_id=exp_data.experiment_id,
        )

        cal.add_parameter_value(param_value, param, qubits, schedule)

    @classmethod
    def update(
        cls,
        calibrations: Calibrations,
        exp_data: ExperimentData,
        parameter: str,
        schedule: Optional[Union[ScheduleBlock, str]],
        result_index: int = -1,
        group: str = "default",
    ):
        """Update the calibrations based on the data.

        Args:
            calibrations: The calibrations to update.
            exp_data: The experiment data from which to update.
            parameter: The name of the parameter in the calibrations to update.
            schedule: The ScheduleBlock instance or the name of the instance to which the parameter
                is attached.
            result_index: The result index to use, defaults to -1.
            group: The calibrations group to update. Defaults to "default."

        Raises:
            CalibrationError: If the analysis result does not contain a frequency variable.
        """

        result = exp_data.analysis_results(result_index).data()

        if cls.__fit_parameter__ not in result["popt_keys"]:
            raise CalibrationError(
                f"{cls.__name__} updates from analysis classes "
                f"which report {cls.__fit_parameter__} in popt."
            )

        param = parameter
        value = result["popt"][result["popt_keys"].index(cls.__fit_parameter__)]

        cls._add_parameter_value(
            calibrations, exp_data, value, param, schedule=schedule, group=group
        )


class Frequency(BaseUpdater):
    """Update frequencies."""

    __fit_parameter__ = "freq"

    # pylint: disable=arguments-differ,unused-argument
    @classmethod
    def update(
        cls,
        calibrations: BackendCalibrations,
        exp_data: ExperimentData,
        parameter: str = BackendCalibrations.__qubit_freq_parameter__,
        result_index: int = -1,
        group: str = "default",
        **options,
    ):
        """Update a qubit frequency from, e.g., QubitSpectroscopy."""
        super().update(
            calibrations, exp_data, parameter, schedule=None, result_index=result_index, group=group
        )


class Drag(BaseUpdater):
    """Update drag parameters."""

    __fit_parameter__ = "beta"


class Amplitude(BaseUpdater):
    """Update pulse amplitudes."""

    # pylint: disable=arguments-differ,unused-argument
    @classmethod
    def update(
        cls,
        calibrations: Calibrations,
        exp_data: ExperimentData,
        result_index: int = -1,
        group: str = "default",
        angles_schedules: List[Tuple[float, str, Union[str, ScheduleBlock]]] = None,
        **options,
    ):
        """Update the amplitude of pulses.

        The value of the amplitude must be derived from the fit so the base method cannot be used.

        Args:
            calibrations: The calibrations to update.
            exp_data: The experiment data from which to update.
            result_index: The result index to use, defaults to -1.
            group: The calibrations group to update. Defaults to "default."
            angles_schedules: A list of tuples specifying which angle to update for which
                pulse schedule. Each tuple is of the form: (angle, parameter_name,
                schedule). Here, angle is the rotation angle for which to extract the amplitude,
                parameter_name is the name of the parameter whose value is to be updated, and
                schedule is the schedule or its name that contains the parameter.
            options: Trailing options.

        Raises:
            CalibrationError: If the experiment is not of the supported type.
        """
        from qiskit_experiments.library.calibration.rabi import Rabi
        from qiskit_experiments.library.calibration.fine_amplitude import FineAmplitude

        if angles_schedules is None:
            angles_schedules = [(np.pi, "amp", "xp")]

        result = exp_data.analysis_results(result_index).data()

        if isinstance(exp_data.experiment, Rabi):
            freq = result["popt"][result["popt_keys"].index("freq")]
            rate = 2 * np.pi * freq

            for angle, param, schedule in angles_schedules:
                value = np.round(angle / rate, decimals=8)

                cls._add_parameter_value(calibrations, exp_data, value, param, schedule, group)

        elif isinstance(exp_data.experiment, FineAmplitude):
            d_theta = result["popt"][result["popt_keys"].index("d_theta")]

            for target_angle, param, schedule in angles_schedules:

                qubits = exp_data.data(0)["metadata"]["qubits"]

                prev_amp = calibrations.get_parameter_value(param, qubits, schedule, group=group)
                scale = target_angle / (target_angle + d_theta)
                new_amp = prev_amp * scale

                cls._add_parameter_value(calibrations, exp_data, new_amp, param, schedule, group)

        else:
            raise CalibrationError(f"{cls.__name__} updates from {type(Rabi.__name__)}.")
