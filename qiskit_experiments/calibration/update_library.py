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

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Tuple, Union
import numpy as np

from qiskit.circuit import Parameter
from qiskit.pulse import ScheduleBlock

from qiskit_experiments.experiment_data import ExperimentData
from qiskit_experiments.calibration.backend_calibrations import BackendCalibrations
from qiskit_experiments.calibration.calibrations import Calibrations
from qiskit_experiments.calibration.parameter_value import ParameterValue
from qiskit_experiments.calibration.exceptions import CalibrationError
from qiskit_experiments.calibration.calibration_key_types import ParameterValueType


class BaseUpdater(ABC):
    """A base class to update calibrations."""

    def __init__(self):
        """Updaters are not meant to be instantiated."""
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

        return datetime.now()

    @classmethod
    def _add_parameter_value(
        cls,
        exp_data: ExperimentData,
        cal: Calibrations,
        value: ParameterValueType,
        param: Union[Parameter, str],
        schedule: Union[ScheduleBlock, str] = None,
        group: str = "default",
    ):
        """Update the calibrations with the given value.

        Args:
            exp_data: The ExperimentData instance that contains the result and the experiment data.
            cal: The Calibrations instance to update.
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
    @abstractmethod
    def update(cls, calibrations: BackendCalibrations, exp_data: ExperimentData, **options):
        """Update the calibrations based on the data.

        Child update classes must implement this function. This function defines how the data
        is extracted from an experiment and then used to update the values of one or more
        parameters in the calibrations.
        """


class Frequency(BaseUpdater):
    """Update frequencies."""

    # pylint: disable=arguments-differ
    @classmethod
    def update(
        cls,
        calibrations: BackendCalibrations,
        exp_data: ExperimentData,
        result_index: int = -1,
        group: str = "default",
        parameter: str = BackendCalibrations.__qubit_freq_parameter__,
    ):
        """Update a qubit frequency from, e.g., QubitSpectroscopy.

        Args:
            calibrations: The calibrations to update.
            exp_data: The experiment data from which to update.
            result_index: The result index to use, defaults to -1.
            group: The calibrations group to update. Defaults to "default."
            parameter: The name of the parameter to update. If it is not specified
                this will default to the qubit frequency.

        Raises:
            CalibrationError: If the analysis result does not contain a frequency variable.
        """

        from qiskit_experiments.characterization.qubit_spectroscopy import SpectroscopyAnalysis

        result = exp_data.analysis_result(result_index)

        if "freq" not in result["popt_keys"]:
            raise CalibrationError(
                f"{cls.__name__} updates from analysis classes such as "
                f'{type(SpectroscopyAnalysis.__name__)} which report "freq" in popt.'
            )

        param = parameter
        value = result["popt"][result["popt_keys"].index("freq")]

        cls._add_parameter_value(exp_data, calibrations, value, param, schedule=None, group=group)


class Amplitude(BaseUpdater):
    """Update pulse amplitudes."""

    # pylint: disable=arguments-differ
    @classmethod
    def update(
        cls,
        calibrations: Calibrations,
        exp_data: ExperimentData,
        result_index: int = -1,
        group: str = "default",
        angles_schedules: List[Tuple[float, str, Union[str, ScheduleBlock]]] = None,
    ):
        """Update the amplitude of pulses.

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

        Raises:
            CalibrationError: If the experiment is not of the supported type.
        """
        from qiskit_experiments.calibration.experiments.rabi import Rabi

        if angles_schedules is None:
            angles_schedules = [(np.pi, "amp", "xp")]

        if isinstance(exp_data.experiment, Rabi):
            result = exp_data.analysis_result(result_index)

            freq = result["popt"][result["popt_keys"].index("freq")]
            rate = 2 * np.pi * freq

            for angle, param, schedule in angles_schedules:
                value = np.round(angle / rate, decimals=8)

                cls._add_parameter_value(exp_data, calibrations, value, param, schedule, group)
        else:
            raise CalibrationError(f"{cls.__name__} updates from {type(Rabi.__name__)}.")
