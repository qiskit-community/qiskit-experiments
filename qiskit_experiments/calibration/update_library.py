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

from qiskit.pulse import ScheduleBlock

from qiskit_experiments.experiment_data import ExperimentData
from qiskit_experiments.calibration.backend_calibrations import BackendCalibrations
from qiskit_experiments.calibration.calibrations import Calibrations
from qiskit_experiments.calibration.parameter_value import ParameterValue
from qiskit_experiments.calibration.exceptions import CalibrationError


class BaseUpdater(ABC):
    """A base class to update calibrations."""

    def __init__(self):
        """Initialize the class."""
        self.qubits = None
        self.param = None
        self.value = None
        self.schedule = None

    @staticmethod
    def _time_stamp(exp_data: ExperimentData) -> datetime:
        """Helper method to extract the datetime."""
        all_times = exp_data.completion_times.values()
        if all_times:
            return max(all_times)

        return datetime.now()

    def _update(self, exp_data: ExperimentData, cal: Calibrations, group: str = "default"):
        """Update the calibrations with the values."""
        value = ParameterValue(
            value=self.value,
            date_time=BaseUpdater._time_stamp(exp_data),
            group=group,
            exp_id=exp_data.experiment_id,
        )

        cal.add_parameter_value(value, self.param, self.qubits, self.schedule)

    @abstractmethod
    def update(self, exp_data: ExperimentData, calibrations: BackendCalibrations, **options):
        """Update the calibrations based on the data.

        Child update classes must implement this function.
        """


class Frequency(BaseUpdater):
    """Update frequencies."""

    # pylint: disable=arguments-differ
    def update(
        self,
        exp_data: ExperimentData,
        calibrations: BackendCalibrations,
        result_index: int = -1,
        group: str = "default",
        parameter: str = BackendCalibrations.__qubit_freq_parameter__,
    ):
        """Update a qubit frequency from QubitSpectroscopy.

        Args:
            exp_data: The experiment data from which to update.
            calibrations: The calibrations to update.
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
                f"{self.__class__.__name__} updates from analysis classes such as "
                f'{type(SpectroscopyAnalysis.__name__)} which report "freq" in popt.'
            )

        self.qubits = exp_data.data(0)["metadata"]["qubits"]
        self.param = parameter
        self.value = result["popt"][result["popt_keys"].index("freq")]

        self._update(exp_data, calibrations, group)


class Amplitude(BaseUpdater):
    """Update pulse amplitudes."""

    # pylint: disable=arguments-differ
    def update(
        self,
        exp_data: ExperimentData,
        calibrations: Calibrations,
        result_index: int = -1,
        group: str = "default",
        angles_schedules: List[Tuple[float, str, Union[str, ScheduleBlock]]] = None,
    ):
        """
        Args:
            exp_data: The experiment data from which to update.
            calibrations: The calibrations to update.
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

        self.qubits = exp_data.data(0)["metadata"]["qubits"]

        if isinstance(exp_data.experiment, Rabi):
            result = exp_data.analysis_result(result_index)

            freq = result["popt"][result["popt_keys"].index("freq")]

            rate = 2 * np.pi * freq
            for angle, param, schedule in angles_schedules:
                self.value = np.round(angle / rate, decimals=8)
                self.schedule = schedule
                self.param = param

                self._update(exp_data, calibrations, group)
        else:
            raise CalibrationError(f"{self.__class__.__name__} updates from {type(Rabi.__name__)}.")
