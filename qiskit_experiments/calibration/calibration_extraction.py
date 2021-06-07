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

"""Class to extract arbitrary data from a result object."""

from abc import abstractmethod
from typing import List, Tuple, Union

from qiskit.circuit import Parameter

from qiskit_experiments.base_analysis import AnalysisResult
from qiskit_experiments.calibration.calibration_types import ParameterValueType


class CalibrationExtraction:
    """Performs non-trivial calibration parameter value extraction from analysis results.

    Most analysis results will contain the value of the calibration parameter under the
    "value" key. However, in some instance more complex parameter value extraction is
    required. For example, from a single Rabi experiment we may update the pulse amplitude
    of the xp pulse as well as the x90p pulse. Both of these amplitudes must be extracted
    from the same result object which can be done by subclassing :class:`CalibrationExtraction`.
    """

    def __init__(self, parameters: List[Union[str, Parameter]], schedule_names: List[str]):
        """
        Args:
            parameters: The parameters to update.
            schedule_names: The names of the schedules that will be updated.
        """
        self._parameters = parameters
        self._schedules = schedule_names

    @abstractmethod
    def __call__(
            self,
            result: AnalysisResult
    ) -> List[Tuple[ParameterValueType, str, Tuple[int, ...], str]]:
        """Method to extract calibration data from a result instance.

        Args:
            result: The result instance from which to extract the parameters.

        Returns:
            A list of tuples. Each tuple is a parameter value, the name of the parameter to
            update, the qubits to update, and the name of the schedule to which the parameter
            belongs.
        """

class RabiExtraction(CalibrationExtraction):
    """Extract rotation angles from a cosine fit."""

    def __init__(self, params_angles_schedules: List[Tuple[str, float, str]]):
        """Class to extract amplitudes for different rotation angles.

        Args:
            params_angles_schedules: A list of tuples. Each tuple corresponds to
                the parameter name to update, the corresponding rotation angle, and the
                name of the schedule to which the parameter belongs.
        """

        parameters, schedules = [], []
        self._angles = []
        for parameter, angle, schedule in params_angles_schedules:
            parameters.append(parameter)
            schedules.append(schedule)
            self._angles.append(angle)

        super().__init__(parameters, schedules)

    def __call__(
        self,
        result: AnalysisResult
    ) -> List[Tuple[ParameterValueType, str, Tuple[int, ...], str]]:
        """TODO"""


