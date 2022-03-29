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

"""Data class for parameter values."""

from dataclasses import dataclass
from datetime import datetime
from typing import Union

from qiskit_experiments.exceptions import CalibrationError


@dataclass
class ParameterValue:
    """A data class to store parameter values."""

    # Value assumed by the parameter
    value: Union[int, float, complex] = None

    # Data time when the value of the parameter was generated
    date_time: datetime = datetime.fromtimestamp(0)

    # A bool indicating if the parameter is valid
    valid: bool = True

    # The experiment from which the value of this parameter was generated.
    exp_id: str = None

    # The group of calibrations to which this parameter belongs
    group: str = "default"

    def __post_init__(self):
        """
        Ensure that the variables in self have the proper types. This allows
        us to give strings to self.__init__ as input which is useful when loading
        serialized parameter values.
        """
        if isinstance(self.valid, str):
            if self.valid == "True":
                self.valid = True
            else:
                self.valid = False

        if isinstance(self.value, str):
            self.value = self._validated_value(self.value)

        if isinstance(self.date_time, str):
            base_fmt = "%Y-%m-%d %H:%M:%S.%f"
            zone_fmts = ["%z", "", "Z"]
            for time_zone in zone_fmts:
                date_format = base_fmt + time_zone
                try:
                    self.date_time = datetime.strptime(self.date_time, date_format)
                    break
                except ValueError:
                    pass
            else:
                formats = list(base_fmt + zone for zone in zone_fmts)
                raise CalibrationError(
                    f"Cannot parse {self.date_time} in either of {formats} formats."
                )

        self.date_time = self.date_time.astimezone()

        if not isinstance(self.value, (int, float, complex)):
            raise CalibrationError(f"Values {self.value} must be int, float or complex.")

        if not isinstance(self.date_time, datetime):
            raise CalibrationError(f"Datetime {self.date_time} must be a datetime.")

        if not isinstance(self.valid, bool):
            raise CalibrationError(f"Valid {self.valid} is not a boolean.")

        if self.exp_id and not isinstance(self.exp_id, str):
            raise CalibrationError(f"Experiment id {self.exp_id} is not a string.")

        if not isinstance(self.group, str):
            raise CalibrationError(f"Group {self.group} is not a string.")

    @staticmethod
    def _validated_value(value: str) -> Union[int, float, complex]:
        """
        Convert the string representation of value to the correct type.

        Args:
            value: The string to convert to either an int, float, or complex.

        Returns:
            value converted to either int, float, or complex.

        Raises:
            CalibrationError: If the conversion fails.
        """
        try:
            return int(value)
        except ValueError:
            pass

        try:
            return float(value)
        except ValueError:
            pass

        try:
            return complex(value)
        except ValueError as val_err:
            raise CalibrationError(
                f"Could not convert {value} to int, float, or complex."
            ) from val_err
