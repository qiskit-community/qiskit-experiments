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


@dataclass
class ParameterValue:
    """A data class to store parameter values."""

    # Value assumed by the parameter
    value: Union[int, float] = None

    # Data time when the value of the parameter was generated
    date_time: datetime = datetime.fromtimestamp(0)

    # A bool indicating if the parameter is valid
    valid: bool = True

    # The experiment from which the value of this parameter was generated.
    exp_id: str = None

    # The group of calibrations to which this parameter belongs
    group: str = 'default'
