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

"""Types used by the calibration module."""

from typing import NamedTuple, Tuple, Union
from collections import namedtuple

from qiskit.circuit import ParameterExpression


ParameterKey = namedtuple("ParameterKey", ["parameter", "qubits", "schedule"])
ScheduleKey = namedtuple("ScheduleKey", ["schedule", "qubits"])
ParameterValueType = Union[ParameterExpression, float, int, complex]


class DefaultCalValue(NamedTuple):
    """Defines the structure of a default value."""

    value: Union[float, int, complex]
    parameter: str
    qubits: Tuple
    schedule_name: str
