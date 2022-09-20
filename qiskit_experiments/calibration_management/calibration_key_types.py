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
ParameterValueType = Union[ParameterExpression, float, int, complex]


class DefaultCalValue(NamedTuple):
    """Defines the structure of a default value."""

    value: Union[float, int, complex]
    parameter: str
    qubits: Tuple
    schedule_name: str


class ScheduleKey(NamedTuple):
    """Defines the structure of a key to find a schedule."""

    schedule: str  # Name of the schedule
    qubits: Tuple  # Qubits the schedule acts on

    def __repr__(self):
        return f"{self.schedule}::{self.qubits}"

    @classmethod
    def from_repr(cls, rep_str: str) -> "ScheduleKey":
        """Construct a key form its representation as a string."""
        name, qubits = rep_str.split("::")
        qubits = tuple(int(qubit) for qubit in qubits.strip("( )").split(",") if qubit != "")
        return ScheduleKey(name, qubits)
