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

"""Extended value."""

from typing import Optional

import numpy as np


class fitval:
    """Extended float type to support value error and unit.

    This value is serializable with the Qiskit Experiment json serializer.
    """

    def __init__(
        self,
        value: float,
        stdev: Optional[float] = None,
        unit: Optional[str] = None,
    ):
        """Create new fit value instance.

        Args:
            value: Value.
            stdev: Optional. Standard deviation of value.
            unit: Optional. Unit of this value.

        Raises:
            TypeError: When ``value`` or ``stdev`` are non floating value.
            ValueError: When negative ``error`` is provided.
        """
        self._value = value
        self._stdev = stdev
        self._unit = unit

        if not np.isreal(self._value):
            raise TypeError("Value should be float value.")

        if not np.isreal(self._stdev):
            raise TypeError("Value error should be float value.")

        if self._stdev is not None and self._stdev < 0:
            raise ValueError("Value error should be positive float value.")

    @property
    def value(self):
        """Return value."""
        return self._value

    @property
    def stdev(self):
        """Return value error."""
        return self._stdev

    @property
    def unit(self):
        """Return unit of the value."""
        return self._unit

    def __hash__(self):
        return hash((self._value, self._stdev, self._unit))

    def __str__(self):
        if self._stdev is not None:
            value_rep = f"{self._value}\u00B1{self._stdev}"
        else:
            value_rep = str(self._value)

        if self._unit:
            return f"{value_rep} [{self._unit}]"
        else:
            return value_rep

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(value={self._value}, stdev={self._stdev}, unit={self._unit})"
        )
