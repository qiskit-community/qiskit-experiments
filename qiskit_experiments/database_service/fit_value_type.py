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

import operator
from typing import Optional

import numpy as np


def _check_values_comparable(method):
    """A method decorator to check if two values are operable."""

    def _wraps(self, other):
        if isinstance(other, FitVal):
            if self.unit is not None and self.unit != other.unit:
                # values with different units are not operable.
                raise ValueError(f"Two values {self} and {other} are not operable.")
            if self.stderr is None or other.stderr is None:
                # no stdev is provided. use equivalent method.
                standard_method = getattr(operator, method.__name__)
                return float(standard_method(self.value, other.value))
            return method(self, other)
        else:
            # other is number object. ignore stdev and unit of self.
            standard_method = getattr(operator, method.__name__)
            return float(standard_method(self.value, other))

    return _wraps


class FitVal:
    """Extended float type to support value error and unit.

    If two ``FitVal`` types are operated, it raises an error if they have different units.
    If one value doesn't provide an error, normal float value operation is performed.
    If both values provide errors, error propagation is also calculated.
    Only basic arithmetic operations are supported.

    .. example::

        >>> a = FitVal(3.0, 0.1, "s")
        >>> b = FitVal(5.0, 0.2, "s")
        >>> print(a + b)
        8.0 ± 0.223606797749979 [s]
        >>> c = 5.0
        >>> print(a + c)
        8.0

    This value is serializable with the Qiskit Experiment json serializer.
    """

    def __init__(
        self,
        value: float,
        stderr: Optional[float] = None,
        unit: Optional[str] = "a.u.",
    ):
        """Create new fit value instance.

        Args:
            value: Value.
            stderr: Optional. Standard error of the value.
            unit: Optional. Unit of the value.

        Raises:
            TypeError: When ``value`` or ``stderr`` are non floating value.
            ValueError: When negative ``error`` is provided.
        """
        self._value = value
        self._stderr = stderr
        self._unit = unit

        if not np.isreal(self._value):
            raise TypeError("Value should be float value.")

        if not np.isreal(self._stderr):
            raise TypeError("Error should be float value.")

        if self._stderr is not None and self._stderr < 0:
            raise ValueError("Error should be positive float value.")

    @property
    def value(self):
        """Return value."""
        return self._value

    @property
    def stderr(self):
        """Return value error."""
        return self._stderr

    @property
    def unit(self):
        """Return unit of the value."""
        return self._unit

    @_check_values_comparable
    def __add__(self, other):
        stderr = np.sqrt(self.stderr ** 2 + other.stderr ** 2)
        return self.__class__(value=self._value + other.value, stderr=stderr, unit=self.unit)

    @_check_values_comparable
    def __sub__(self, other):
        stderr = np.sqrt(self.stderr ** 2 + other.stderr ** 2)
        return self.__class__(value=self._value - other.value, stderr=stderr, unit=self.unit)

    @_check_values_comparable
    def __mul__(self, other):
        stderr = np.sqrt((other.value * self.stderr) ** 2 + (self.value * other.stderr) ** 2)
        return self.__class__(value=self._value * other.value, stderr=stderr, unit=self.unit)

    @_check_values_comparable
    def __truediv__(self, other):
        stderr = np.sqrt(
            (self.stderr / other.value) ** 2 + (other.stderr * (self.value / other.value ** 2)) ** 2
        )
        return self.__class__(value=self._value / other.value, stderr=stderr, unit=self.unit)

    @_check_values_comparable
    def __ge__(self, other):
        return operator.ge(self.value, other.value)

    @_check_values_comparable
    def __le__(self, other):
        return operator.le(self.value, other.value)

    @_check_values_comparable
    def __gt__(self, other):
        return operator.gt(self.value, other.value)

    @_check_values_comparable
    def __lt__(self, other):
        return operator.lt(self.value, other.value)

    @_check_values_comparable
    def __eq__(self, other):
        # compare with float value is allowed. e.g. 3.3 ± 1.0 == 3.3
        return self.value == other.value and self.stderr == other.stderr and self.unit == other.unit

    def __abs__(self):
        return self.__class__(value=abs(self.value), stderr=self.stderr, unit=self.unit)

    def __pos__(self):
        return self.__class__(value=self.value, stderr=self.stderr, unit=self.unit)

    def __neg__(self):
        return self.__class__(value=-self.value, stderr=self.stderr, unit=self.unit)

    def __float__(self):
        return float(self.value)

    def __hash__(self):
        return hash((self._value, self._stderr, self._unit))

    def __str__(self):
        if self._stderr is not None:
            value_rep = f"{self._value}\u00B1{self._stderr}"
        else:
            value_rep = str(self._value)

        if self._unit:
            return f"{value_rep} [{self._unit}]"
        else:
            return value_rep

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(value={self._value}, stderr={self._stderr}, unit={self._unit})"
        )
