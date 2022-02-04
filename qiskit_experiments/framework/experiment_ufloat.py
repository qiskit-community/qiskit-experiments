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
# pylint: disable=method-hidden,too-many-return-statements,c-extension-no-member

"""Monkey patch to uncertainties object for JSON serialization and unit handling.

This module provides the customized :class:`Variable` and :class:`AffineScalarFunc`
class defined in uncertainties (https://pythonhosted.org/uncertainties/).
Relationship of these classes are somewhat of :class:`Parameter`
and :class:`ParameterExpression` in Qiskit.

A single number with the value and standard error can be defined as
a :class:`Variable` instance, however, once we apply mathematical binary operator to
this instance (even with constant number), this instance is typecasted into
its parent class :class:`AffineScalarFunc` which stores correlation of
:class:`Variable` instances comprising the operated outcome.
Note that two numbers having the same nominal values and standard error should be
distinguished to correctly compute correlation,
i.e. <0.1+/-0.2> - <0.1+/-0.2> = <0.0+/-0.28284271247461906>.
In the custom class ExperimentVariable, this mechanism is realized by attaching
UUID at instance creation and hashing the object based on the string.

The computation of error propagation is still offloaded to the original package.
Owing to typecasting during computation, we cannot provide a standalone custom class
in Qiskit Experiments because custom class instance will be immediately typecasted
and designed functionality will be eventually lost.
Instead, this module monkey patches :class:`Variable` and :class:`AffineScalarFunc`
in uncertainties without breaking the error propagation mechanism there.

Basically customized class provides the capability of JSON serialization with
ExperimentEncoder and :attr:`unit` attribute for backward compatibility
with :class:`FitVal` object which had been used to store the measured quantities.
"""

import uuid
import warnings
from typing import Optional, Dict

import copy
import uncertainties
import numpy as np


class ExperimentAffineScalarFunc(uncertainties.core.AffineScalarFunc):

    __slots__ = ("_nominal_value", "_linear_part", "_unit")

    def __init__(self, nominal_value, linear_part):
        super().__init__(nominal_value=nominal_value, linear_part=linear_part)
        self._unit = ""

    @property
    def unit(self) -> str:
        """Return physical unit of this value."""
        return self._unit

    @unit.setter
    def unit(self, unit: str):
        """Set new unit."""
        self._unit = unit

    @property
    def value(self) -> float:
        """Deprecated. Backward compatibility for FitVal."""
        warnings.warn(
            "The FitVal class has been deprecated and replaced with UFloat "
            "objects, use .nominal_value or .n to access the equivalent of "
            "the FitVal.value property",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.nominal_value

    @property
    def stderr(self) -> float:
        """Deprecated. Backward compatibility for FitVal."""
        warnings.warn(
            "The FitVal class has been deprecated and replaced with UFloat "
            "objects, use .std_dev or .s to access the equivalent of the "
            "FitVal.stderr property.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.std_dev

    def __repr__(self):
        # Add unit after number if exist
        num_repr = super().__repr__()

        if self.unit:
            return num_repr + f" {self.unit}"
        return num_repr

    def __deepcopy__(self, memo):
        # same as uncertainties.core.AffineScalarFunc but add units if exist
        instance = ExperimentAffineScalarFunc(
            self._nominal_value,
            copy.deepcopy(self._linear_part),
        )
        instance.unit = self.unit

        return instance

    def __json_encode__(self) -> Dict:
        """Convert to format that can be JSON serialized

        Raises:
            TypeError: When derivative is not JSON serializable.
        """
        # derivative is a dictionary of ExperimentVariables to express correlation of them.
        # ExperimentEncoder assumes the dict key is one of python built-in types,
        # thus it should be converted into list [[key, val], ...] so that key is also hooked.
        # JSON serializer will recursively serialize the object thus
        # no more complicated logic is necessary implemented here.

        # this could be defaultdict
        derivatives = dict(self.derivatives)
        derivatives_list = list(derivatives.items())

        return {
            "nominal_value": self.nominal_value,
            "derivatives": derivatives_list,
            "unit": self.unit,
        }

    @classmethod
    def __json_decode__(cls, value: Dict) -> "ExperimentAffineScalarFunc":
        """Load from JSON compatible format"""
        derivatives = dict(value["derivatives"])

        instance = cls(
            nominal_value=value["nominal_value"],
            linear_part=uncertainties.core.LinearCombination(derivatives),
        )
        instance.unit = value["unit"]

        return instance


class ExperimentVariable(ExperimentAffineScalarFunc):
    """Qiskit Experiment implementation of uncertainties.core.Variable.

    This is monkey patch of Variable in the original module.
    This class supports JSON serialization with ExperimentEncoder and also upgraded
    to take physical unit of the value which may be shown in the database UI.
    """

    __slots__ = ("_std_dev", "tag", "_unit", "_identifier")

    # pylint: disable=unused-argument
    def __new__(
        cls,
        value: float,
        std_dev: float,
        tag: Optional[str] = None,
        unit: Optional[str] = None,
        identifier: Optional[str] = None,
    ):
        # ExperimentVariable relies on self._identifier being set prior to
        # attributes for object hash generation.
        obj = object.__new__(cls)

        if identifier is None:
            # Give unique ID to compute error with parameter correlation.
            obj._identifier = uuid.uuid4().hex
        else:
            # For deserialization
            obj._identifier = identifier

        return obj

    def __init__(
        self,
        value: float,
        std_dev: float,
        tag: Optional[str] = None,
        unit: Optional[str] = None,
    ):
        """Create new value with uncertainty.

        Args:
            value: Nominal part of the value.
            std_dev: Uncertainty of the value which is assumed to be
                standard error. This should be nonzero float or NaN.
            tag: Name of this value.
            unit: Physical unit of this value.
        """
        super().__init__(
            nominal_value=value, linear_part=uncertainties.core.LinearCombination({self: 1.0})
        )
        self._std_dev = std_dev
        self._unit = unit
        self.tag = tag

    @property
    def std_dev(self) -> float:
        """Return standard deviation."""
        # same as uncertainties.core.Variable
        return self._std_dev

    @std_dev.setter
    def std_dev(self, std_dev: float):
        """Set new standard deviation, which is expected behavior."""
        # same as uncertainties.core.Variable
        if std_dev < 0 and not (np.isnan(std_dev) or np.isinf(std_dev)):
            raise ValueError("Standard deviation should be non-negative.")

        self._std_dev = uncertainties.core.CallableStdDev(std_dev)

    def __repr__(self):
        # same as uncertainties.core.Variable
        num_repr = super().__repr__()

        if self.tag is None:
            return num_repr
        else:
            return "< %s = %s >" % (self.tag, num_repr)

    def __copy__(self):
        # same as uncertainties.core.Variable but with more constructor args
        return ExperimentVariable(
            value=self.nominal_value,
            std_dev=self.std_dev,
            tag=self.tag,
            unit=self.unit,
        )

    def __deepcopy__(self, memo):
        # same as uncertainties.core.Variable
        return self.__copy__()

    def __hash__(self):
        # Variable objects are distinguished by unique hash,
        # so that two identical values defined separately can be
        # distinguished, i.e. correlation. In original ``uncertainties``
        # implementation, this returns object id which cannot be
        # JSON serialized. In experiment version this returns UUID.
        return hash(self._identifier)

    def __eq__(self, other):
        # This checks all slots and instance type.
        #
        # In original implementation, __eq__ of AffineScalarFunc is called,
        # which subtracts the self from other, then checks if both the nominal value
        # and standard error become zero, i.e. even if nominal value is zero,
        # non correlated (=non identical) parameters will give finite standard error.
        #
        # The standard error is computed by expanding LinearCombination,
        # which is an object that stores correlation of values as a python dictionary
        # keyed on AffineScalarFunc objects composing the value.
        # Note that ExperimentVariable overrides hash function so that it can be
        # JSON serialized. This yields the situation there are two values in memory,
        # that have identical hash but different object ids because of the round trip
        # of serialization. This crashes construction of LinearCombination dictionary.
        #
        # To avoid this edge case, this class manually checks if all values
        # stored in two instances are identical, rather than computing error propagation.
        #
        # Note that uniqueness of object is guaranteed by _identifier since this is
        # randomly assigned UUID.
        checks = ["_identifier", "_nominal_value", "_std_dev", "tag", "_unit"]
        if type(other) is type(self):
            for check in checks:
                if getattr(self, check, None) != getattr(other, check, None):
                    return False
            return True
        return False

    def __json_encode__(self) -> Dict:
        """Convert to format that can be JSON serialized"""
        return {
            "nominal_value": self.nominal_value,
            "std_dev": self.std_dev,
            "tag": self.tag,
            "unit": self._unit,
            "identifier": self._identifier,
        }

    @classmethod
    def __json_decode__(cls, value: Dict) -> "ExperimentVariable":
        """Load from JSON compatible format"""
        nominal_value = value["nominal_value"]
        std_dev = value["std_dev"]
        tag = value["tag"]
        unit = value["unit"]
        identifier = value["identifier"]

        instance = ExperimentVariable.__new__(
            ExperimentVariable,
            value=nominal_value,
            std_dev=std_dev,
            identifier=identifier,
        )
        instance.__init__(value=nominal_value, std_dev=std_dev, tag=tag, unit=unit)

        return instance


# Monkey patch uncertainty package
uncertainties.core.Variable = ExperimentVariable
uncertainties.core.AffineScalarFunc = ExperimentAffineScalarFunc
