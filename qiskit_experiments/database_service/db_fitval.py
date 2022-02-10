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

"""DB class for fit value with std error and unit."""

import dataclasses
import numbers
import warnings
from typing import Optional

import numpy as np
import uncertainties


@dataclasses.dataclass(frozen=True)
class FitVal:
    """DEPRECATED A data container for the value estimated by the curve fitting.

    This data is serializable with the Qiskit Experiment json serializer.
    """

    value: float
    stderr: Optional[float] = None
    unit: Optional[str] = None

    def __str__(self):
        out = str(self.value)
        if self.stderr is not None:
            out += f" \u00B1 {self.stderr}"
        if self.unit is not None:
            out += f" {str(self.unit)}"
        return out

    def __new__(cls, *args, **kwargs):
        # Note that FitVal can be instantiated from the json loader thus
        # DeprecationWarning is not captured by default execution setting.
        # It can be only seen if the code is executed from __main__.
        warnings.warn(
            "The FitVal class is deprecated as of Qiskit Experiments 0.3 and will"
            " be removed in a future version. Re-saving loaded experiment data or "
            " analysis results will convert FitVals to UFloat.",
            FutureWarning,
        )
        if len(args) > 0:
            nominal_value = args[0]
        else:
            nominal_value = kwargs.get("value")
        if len(args) > 1:
            std_dev = args[1]
        else:
            std_dev = kwargs.get("stderr")
        if len(args) > 2:
            tag = args[2]
        else:
            tag = kwargs.get("unit")

        if isinstance(nominal_value, numbers.Number):
            return uncertainties.ufloat(
                nominal_value=nominal_value,
                std_dev=std_dev,
                tag=tag,
            )
        elif isinstance(nominal_value, (np.ndarray, list)):
            warnings.warn(
                "The analysis result .value of the @Parameters_ entry now "
                "only returns nominal values. This is because these values should consider "
                "correlation in standard error with the covariance matrix in .extra field. "
                "You can create UFloat values with parameter correlation "
                "with uncertainties.correlated_values(nominal_values, cov_matrix).",
                UserWarning,
            )
            return list(nominal_value)

        raise TypeError(f"Invalid data format {type(nominal_value)} for FitVal data type.")


# Monkey patch uncertainties UFloat class so that it behaves like
# FitVal with deprecation warnings when used as a replacement for
# for analysis result value types


def value(self):
    """DEPRECATED"""
    warnings.warn(
        "The FitVal class has been depreacted and replaced with UFloat "
        "objects, use .nominal_value or .n to access the equivalent of "
        "the FitVal.value property",
        DeprecationWarning,
        stacklevel=2,
    )
    # deprecation warning
    return self.nominal_value


def stderr(self):
    """DEPRECATED"""
    warnings.warn(
        "The FitVal class has been depreacted and replaced with UFloat "
        "objects, use .std_dev or .s to access the equivalent of the "
        "FitVal.stderr property.",
        DeprecationWarning,
        stacklevel=2,
    )
    # deprecation warning
    return self.std_dev


def unit(self):
    """DEPRECATED"""
    warnings.warn(
        "The FitVal class has been depreacted and replaced with UFloat "
        "objects which do not contain units. This will return the .tag "
        "property which may be equivalent to the FitVal.unit property "
        "if constructed from a loaded FitVal.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.tag


# Monkey patch ufloat for deprecated FitVal equivalent API
uncertainties.core.Variable.value = property(value)
uncertainties.core.Variable.stderr = property(stderr)
uncertainties.core.Variable.unit = property(unit)
