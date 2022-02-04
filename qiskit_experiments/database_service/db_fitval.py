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
import warnings
from typing import Optional

from qiskit_experiments.framework import ExperimentVariable


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

    def __new__(cls, *args, **kwargs) -> ExperimentVariable:
        # Note that FitVal can be instantiated from the json loader thus
        # DeprecationWarning is not captured by default execution setting.
        # It can be only seen if the code is executed from __main__.
        warnings.warn(
            "The FitVal class is deprecated as of Qiskit Experiments 0.3 and will"
            " be removed in a future version. Re-saving loaded experiment data or "
            " analysis results will convert FitVals to ExperimentVariable.",
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
            unit = args[2]
        else:
            unit = kwargs.get("unit")

        return ExperimentVariable(
            value=nominal_value,
            std_dev=std_dev,
            unit=unit,
        )
