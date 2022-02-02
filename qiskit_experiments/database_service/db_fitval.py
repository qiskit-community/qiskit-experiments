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
from typing import Optional


@dataclasses.dataclass(frozen=True)
class FitVal:
    """A data container for the value estimated by the curve fitting.

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
