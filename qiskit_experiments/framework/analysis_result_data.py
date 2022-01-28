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

"""Helper dataclass for constructing analysis results."""

import dataclasses
import logging
from typing import Optional, Dict, Any, List

LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class AnalysisResultData:
    """Dataclass for experiment analysis results"""

    # TODO: move stderr and unit into custom value class
    name: str
    value: Any
    chisq: Optional[float] = None
    quality: Optional[str] = None
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict, hash=False, compare=False)
    device_components: List = dataclasses.field(default_factory=list)

    def __str__(self):
        out = f"{self.name}:"
        out += f"\n- value:{self.value}"
        if self.chisq is not None:
            out += f"\n- chisq: {self.chisq}"
        if self.quality is not None:
            out += f"\n- quality: {self.quality}"
        if self.extra:
            out += f"\n- extra: <{len(self.extra)} items>"
        if self.device_components:
            out += f"\n- device_components: {[str(i) for i in self.device_components]}"
        return out

    def __iter__(self):
        """Return iterator of data fields (attr, value)"""
        return iter((field.name, getattr(self, field.name)) for field in dataclasses.fields(self))
