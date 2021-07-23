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
"""
Experiment Data class
"""
import logging
from typing import Optional, Dict, List
import dataclasses

from qiskit_experiments.database_service import DbAnalysisResultV1
from numpy import sqrt

LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class AnalysisResult:
    """Experiment analysis result dataclass"""

    name: str
    value: any
    stderr: Optional[any] = None
    unit: Optional[str] = None
    chisq: Optional[float] = None
    quality: Optional[str] = None
    verified: bool = False
    result_id: Optional[str] = None
    tags: List[str] = dataclasses.field(default_factory=list)
    device_components: List = dataclasses.field(default_factory=list)
    extra: Dict[str, any] = dataclasses.field(default_factory=dict, hash=False, compare=False)

    def __str__(self):
        out = f"{type(self).__name__}: {self.name}"
        value = f"value: {self.value}"
        if self.stderr is not None:
            value += f" \u00B1 {self.stderr}"
        if self.unit is not None:
            value += f" {self.unit}"
        out += f"\n- {value}"
        if self.chisq is not None:
            out += f"\n- chisq: {self.chisq}"
        if self.quality is not None:
            out += f"\n- quality: {self.quality}"
        if self.tags:
            out += f"\n- tags: {self.tags}"
        if self.extra:
            out += f"\n- extra: <{len(self.extra)} items>"
        return out

    def __iter__(self):
        """Return iterator of data fields (attr, value)"""
        return iter((field.name, getattr(self, field.name)) for field in dataclasses.fields(self))


def analysis_result_to_db(data: AnalysisResult, experiment_id: str) -> DbAnalysisResultV1:
    """Convert an AnalysisResult to a DbAnalysisResult"""
    result_data = {"value": data.value}
    if data.stderr is not None:
        result_data["stderr"] = data.stderr
        result_data["variance"] = data.stderr ** 2
    if data.unit is not None:
        result_data["unit"] = data.unit
    for key, val in data.extra.items():
        result_data[key] = val

    return DbAnalysisResultV1(
        result_data=result_data,
        result_type=data.name,
        device_components=data.device_components,
        experiment_id=experiment_id,
        chisq=data.chisq,
        quality=data.quality,
        result_id=data.result_id,
        verified=False,
        tags=data.tags,
    )


def db_to_analysis_result(data: DbAnalysisResultV1) -> AnalysisResult:
    """Convert a DbAnalysisResult to an AnalysisResult"""
    extra = data.data().copy()
    value = extra.pop("value")
    unit = extra.pop("unit", None)
    stderr = extra.pop("stderr", None)
    variance = extra.pop("variance", None)
    if stderr is None and variance is not None:
        stderr = sqrt(variance)
    return AnalysisResult(
        name=data.result_type,
        value=value,
        stderr=stderr,
        unit=unit,
        chisq=data.chisq,
        quality=data.quality,
        verified=data.verified,
        result_id=data.result_id,
        device_components=data.device_components,
        tags=data.tags(),
        extra=extra,
    )
