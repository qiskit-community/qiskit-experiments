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

from qiskit_experiments.database_service.device_component import DeviceComponent


LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class AnalysisResultData:
    """Dataclass for experiment analysis results"""

    name: str
    value: Any
    experiment: str = None
    chisq: Optional[float] = None
    quality: Optional[str] = None
    experiment_id: Optional[str] = None
    result_id: Optional[str] = None
    tags: List = dataclasses.field(default_factory=list)
    backend: Optional[str] = None
    run_time: Optional[str] = None
    created_time: Optional[str] = None
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict, hash=False, compare=False)
    device_components: List = dataclasses.field(default_factory=list)

    @classmethod
    def from_table_element(
        cls,
        name: str,
        value: Any,
        experiment: Optional[str] = None,
        components: Optional[List[DeviceComponent]] = None,
        quality: Optional[str] = None,
        experiment_id: Optional[str] = None,
        result_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        backend: Optional[str] = None,
        run_time: Optional[str] = None,
        created_time: Optional[str] = None,
        **extra,
    ):
        """A factory method of AnalysisResultData from a single element in AnalysisResultTable.

        Args:
            name: Name of this entity.
            value: Result value.
            experiment: Type of experiment.
            components: Device component that the experiment was run on.
            quality: Quality of this result.
            experiment_id: ID of associated experiment.
            result_id: Unique ID of this data entry in the storage.
            tags: List of tags.
            backend: Device name that the experiment was run on.
            run_time: A time at the experiment was run.
            created_time: A time at this value was computed.
            **extra: Extra information.
        """
        chisq = extra.pop("chisq", None)

        return AnalysisResultData(
            name=name,
            value=value,
            experiment=experiment,
            chisq=chisq,
            quality=quality,
            experiment_id=experiment_id,
            result_id=result_id,
            tags=tags,
            backend=backend,
            run_time=run_time,
            created_time=created_time,
            device_components=components,
            extra=extra,
        )

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


def as_table_element(
    result_data: AnalysisResultData,
) -> Dict[str, Any]:
    """Python dataclass as_dict-like function to return
    canonical data for analysis AnalysisResultTable.

    Args:
        result_data: AnalysisResultData dataclass to format.

    Returns:
        Formatted data representation in dictionary format.
    """
    out = {
        "name": result_data.name,
        "experiment": result_data.experiment,
        "components": result_data.device_components,
        "value": result_data.value,
        "quality": result_data.quality,
        "experiment_id": result_data.experiment_id,
        "result_id": result_data.result_id,
        "tags": result_data.tags,
        "backend": result_data.backend,
        "run_time": result_data.run_time,
        "created_time": result_data.created_time,
    }
    if result_data.chisq is not None:
        out["chisq"] = result_data.chisq
    out.update(result_data.extra)

    return out
