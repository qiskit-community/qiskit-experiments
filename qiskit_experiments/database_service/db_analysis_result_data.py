# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Dataclass for analysis result data in the database"""
from __future__ import annotations

import copy
import uuid

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from datetime import datetime

from .constants import ResultQuality
from .device_component import DeviceComponent

if TYPE_CHECKING:
    from typing import Self


@dataclass
class DbAnalysisResultData:
    """Dataclass for experiment analysis results in the database.

    .. note::

        The documentation does not currently render all the fields of this
        dataclass.

    .. note::

        This class is named DbAnalysisResultData to avoid confusion with the
        :class:`~qiskit_experiments.framework.AnalysisResult` class.
    """

    result_id: str | None = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str | None = None
    result_type: str | None = None
    result_data: dict[str, Any] | None = field(default_factory=dict)
    device_components: list[str | DeviceComponent] | str | DeviceComponent | None = field(
        default_factory=list
    )
    quality: ResultQuality | None = ResultQuality.UNKNOWN
    verified: bool | None = False
    tags: list[str] | None = field(default_factory=list)
    backend_name: str | None = None
    creation_datetime: datetime | None = None
    updated_datetime: datetime | None = None
    chisq: float | None = None

    def __json_encode__(self) -> dict[str, Any]:
        return self.__dict__

    @classmethod
    def __json_decode__(cls, value: dict[str, Any]) -> Self:
        return cls(**value)

    def __str__(self):
        ret = f"Result {self.result_type}"
        ret += f"\nResult ID: {self.result_id}"
        ret += f"\nExperiment ID: {self.experiment_id}"
        ret += f"\nBackend: {self.backend_name}"
        ret += f"\nQuality: {self.quality}"
        ret += f"\nVerified: {self.verified}"
        ret += f"\nDevice components: {self.device_components}"
        ret += f"\nData: {self.result_data}"
        if self.chisq:
            ret += f"\nChi Square: {self.chisq}"
        if self.tags:
            ret += f"\nTags: {self.tags}"
        if self.creation_datetime:
            ret += f"\nCreated at: {self.creation_datetime}"
        if self.updated_datetime:
            ret += f"\nUpdated at: {self.updated_datetime}"
        return ret

    def copy(self):
        """Creates a deep copy of the data"""
        return DbAnalysisResultData(
            result_id=self.result_id,
            experiment_id=self.experiment_id,
            result_type=self.result_type,
            result_data=copy.deepcopy(self.result_data),
            device_components=copy.copy(self.device_components),
            quality=self.quality,
            verified=self.verified,
            tags=copy.copy(self.tags),
            backend_name=self.backend_name,
            creation_datetime=self.creation_datetime,
            updated_datetime=self.updated_datetime,
            chisq=self.chisq,
        )
