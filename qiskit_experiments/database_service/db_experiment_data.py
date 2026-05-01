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

"""Dataclass for experiment data in the database"""
from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self


@dataclass
class DbExperimentData:
    """Dataclass for experiments in the database.

    .. note::

        The documentation does not currently render all the fields of this
        dataclass.

    .. note::

        This is named DbExperimentData to avoid confusion with the main
        :class:`~qiskit_experiments.framework.ExperimentData` class.
    """

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    experiment_type: str | None = None
    backend: str | None = None
    tags: list[str] | None = field(default_factory=list)
    job_ids: list[str] | None = field(default_factory=list)
    share_level: str | None = None
    metadata: dict[str, str] | None = field(default_factory=dict)
    figure_names: list[str] | None = field(default_factory=list)
    notes: str | None = None
    hub: str | None = None
    group: str | None = None
    project: str | None = None
    owner: str | None = None
    creation_datetime: datetime | None = None
    start_datetime: datetime | None = None
    end_datetime: datetime | None = None
    updated_datetime: datetime | None = None

    def __json_encode__(self) -> dict[str, Any]:
        return self.__dict__

    @classmethod
    def __json_decode__(cls, value: dict[str, Any]) -> Self:
        return cls(**value)

    def __str__(self):
        ret = ""
        ret += f"Experiment: {self.experiment_type}"
        ret += f"\nExperiment ID: {self.experiment_id}"
        if self.backend:
            ret += f"\nBackend: {self.backend}"
        if self.tags:
            ret += f"\nTags: {self.tags}"
        ret += f"\nHub\\Group\\Project: {self.hub}\\{self.group}\\{self.project}"
        if self.creation_datetime:
            ret += f"\nCreated at: {self.creation_datetime}"
        if self.start_datetime:
            ret += f"\nStarted at: {self.start_datetime}"
        if self.end_datetime:
            ret += f"\nEnded at: {self.end_datetime}"
        if self.updated_datetime:
            ret += f"\nUpdated at: {self.updated_datetime}"
        if self.metadata:
            ret += f"\nMetadata: {self.metadata}"
        if self.figure_names:
            ret += f"\nFigures: {self.figure_names}"
        return ret

    def copy(self):
        """Creates a deep copy of the data"""
        return DbExperimentData(
            experiment_id=self.experiment_id,
            parent_id=self.parent_id,
            experiment_type=self.experiment_type,
            backend=self.backend,
            tags=copy.copy(self.tags),
            job_ids=copy.copy(self.job_ids),
            share_level=self.share_level,
            metadata=copy.deepcopy(self.metadata),
            figure_names=copy.copy(self.figure_names),
            notes=self.notes,
            hub=self.hub,
            group=self.group,
            project=self.project,
            owner=self.owner,
            creation_datetime=self.creation_datetime,
            start_datetime=self.start_datetime,
            end_datetime=self.end_datetime,
            updated_datetime=self.updated_datetime,
        )
