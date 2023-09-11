# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Entry for artifact data.
"""

from typing import Any, Optional
from datetime import datetime
import json
import uuid

from dateutil import tz

from qiskit_experiments.framework.json import ExperimentEncoder, ExperimentDecoder


class ArtifactData:
    """"""

    _json_encoder = ExperimentEncoder
    _json_decoder = ExperimentDecoder

    def __init__(
        self,
        name: str,
        data: Any,
        artifact_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        created_time: Optional[str] = None,
    ):
        self._name = name
        self._data = data
        self._experiment_id = experiment_id
        self._artifact_id = artifact_id or str(uuid.uuid4())
        self._created_time = created_time or datetime.now(tz.tzlocal())

    @property
    def name(self):
        return self._name

    @property
    def data(self):
        return self._data

    @property
    def dtype(self):
        """Data type of the payload."""
        return self._data.__class__.__name__

    @property
    def experiment_id(self):
        return self._experiment_id

    @experiment_id.setter
    def experiment_id(self, new_id: str):
        if self._experiment_id:
            raise ValueError("Experiment ID is already set.")
        self._experiment_id = new_id

    @property
    def artifact_id(self):
        return self._artifact_id

    @property
    def created_time(self):
        return self._created_time

    def __repr__(self):
        return f"ArtifactData(name={self._name}, dtype={self.dtype}, uid={self._artifact_id})"

    def __json_encode__(self):
        return {
            "name": self._name,
            "data": json.dumps(self._data, cls=self._json_encoder),
            "experiment_id": self._experiment_id,
            "artifact_id": self._artifact_id,
            "created_time": self._created_time.isoformat(),
        }
