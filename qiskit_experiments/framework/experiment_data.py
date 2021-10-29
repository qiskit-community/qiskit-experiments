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
from typing import Dict, Optional
from datetime import datetime

from qiskit_experiments.database_service import DbExperimentDataV1
from qiskit_experiments.database_service.database_service import DatabaseServiceV1

LOG = logging.getLogger(__name__)


class ExperimentData(DbExperimentDataV1):
    """Qiskit Experiments Data container class"""

    def __init__(self, experiment=None, backend=None, parent_id=None, job_ids=None):
        """Initialize experiment data.

        Args:
            experiment (BaseExperiment): Optional, experiment object that generated the data.
            backend (Backend): Optional, Backend the experiment runs on.
            parent_id (str): Optional, ID of the parent experiment data
                in the setting of a composite experiment
            job_ids (list[str]): Optional, IDs of jobs submitted for the experiment.
        """
        if experiment is not None:
            backend = backend or experiment.backend
            experiment_type = experiment.experiment_type
        else:
            experiment_type = None

        self._experiment = experiment
        super().__init__(
            experiment_type=experiment_type,
            backend=backend,
            parent_id=parent_id,
            job_ids=job_ids,
            metadata=experiment._metadata() if experiment else {},
        )

    @property
    def experiment(self):
        """Return the experiment for this data.

        Returns:
            BaseExperiment: the experiment object.
        """
        return self._experiment

    @property
    def completion_times(self) -> Dict[str, datetime]:
        """Returns the completion times of the jobs."""
        job_times = {}
        for job_id, job in self._jobs.items():
            if job is not None and "COMPLETED" in job.time_per_step():
                job_times[job_id] = job.time_per_step().get("COMPLETED")

        return job_times

    @classmethod
    def load(cls, experiment_id: str, service: DatabaseServiceV1) -> "ExperimentData":
        """Load a saved experiment data from a database service.

        Args:
            experiment_id: Experiment ID.
            service: the database service.

        Returns:
            The loaded experiment data.
        """
        expdata = DbExperimentDataV1.load(experiment_id, service)
        expdata.__class__ = ExperimentData
        expdata._experiment = None
        return expdata

    def _copy_metadata(self, new_instance: Optional["ExperimentData"] = None) -> "ExperimentData":
        """Make a copy of the experiment metadata.

        Note:
            This method only copies experiment data and metadata, not its
            figures nor analysis results. The copy also contains a different
            experiment ID.

        Returns:
            A copy of the ``ExperimentData`` object with the same data
            and metadata but different ID.
        """
        if new_instance is None:
            new_instance = self.__class__(
                experiment=self.experiment, backend=self.backend, job_ids=self.job_ids
            )
        return super()._copy_metadata(new_instance)

    def __repr__(self):
        out = (
            f"<ExperimentData[{self.experiment_type}]"
            f", backend: {self.backend}"
            f", status: {self.status()}"
            f", experiment_id: {self.experiment_id}>"
        )
        return out
