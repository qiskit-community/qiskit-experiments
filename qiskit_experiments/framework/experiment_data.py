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

LOG = logging.getLogger(__name__)


class ExperimentData(DbExperimentDataV1):
    """Qiskit Experiments Data container class"""

    def __init__(
        self,
        experiment=None,
        backend=None,
        job_ids=None,
    ):
        """Initialize experiment data.

        Args:
            experiment (BaseExperiment): Optional, experiment object that generated the data.
            backend (Backend): Optional, Backend the experiment runs on.
            job_ids (list[str]): Optional, IDs of jobs submitted for the experiment.

        Raises:
            ExperimentError: If an input argument is invalid.
        """
        self._experiment = experiment
        super().__init__(
            experiment_type=experiment.experiment_type if experiment else None,
            backend=backend,
            job_ids=job_ids,
            metadata=experiment._metadata() if experiment else {},
        )

    @property
    def experiment(self):
        """Return Experiment object.

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
            new_instance = ExperimentData(
                experiment=self.experiment, backend=self.backend, job_ids=self.job_ids
            )
        return super()._copy_metadata(new_instance)

    def __repr__(self):
        out = f"{type(self).__name__}({self.experiment_type}"
        out += f", {self.experiment_id}"
        if self.backend:
            out += f", backend={self.backend}"
        if self.job_ids:
            out += f", job_ids={self.job_ids}"
        out += ")"
        return out
