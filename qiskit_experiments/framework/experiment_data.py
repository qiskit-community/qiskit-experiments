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
from typing import Dict, Union, List, Optional
from datetime import datetime

from qiskit_experiments.database_service import DbExperimentDataV1, DbAnalysisResultV1
from qiskit_experiments.framework.analysis_result import (
    AnalysisResult,
    db_to_analysis_result,
    analysis_result_to_db,
)


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

    def add_analysis_results(
        self,
        results: Union[AnalysisResult, List[AnalysisResult]],
    ) -> None:
        """Save the analysis result.

        Args:
            results: Analysis results to be saved.
        """
        if not isinstance(results, list):
            results = [results]

        super().add_analysis_results(
            [analysis_result_to_db(result, self.experiment_id) for result in results]
        )

    def analysis_results(
        self, index: Optional[Union[int, slice, str]] = None, refresh: bool = False
    ) -> Union[AnalysisResult, List[AnalysisResult]]:
        # TODO: This needs handling of the converted results so any changes
        # to the returned objects (such as setting tags, verified etc)
        # can be updated and autosaved to the database results.
        results = super().analysis_results(index, refresh)
        if isinstance(results, list):
            return [db_to_analysis_result(i) for i in results]
        return db_to_analysis_result(results)

    @property
    def completion_times(self) -> Dict[str, datetime]:
        """Returns the completion times of the jobs."""
        job_times = {}
        for job_id, job in self._jobs.items():
            if job is not None and "COMPLETED" in job.time_per_step():
                job_times[job_id] = job.time_per_step().get("COMPLETED")

        return job_times
