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
from typing import List, Union
from datetime import datetime
from dataclasses import dataclass

from qiskit_experiments.database_service import DbExperimentDataV1
from qiskit_experiments.database_service.device_component import DeviceComponent


LOG = logging.getLogger(__name__)

@dataclass
class AnalysisResultData:
    """Placeholder class"""
    result_data: dict
    result_type: str
    device_components: List[Union[DeviceComponent, str]]
    chisq: float = None
    quality: str = None
    verified: bool = False

    __keys_not_shown__ = tuple()
    """Data keys of analysis result which are not directly shown in `__str__` method"""

    def __str__(self):
        out = f"AnalysisResultData"
        out += f"\n- result_type: {self.result_type}"
        out += f"\n- device_components: {self.device_components}"
        if self.chisq:
            out += f"\n- chisq: {self.chisq}"
        if self.quality:
            out += f"\n- quality: {self.quality}"
        out += f"\n- verified: {self.verified}"
        out += f"\n- result_data:"
        for key, value in self.items():
            if key in self.__keys_not_shown__:
                continue
            out += f"\n  - {key}: {value}"
        return out


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
