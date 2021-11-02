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
from __future__ import annotations
import logging
from typing import Dict, Optional, List, Union
from datetime import datetime
import warnings
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit_experiments.database_service import DbExperimentDataV1 as DbExperimentData
from qiskit_experiments.database_service.database_service import (
    DatabaseServiceV1 as DatabaseService,
)
from qiskit_experiments.database_service.utils import ThreadSafeOrderedDict, combined_timeout

LOG = logging.getLogger(__name__)


class ExperimentData(DbExperimentData):
    """Qiskit Experiments Data container class"""

    def __init__(
        self,
        experiment: Optional["BaseExperiment"] = None,
        backend: Optional[Backend] = None,
        parent_id: Optional[str] = None,
        job_ids: Optional[List[str]] = None,
        child_data: Optional[List[ExperimentData]] = None,
    ):
        """Initialize experiment data.

        Args:
            experiment: Optional, experiment object that generated the data.
            backend: Optional, Backend the experiment runs on.
            parent_id: Optional, ID of the parent experiment data
                in the setting of a composite experiment
            job_ids: Optional, IDs of jobs submitted for the experiment.
            child_data: Optional, list of child experiment data.
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

        # Add component data and set parent ID to current container
        self._child_data = ThreadSafeOrderedDict()
        if child_data is not None:
            self._set_child_data(child_data)

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

    def add_child_data(self, experiment_data: ExperimentData):
        """Add child experiment data to the current experiment data"""
        experiment_data._parent_id = self.experiment_id
        self._child_data[experiment_data.experiment_id] = experiment_data

    def child_data(
        self, index: Optional[Union[int, slice, str]] = None
    ) -> Union[ExperimentData, List[ExperimentData]]:
        """Return child experiment data.

        Args:
            index: Index of the child experiment data to be returned.
                Several types are accepted for convenience:

                    * None: Return all child data.
                    * int: Specific index of the child data.
                    * slice: A list slice of indexes.
                    * str: experiment ID of the child data.

        Returns:
            The requested single or list of child experiment data.

        Raises:
            QiskitError: if the index or ID of the child experiment data
                         cannot be found.
        """
        if index is None:
            return self._child_data.values()
        if isinstance(index, (int, slice)):
            return self._child_data.values()[index]
        if isinstance(index, str):
            return self._child_data[index]
        raise QiskitError(f"Invalid index type {type(index)}.")

    def component_experiment_data(
        self, index: Optional[Union[int, slice]] = None
    ) -> Union[ExperimentData, List[ExperimentData]]:
        """Return child experiment data"""
        warnings.warn(
            "This method is deprecated and will be removed next release. "
            "Use the `child_data` method instead.",
            DeprecationWarning,
        )
        return self.child_data(index)

    def save(self) -> None:
        super().save()
        for data in self._child_data.values():
            original_verbose = data.verbose
            data.verbose = False
            data.save()
            data.verbose = original_verbose

    def save_metadata(self) -> None:
        # Copy child experiment IDs to metadata
        if self._child_data:
            self._metadata["child_data_ids"] = self._child_data.keys()
        super().save_metadata()
        for data in self._child_data.values():
            data.save_metadata()

    @classmethod
    def load(cls, experiment_id: str, service: DatabaseService) -> ExperimentData:
        expdata = DbExperimentData.load(experiment_id, service)
        expdata.__class__ = ExperimentData
        expdata._experiment = None
        child_data_ids = expdata.metadata.pop("child_data_ids", [])
        child_data = [ExperimentData.load(child_id, service) for child_id in child_data_ids]
        expdata._set_child_data(child_data)
        return expdata

    def _set_child_data(self, child_data: List[ExperimentData]):
        """Set child experiment data for the current experiment."""
        self._child_data = ThreadSafeOrderedDict()
        for data in child_data:
            self.add_child_data(data)

    def _set_service(self, service: DatabaseService) -> None:
        """Set the service to be used for storing experiment data.

        Args:
            service: Service to be used.

        Raises:
            DbExperimentDataError: If an experiment service is already being used.
        """
        super()._set_service(service)
        for data in self._child_data.values():
            data._set_service(service)

    @DbExperimentData.share_level.setter
    def share_level(self, new_level: str) -> None:
        """Set the experiment share level.

        Args:
            new_level: New experiment share level. Valid share levels are provider-
                specified. For example, IBM Quantum experiment service allows
                "public", "hub", "group", "project", and "private".
        """
        self._share_level = new_level
        for data in self._child_data.values():
            original_auto_save = data.auto_save
            data.auto_save = False
            data.share_level = new_level
            data.auto_save = original_auto_save
        if self.auto_save:
            self.save_metadata()

    def block_for_results(self, timeout: Optional[float] = None) -> ExperimentData:
        """Block until all pending jobs and analysis callbacks finish.

        Args:
            timeout: Timeout waiting for results.

        Returns:
            The experiment data with finished jobs and post-processing.
        """
        _, timeout = combined_timeout(super().block_for_results, timeout)
        for subdata in self._child_data.values():
            _, timeout = combined_timeout(subdata.block_for_results, timeout)
        return self

    def _copy_metadata(self, new_instance: Optional[ExperimentData] = None) -> ExperimentData:
        """Make a copy of the experiment metadata.

        Note:
            This method only copies experiment data and metadata, not its
            figures nor analysis results. The copy also contains a different
            experiment ID.

        Returns:
            A copy of the ``ExperimentData`` object with the same data
            and metadata but different ID.
        """
        new_instance = super()._copy_metadata(new_instance)
        if self.experiment is None:
            new_instance._experiment = None
        else:
            new_instance._experiment = self.experiment.copy()

        # Recursively copy metadata of child data
        child_data = [data._copy_metadata() for data in self._child_data.values()]
        new_instance._set_child_data(child_data)
        return new_instance

    def __repr__(self):
        out = (
            f"<ExperimentData[{self.experiment_type}]"
            f", backend: {self.backend}"
            f", status: {self.status()}"
            f", experiment_id: {self.experiment_id}>"
        )
        return out

    def __str__(self):
        line = 51 * "-"
        n_res = len(self._analysis_results)
        status = self.status()
        ret = line
        ret += f"\nExperiment: {self.experiment_type}"
        ret += f"\nExperiment ID: {self.experiment_id}"
        if self._parent_id:
            ret += f"\nParent ID: {self._parent_id}"
        if self._child_data:
            ret += f"\nChild Experiment Data: {len(self._child_data)}"
        ret += f"\nStatus: {status}"
        if status == "ERROR":
            ret += "\n  "
            ret += "\n  ".join(self._errors)
        if self.backend:
            ret += f"\nBackend: {self.backend}"
        if self.tags:
            ret += f"\nTags: {self.tags}"
        ret += f"\nData: {len(self._data)}"
        ret += f"\nAnalysis Results: {n_res}"
        ret += f"\nFigures: {len(self._figures)}"
        return ret
