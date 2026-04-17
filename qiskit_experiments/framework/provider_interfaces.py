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
Definitions of interfaces for classes working with circuit execution

Qiskit Experiments tries to maintain the flexibility to work with multiple
providers of quantum circuit execution, like Qiskit IBM Runtime
and Qiskit Aer. These different circuit execution providers do not
follow exactly the same interface. This module provides definitions of the
subset of the interfaces that Qiskit Experiments needs in order to analyze
experiment results.
"""

from __future__ import annotations

from enum import Enum, IntEnum
from json import JSONEncoder, JSONDecoder
from typing import Protocol, TYPE_CHECKING

from qiskit.result import Result
from qiskit.primitives import PrimitiveResult
from qiskit.providers import Backend, JobStatus

if TYPE_CHECKING:
    from qiskit_experiments.database_service import DbExperimentData, DbAnalysisResultData


class BaseJob(Protocol):
    """Required interface definition of a job class as needed for experiment data"""

    def cancel(self):
        """Cancel the job"""
        raise NotImplementedError

    def job_id(self) -> str:
        """Return the ID string for the job"""
        raise NotImplementedError

    def result(self) -> Result | PrimitiveResult:
        """Return the job result data"""
        raise NotImplementedError

    def status(self) -> JobStatus | str:
        """Return the status of the job"""
        raise NotImplementedError


class ExtendedJob(BaseJob, Protocol):
    """Job interface with methods to support all of experiment data's features"""

    def backend(self) -> Backend:
        """Return the backend associated with a job"""
        raise NotImplementedError

    def error_message(self) -> str | None:
        """Returns the reason the job failed"""
        raise NotImplementedError


Job = BaseJob | ExtendedJob
"""Union type of job interfaces supported by Qiskit Experiments"""


class BaseProvider(Protocol):
    """Interface definition of a provider class as needed for experiment data"""

    def job(self, job_id: str) -> Job:
        """Retrieve a job object using its job ID

        Args:
            job_id: Job ID.

        Returns:
            The retrieved job
        """
        raise NotImplementedError


Provider = BaseProvider
"""Type alias of provider interface supported by Qiskit Experiments"""


class MeasReturnType(str, Enum):
    """Backend return types for Qobj and backend.run jobs"""

    AVERAGE = "avg"
    SINGLE = "single"


class MeasLevel(IntEnum):
    """Measurement level types for legacy Qobj and Sampler jobs"""

    RAW = 0
    KERNELED = 1
    CLASSIFIED = 2


class ExperimentService(Protocol):
    """Interface definition for experiment database service.

    This interface defines the methods needed by ExperimentData to interact
    with an experiment database service, whether local or remote.

    .. note::

        Some of the type signatures of the methods of this protocol could
        change in future versions of Qiskit Experiments without a transition
        period.
    """

    @property
    def options(self) -> dict:
        """Return service options dictionary.

        Returns:
            Dictionary of service options
        """
        raise NotImplementedError

    def create_or_update_experiment(
        self,
        data: DbExperimentData,
        json_encoder: type[JSONEncoder] | None = None,
        create: bool = True,
        max_attempts: int = 3,
        **kwargs,
    ) -> DbExperimentData:
        """Create or update an experiment in the database.

        Args:
            data: Experiment data to save
            json_encoder: Custom JSON encoder
            create: Whether to attempt create first
            max_attempts: Maximum number of attempts
            **kwargs: Additional parameters

        Returns:
            Experiment data of the experiment
        """
        raise NotImplementedError

    def create_or_update_analysis_result(
        self,
        data: DbAnalysisResultData,
        json_encoder: type[JSONEncoder] | None = None,
        create: bool = True,
        max_attempts: int = 3,
    ) -> str:
        """Create or update an analysis result in the database.

        Args:
            data: Analysis result data to save
            json_encoder: Custom JSON encoder
            create: Whether to attempt create first
            max_attempts: Maximum number of attempts

        Returns:
            Analysis result ID
        """
        raise NotImplementedError

    def create_analysis_results(
        self,
        data: list,
        blocking: bool = True,
        max_workers: int = 100,
        json_encoder: type[JSONEncoder] | None = None,
    ):
        """Create multiple analysis results in the database.

        Args:
            data: List of analysis result data to save
            blocking: Whether to wait for completion
            max_workers: Maximum number of worker threads
            json_encoder: Custom JSON encoder

        Returns:
            Status dictionary or handler object
        """
        raise NotImplementedError

    def experiment(
        self,
        experiment_id: str,
        json_decoder: type[JSONDecoder] | None = None,
    ) -> DbExperimentData:
        """Retrieve a single experiment from the database.

        Args:
            experiment_id: Experiment ID
            json_decoder: Custom JSON decoder

        Returns:
            Retrieved experiment data
        """
        raise NotImplementedError

    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment from the database.

        Args:
            experiment_id: Experiment ID to delete
        """
        raise NotImplementedError

    def analysis_result(
        self,
        result_id: str,
        json_decoder: type[JSONDecoder] | None = None,
    ) -> DbAnalysisResultData:
        """Retrieve a single analysis result from the database.

        Args:
            result_id: Analysis result ID
            json_decoder: Custom JSON decoder

        Returns:
            Retrieved analysis result data
        """
        raise NotImplementedError

    def analysis_results(
        self,
        experiment_id: str | None = None,
        limit: int | None = None,
        json_decoder: type[JSONDecoder] | None = None,
        **filters,
    ) -> list:
        """Query analysis results from the database.

        Args:
            experiment_id: Filter by experiment ID
            limit: Maximum number of results
            json_decoder: Custom JSON decoder
            **filters: Additional filter parameters

        Returns:
            List of analysis results
        """
        raise NotImplementedError

    def delete_analysis_result(self, result_id: str) -> None:
        """Delete an analysis result from the database.

        Args:
            result_id: Analysis result ID to delete
        """
        raise NotImplementedError

    def create_or_update_figure(
        self,
        experiment_id: str,
        figure: bytes | str,
        figure_name: str | None = None,
        create: bool = True,
        max_attempts: int = 3,
    ) -> tuple:
        """Create or update a figure in the database.

        Args:
            experiment_id: Experiment ID
            figure: Figure data or file path
            figure_name: Name for the figure
            create: Whether to attempt create first
            max_attempts: Maximum number of attempts

        Returns:
            Tuple of (figure_name, size)
        """
        raise NotImplementedError

    def create_figures(
        self,
        experiment_id: str,
        figure_list: list,
        blocking: bool = True,
        max_workers: int = 100,
    ):
        """Create multiple figures in the database.

        Args:
            experiment_id: Experiment ID
            figure_list: List of (figure, name) tuples
            blocking: Whether to wait for completion
            max_workers: Maximum number of worker threads

        Returns:
            Status dictionary or handler object
        """
        raise NotImplementedError

    def figure(
        self,
        experiment_id: str,
        figure_name: str,
        file_name: str | None = None,
    ) -> bytes | int:
        """Retrieve a figure from the database.

        Args:
            experiment_id: Experiment ID
            figure_name: Name of the figure
            file_name: Optional local file to save to

        Returns:
            Figure bytes if file_name is None, otherwise size written
        """
        raise NotImplementedError

    def delete_figure(self, experiment_id: str, figure_name: str) -> None:
        """Delete a figure from the database.

        Args:
            experiment_id: Experiment ID
            figure_name: Name of the figure to delete
        """
        raise NotImplementedError

    def files(self, experiment_id: str) -> dict:
        """Retrieve the file list for an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Dictionary with file list metadata
        """
        raise NotImplementedError

    def file_upload(
        self,
        experiment_id: str,
        file_name: str,
        file_data: dict | str | bytes,
        json_encoder: type[JSONEncoder] | None = None,
    ) -> None:
        """Upload a file to the database.

        Args:
            experiment_id: Experiment ID
            file_name: Name for the file
            file_data: File data (dict or JSON string or file bytes)
            json_encoder: Custom JSON encoder
        """
        raise NotImplementedError

    def file_delete(
        self,
        experiment_id: str,
        file_name: str,
    ):
        """Delete a file from the database

        Args:
            experiment_id: Experiment ID
            file_name: Name for the file
        """
        raise NotImplementedError

    def file_download(
        self,
        experiment_id: str,
        file_name: str,
        json_decoder: type[JSONDecoder] | None = None,
    ) -> dict:
        """Download a file from the database.

        Args:
            experiment_id: Experiment ID
            file_name: Name of the file
            json_decoder: Custom JSON decoder

        Returns:
            Deserialized file data
        """
        raise NotImplementedError
