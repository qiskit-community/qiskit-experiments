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
import dataclasses
from typing import Dict, Optional, List, Union, Any, Callable, Tuple, TYPE_CHECKING
from datetime import datetime
from concurrent import futures
from threading import Event
from functools import wraps
from collections import deque
import contextlib
import warnings
import copy
import uuid
import enum
import time
import io
import sys
import traceback
import numpy as np
from matplotlib import pyplot
from matplotlib.figure import Figure as MatplotlibFigure
from qiskit.result import Result
from qiskit.providers.jobstatus import JobStatus, JOB_FINAL_STATES
from qiskit.exceptions import QiskitError
from qiskit.providers import Job, Backend, Provider

from qiskit_ibm_experiment import IBMExperimentService
from qiskit_ibm_experiment import ExperimentData as ExperimentDataclass
from qiskit_experiments.framework.json import ExperimentEncoder, ExperimentDecoder
from qiskit_experiments.database_service.utils import (
    qiskit_version,
    plot_to_svg_bytes,
    ThreadSafeOrderedDict,
    ThreadSafeList,
)
from qiskit_experiments.framework.analysis_result import AnalysisResult
from qiskit_experiments.framework import BackendData
from qiskit_experiments.database_service.exceptions import (
    ExperimentDataError,
    ExperimentEntryNotFound,
    ExperimentEntryExists,
)

if TYPE_CHECKING:
    # There is a cyclical dependency here, but the name needs to exist for
    # Sphinx on Python 3.9+ to link type hints correctly.  The gating on
    # `TYPE_CHECKING` means that the import will never be resolved by an actual
    # interpreter, only static analysis.
    from . import BaseExperiment

LOG = logging.getLogger(__name__)


def do_auto_save(func: Callable):
    """Decorate the input function to auto save data."""

    @wraps(func)
    def _wrapped(self, *args, **kwargs):
        return_val = func(self, *args, **kwargs)
        if self.auto_save:
            self.save_metadata()
        return return_val

    return _wrapped


class FigureData:
    """Wrapper class for figures and figure metadata. The raw figure can be accessed with
    the ``figure`` attribute."""

    def __init__(self, figure, name=None, metadata=None):
        """Creates a new figure data object.

        Args:
            figure: the raw figure itself. Can be SVG or matplotlib.Figure.
            name: Optional, the name of the figure.
            metadata: Optional, any metadata to be stored with the figure."""
        self.figure = figure
        self._name = name
        self.metadata = metadata or {}

    # name is read only
    @property
    def name(self) -> str:
        """The name of the figure"""
        return self._name

    @property
    def metadata(self) -> dict:
        """The metadata dictionary stored with the figure"""
        return self._metadata

    @metadata.setter
    def metadata(self, new_metadata: dict):
        """Set the metadata to new value; must be a dictionary"""
        if not isinstance(new_metadata, dict):
            raise ValueError("figure metadata must be a dictionary")
        self._metadata = new_metadata

    def copy(self, new_name: Optional[str] = None):
        """Creates a copy of the figure data"""
        name = new_name or self.name
        return FigureData(figure=self.figure, name=name, metadata=copy.deepcopy(self.metadata))

    def __json_encode__(self) -> Dict[str, Any]:
        """Return the json representation of the figure data"""
        return {"figure": self.figure, "name": self.name, "metadata": self.metadata}

    @classmethod
    def __json_decode__(cls, args: Dict[str, Any]) -> "FigureData":
        """Initialize a figure data from the json representation"""
        return cls(**args)

    def _repr_png_(self):
        if isinstance(self.figure, MatplotlibFigure):
            b = io.BytesIO()
            self.figure.savefig(b, format="png", bbox_inches="tight")
            png = b.getvalue()
            return png
        else:
            return None

    def _repr_svg_(self):
        if isinstance(self.figure, str):
            return self.figure
        if isinstance(self.figure, bytes):
            return str(self.figure)
        return None


class ExperimentData:
    """Qiskit Experiments Data container class.

    This class handles the following:
    1. Storing the data related to an experiment - the experiment's metadata,
       the analysis results and the figures
    2. Managing jobs and adding data from jobs automatically
    3. Saving/Loading data from the result database

    The field `db_data` is a dataclass (`ExperimentDataclass`) containing
    all the data that can be stored in the database and loaded from it, and
    as such is subject to strict conventions.

    Other data fields can be added and used freely, but they won't be saved
    to the database.
    """

    _metadata_version = 1
    _job_executor = futures.ThreadPoolExecutor()

    _json_encoder = ExperimentEncoder
    _json_decoder = ExperimentDecoder

    _metadata_filename = "metadata.json"

    def __init__(
        self,
        experiment: Optional["BaseExperiment"] = None,
        backend: Optional[Backend] = None,
        service: Optional[IBMExperimentService] = None,
        parent_id: Optional[str] = None,
        job_ids: Optional[List[str]] = None,
        child_data: Optional[List[ExperimentData]] = None,
        verbose: Optional[bool] = True,
        db_data: Optional[ExperimentDataclass] = None,
        **kwargs,
    ):
        """Initialize experiment data.

        Args:
            experiment: Optional, experiment object that generated the data.
            backend: Optional, Backend the experiment runs on; overrides the
            backend in the experiment object
            service: The service that stores the experiment results to the database
            parent_id: Optional, ID of the parent experiment data
                in the setting of a composite experiment
            job_ids: Optional, IDs of jobs submitted for the experiment.
            child_data: Optional, list of child experiment data.
            verbose: Optional, whether to print messages
            db_data: Optional, a prepared ExperimentDataclass of the experiment info;
            overrides other db parameters.
        """
        if experiment is not None:
            backend = backend or experiment.backend
            experiment_type = experiment.experiment_type
        else:
            experiment_type = None
        if job_ids is None:
            job_ids = []

        self._experiment = experiment

        # data stored in the database
        metadata = dict()
        if experiment is not None:
            metadata = copy.deepcopy(experiment._metadata())
        source = metadata.pop(
            "_source",
            {
                "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
                "metadata_version": self.__class__._metadata_version,
                "qiskit_version": qiskit_version(),
            },
        )
        metadata["_source"] = source
        experiment_id = kwargs.get("experiment_id", str(uuid.uuid4()))
        if db_data is None:
            self._db_data = ExperimentDataclass(
                experiment_id=experiment_id,
                experiment_type=experiment_type,
                parent_id=parent_id,
                job_ids=job_ids,
                metadata=metadata,
            )
        else:
            self._db_data = db_data

        for key, value in kwargs.items():
            if hasattr(self._db_data, key):
                setattr(self._db_data, key, value)
            else:
                LOG.warning("Key '%s' not stored in the database", key)

        # general data related
        self._backend = None
        if backend is not None:
            self._set_backend(backend, recursive=False)
        self._service = service
        if self._service is None and self.backend is not None:
            self._service = self.get_service_from_backend(self.backend)
        self._auto_save = False
        self._created_in_db = False
        self._extra_data = kwargs
        self.verbose = verbose

        # job handling related
        self._jobs = ThreadSafeOrderedDict(job_ids)
        self._job_futures = ThreadSafeOrderedDict()
        self._analysis_callbacks = ThreadSafeOrderedDict()
        self._analysis_futures = ThreadSafeOrderedDict()
        # Set 2 workers for analysis executor so there can be 1 actively running
        # future and one waiting "running" future. This is to allow the second
        # future to be cancelled without waiting for the actively running future
        # to finish first.
        self._analysis_executor = futures.ThreadPoolExecutor(max_workers=2)
        self._monitor_executor = futures.ThreadPoolExecutor()

        # data storage
        self._result_data = ThreadSafeList()
        self._figures = ThreadSafeOrderedDict(self._db_data.figure_names)
        self._analysis_results = ThreadSafeOrderedDict()

        self._deleted_figures = deque()
        self._deleted_analysis_results = deque()

        # Child related
        # Add component data and set parent ID to current container
        self._child_data = ThreadSafeOrderedDict()
        if child_data is not None:
            self._set_child_data(child_data)

    # Getters/setters for experiment metadata

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

    @property
    def tags(self) -> List[str]:
        """Return tags assigned to this experiment data.

        Returns:
            A list of tags assigned to this experiment data.

        """
        return self._db_data.tags

    @tags.setter
    def tags(self, new_tags: List[str]) -> None:
        """Set tags for this experiment."""
        if not isinstance(new_tags, list):
            raise ExperimentDataError(f"The `tags` field of {type(self).__name__} must be a list.")
        self._db_data.tags = np.unique(new_tags).tolist()
        if self.auto_save:
            self.save_metadata()

    @property
    def metadata(self) -> Dict:
        """Return experiment metadata.

        Returns:
            Experiment metadata.
        """
        return self._db_data.metadata

    @property
    def creation_datetime(self) -> "datetime":
        """Return the creation datetime of this experiment data.

        Returns:
            The creation datetime of this experiment data.

        """
        return self._db_data.creation_datetime

    @property
    def start_datetime(self) -> "datetime":
        """Return the start datetime of this experiment data.

        Returns:
            The start datetime of this experiment data.

        """
        return self._db_data.end_datetime

    @property
    def updated_datetime(self) -> "datetime":
        """Return the update datetime of this experiment data.

        Returns:
            The update datetime of this experiment data.

        """
        return self._db_data.updated_datetime

    @property
    def end_datetime(self) -> "datetime":
        """Return the end datetime of this experiment data.

        Returns:
            The end datetime of this experiment data.

        """
        return self._db_data.end_datetime

    @property
    def hub(self) -> str:
        """Return the hub of this experiment data.

        Returns:
            The hub of this experiment data.

        """
        return self._db_data.hub

    @property
    def group(self) -> str:
        """Return the group of this experiment data.

        Returns:
            The group of this experiment data.

        """
        return self._db_data.group

    @property
    def project(self) -> str:
        """Return the project of this experiment data.

        Returns:
            The project of this experiment data.

        """
        return self._db_data.project

    @property
    def _provider(self) -> Optional[Provider]:
        """Return the provider.

        Returns:
            Provider used for the experiment, or ``None`` if unknown.
        """
        if self._backend is None:
            return None
        return self._backend.provider()

    @property
    def experiment_id(self) -> str:
        """Return experiment ID

        Returns:
            Experiment ID.
        """

        return self._db_data.experiment_id

    @property
    def experiment_type(self) -> str:
        """Return experiment type

        Returns:
            Experiment type.
        """

        return self._db_data.experiment_type

    @experiment_type.setter
    def experiment_type(self, new_type: str) -> None:
        """Sets the parent id"""
        self._db_data.experiment_type = new_type

    @property
    def parent_id(self) -> str:
        """Return parent experiment ID

        Returns:
            Parent ID.
        """
        return self._db_data.parent_id

    @parent_id.setter
    def parent_id(self, new_id: str) -> None:
        """Sets the parent id"""
        self._db_data.parent_id = new_id

    @property
    def job_ids(self) -> List[str]:
        """Return experiment job IDs.

        Returns: IDs of jobs submitted for this experiment.
        """
        return self._db_data.job_ids

    @property
    def figure_names(self) -> List[str]:
        """Return names of the figures associated with this experiment.

        Returns:
            Names of figures associated with this experiment.
        """
        return self._figures.keys()

    @property
    def share_level(self) -> str:
        """Return the share level for this experiment

        Returns:
            Experiment share level.
        """
        return self._db_data.share_level

    @share_level.setter
    def share_level(self, new_level: str) -> None:
        """Set the experiment share level,
           to this experiment itself and its descendants.

        Args:
            new_level: New experiment share level. Valid share levels are provider-
                specified. For example, IBM Quantum experiment service allows
                "public", "hub", "group", "project", and "private".
        """
        self._db_data.share_level = new_level
        for data in self._child_data.values():
            original_auto_save = data.auto_save
            data.auto_save = False
            data.share_level = new_level
            data.auto_save = original_auto_save
        if self.auto_save:
            self.save_metadata()

    @property
    def notes(self) -> str:
        """Return experiment notes.

        Returns:
            Experiment notes.
        """
        return self._db_data.notes

    @notes.setter
    def notes(self, new_notes: str) -> None:
        """Update experiment notes.

        Args:
            new_notes: New experiment notes.
        """
        self._db_data.notes = new_notes
        if self.auto_save:
            self.save_metadata()

    @property
    def backend_name(self) -> str:
        """Return the backend's name"""
        return self._db_data.backend

    @property
    def backend(self) -> Backend:
        """Return backend.

        Returns:
            Backend.
        """
        return self._backend

    @backend.setter
    def backend(self, new_backend: Backend) -> None:
        """Update backend.

        Args:
            new_backend: New backend.
        """
        self._set_backend(new_backend)
        if self.auto_save:
            self.save_metadata()

    def _set_backend(self, new_backend: Backend, recursive: bool = True) -> None:
        """Set backend.
        Args:
            new_backend: New backend.
            recursive: should set the backend for children as well
        """
        # defined independently from the setter to enable setting without autosave

        self._backend = new_backend
        self._backend_data = BackendData(new_backend)
        self._db_data.backend = self._backend_data.name
        if self._db_data.backend is None:
            self._db_data.backend = str(new_backend)
        provider = self._backend_data.provider
        if provider is not None:
            self._set_hgp_from_provider(provider)
        if recursive:
            for data in self.child_data():
                data._set_backend(new_backend)

    def _set_hgp_from_provider(self, provider):
        try:
            creds = provider.credentials
            hub = self._db_data.hub or creds.hub
            group = self._db_data.group or creds.group
            project = self._db_data.project or creds.project
            self._db_data.hub = hub
            self._db_data.group = group
            self._db_data.project = project
        except AttributeError:
            return

    def _clear_results(self):
        """Delete all currently stored analysis results and figures"""
        # Schedule existing analysis results for deletion next save call
        for key in self._analysis_results.keys():
            self._deleted_analysis_results.append(key)
        self._analysis_results = ThreadSafeOrderedDict()
        # Schedule existing figures for deletion next save call
        for key in self._figures.keys():
            self._deleted_figures.append(key)
        self._figures = ThreadSafeOrderedDict()

    @property
    def service(self) -> Optional[IBMExperimentService]:
        """Return the database service.

        Returns:
            Service that can be used to access this experiment in a database.
        """
        return self._service

    @service.setter
    def service(self, service: IBMExperimentService) -> None:
        """Set the service to be used for storing experiment data

        Args:
            service: Service to be used.

        Raises:
            ExperimentDataError: If an experiment service is already being used.
        """
        self._set_service(service)

    @property
    def auto_save(self) -> bool:
        """Return current auto-save option.

        Returns:
            Whether changes will be automatically saved.
        """
        return self._auto_save

    @auto_save.setter
    def auto_save(self, save_val: bool) -> None:
        """Set auto save preference.

        Args:
            save_val: Whether to do auto-save.
        """
        if save_val is True and not self._auto_save:
            self.save()
        self._auto_save = save_val
        for res in self._analysis_results.values():
            # Setting private variable directly to avoid duplicate save. This
            # can be removed when we start tracking changes.
            res._auto_save = save_val
        for data in self.child_data():
            data.auto_save = save_val

    @property
    def source(self) -> Dict:
        """Return the class name and version."""
        return self._db_data.metadata["_source"]

    # Data addition and deletion

    def add_data(
        self,
        data: Union[Result, List[Result], Job, List[Job], Dict, List[Dict]],
    ) -> None:
        """Add experiment data.

        Args:
            data: Experiment data to add. Several types are accepted for convenience
                * Result: Add data from this ``Result`` object.
                * List[Result]: Add data from the ``Result`` objects.
                * Dict: Add this data.
                * List[Dict]: Add this list of data.
                * Job: (Deprecated) Add data from the job result.
                * List[Job]: (Deprecated) Add data from the job results.

        Raises:
            TypeError: If the input data type is invalid.
        """
        if any(not future.done() for future in self._analysis_futures.values()):
            LOG.warning(
                "Not all analysis has finished running. Adding new data may "
                "create unexpected analysis results."
            )
        if not isinstance(data, list):
            data = [data]

        # Directly add non-job data
        with self._result_data.lock:
            for datum in data:
                if isinstance(datum, dict):
                    self._result_data.append(datum)
                elif isinstance(datum, Result):
                    self._add_result_data(datum)
                else:
                    raise TypeError(f"Invalid data type {type(datum)}.")

    def add_jobs(
        self,
        jobs: Union[Job, List[Job]],
        timeout: Optional[float] = None,
    ) -> None:
        """Add experiment data.

        Args:
            jobs: The Job or list of Jobs to add result data from.
            timeout: Optional, time in seconds to wait for all jobs to finish
                     before cancelling them.

        Raises:
            TypeError: If the input data type is invalid.

        .. note::
            If a timeout is specified the :meth:`cancel_jobs` method will be
            called after timing out to attempt to cancel any unfinished jobs.

            If you want to wait for jobs without cancelling, use the timeout
            kwarg of :meth:`block_for_results` instead.
        """
        if any(not future.done() for future in self._analysis_futures.values()):
            LOG.warning(
                "Not all analysis has finished running. Adding new jobs may "
                "create unexpected analysis results."
            )
        if isinstance(jobs, Job):
            jobs = [jobs]

        # Add futures for extracting finished job data
        timeout_ids = []
        for job in jobs:
            if self.backend is not None:
                backend_name = BackendData(self.backend).name
                job_backend_name = BackendData(job.backend()).name
                if self.backend and backend_name != job_backend_name:
                    LOG.warning(
                        "Adding a job from a backend (%s) that is different "
                        "than the current backend (%s). "
                        "The new backend will be used, but "
                        "service is not changed if one already exists.",
                        job.backend(),
                        self.backend,
                    )
            self.backend = job.backend()

            jid = job.job_id()
            if jid in self._jobs:
                LOG.warning(
                    "Skipping duplicate job, a job with this ID already exists [Job ID: %s]", jid
                )
            else:
                self.job_ids.append(jid)
                self._jobs[jid] = job
                if jid in self._job_futures:
                    LOG.warning("Job future has already been submitted [Job ID: %s]", jid)
                else:
                    self._add_job_future(job)
                    if timeout is not None:
                        timeout_ids.append(jid)

        # Add future for cancelling jobs that timeout
        if timeout_ids:
            self._job_executor.submit(self._timeout_running_jobs, timeout_ids, timeout)

        if self.auto_save:
            self.save_metadata()

    def _timeout_running_jobs(self, job_ids, timeout):
        """Function for cancelling jobs after timeout length.

        This function should be submitted to an executor to run as a future.

        Args:
            job_ids: the IDs of jobs to wait for.
            timeout: The total time to wait for all jobs before cancelling.
        """
        futs = [self._job_futures[jid] for jid in job_ids]
        waited = futures.wait(futs, timeout=timeout)

        # Try to cancel timed-out jobs
        if waited.not_done:
            LOG.debug("Cancelling running jobs that exceeded add_jobs timeout.")
            done_ids = {fut.result()[0] for fut in waited.done}
            notdone_ids = [jid for jid in job_ids if jid not in done_ids]
            self.cancel_jobs(notdone_ids)

    def _add_job_future(self, job):
        """Submit new _add_job_data job to executor"""
        jid = job.job_id()
        if jid in self._job_futures:
            LOG.warning("Job future has already been submitted [Job ID: %s]", jid)
        else:
            self._job_futures[jid] = self._job_executor.submit(self._add_job_data, job)

    def _add_job_data(
        self,
        job: Job,
    ) -> Tuple[str, bool]:
        """Wait for a job to finish and add job result data.

        Args:
            job: the Job to wait for and add data from.

        Returns:
            A tuple (str, bool) of the job id and bool of if the job data was added.

        Raises:
            Exception: If an error occured when adding job data.
        """
        jid = job.job_id()
        try:
            job_result = job.result()
            self._add_result_data(job_result)
            LOG.debug("Job data added [Job ID: %s]", jid)
            return jid, True
        except Exception as ex:  # pylint: disable=broad-except
            # Handle cancelled jobs
            status = job.status()
            if status == JobStatus.CANCELLED:
                LOG.warning("Job was cancelled before completion [Job ID: %s]", jid)
                return jid, False
            if status == JobStatus.ERROR:
                LOG.error(
                    "Job data not added for errorred job [Job ID: %s]" "\nError message: %s",
                    jid,
                    job.error_message(),
                )
                return jid, False
            LOG.warning("Adding data from job failed [Job ID: %s]", job.job_id())
            raise ex

    def add_analysis_callback(self, callback: Callable, **kwargs: Any):
        """Add analysis callback for running after experiment data jobs are finished.

        This method adds the `callback` function to a queue to be run
        asynchronously after completion of any running jobs, or immediately
        if no running jobs. If this method is called multiple times the
        callback functions will be executed in the order they were
        added.

        Args:
            callback: Callback function invoked when job finishes successfully.
                      The callback function will be called as
                      ``callback(expdata, **kwargs)`` where `expdata` is this
                      ``DbExperimentData`` object, and `kwargs` are any additional
                      keywork arguments passed to this method.
            **kwargs: Keyword arguments to be passed to the callback function.
        """
        with self._job_futures.lock and self._analysis_futures.lock:
            # Create callback dataclass
            cid = uuid.uuid4().hex
            self._analysis_callbacks[cid] = AnalysisCallback(
                name=callback.__name__,
                callback_id=cid,
            )

            # Futures to wait for
            futs = self._job_futures.values() + self._analysis_futures.values()
            wait_future = self._monitor_executor.submit(
                self._wait_for_futures, futs, name="jobs and analysis"
            )

            # Create a future to monitor event for calls to cancel_analysis
            def _monitor_cancel():
                self._analysis_callbacks[cid].event.wait()
                return False

            cancel_future = self._monitor_executor.submit(_monitor_cancel)

            # Add run analysis future
            self._analysis_futures[cid] = self._analysis_executor.submit(
                self._run_analysis_callback, cid, wait_future, cancel_future, callback, **kwargs
            )

    def _run_analysis_callback(
        self,
        callback_id: str,
        wait_future: futures.Future,
        cancel_future: futures.Future,
        callback: Callable,
        **kwargs,
    ):
        """Run an analysis callback after specified futures have finished."""
        if callback_id not in self._analysis_callbacks:
            raise ValueError(f"No analysis callback with id {callback_id}")

        # Monitor jobs and cancellation event to see if callback should be run
        # or cancelled
        # Future which returns if either all jobs finish, or cancel event is set
        waited = futures.wait([wait_future, cancel_future], return_when="FIRST_COMPLETED")
        cancel = not all(fut.result() for fut in waited.done)

        # Ensure monitor event is set so monitor future can terminate
        self._analysis_callbacks[callback_id].event.set()

        # If not ready cancel the callback before running
        if cancel:
            self._analysis_callbacks[callback_id].status = AnalysisStatus.CANCELLED
            LOG.info(
                "Cancelled analysis callback [Experiment ID: %s][Analysis Callback ID: %s]",
                self.experiment_id,
                callback_id,
            )
            return callback_id, False

        # Run callback function
        self._analysis_callbacks[callback_id].status = AnalysisStatus.RUNNING
        try:
            LOG.debug(
                "Running analysis callback '%s' [Experiment ID: %s][Analysis Callback ID: %s]",
                self._analysis_callbacks[callback_id].name,
                self.experiment_id,
                callback_id,
            )
            callback(self, **kwargs)
            self._analysis_callbacks[callback_id].status = AnalysisStatus.DONE
            LOG.debug(
                "Analysis callback finished [Experiment ID: %s][Analysis Callback ID: %s]",
                self.experiment_id,
                callback_id,
            )
            return callback_id, True
        except Exception as ex:  # pylint: disable=broad-except
            self._analysis_callbacks[callback_id].status = AnalysisStatus.ERROR
            tb_text = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
            error_msg = (
                f"Analysis callback failed [Experiment ID: {self.experiment_id}]"
                f"[Analysis Callback ID: {callback_id}]:\n{tb_text}"
            )
            self._analysis_callbacks[callback_id].error_msg = error_msg
            LOG.warning(error_msg)
            return callback_id, False

    def _add_result_data(self, result: Result) -> None:
        """Add data from a Result object

        Args:
            result: Result object containing data to be added.
        """
        if result.job_id not in self._jobs:
            self._jobs[result.job_id] = None
            self.job_ids.append(result.job_id)
        with self._result_data.lock:
            # Lock data while adding all result data
            for i, _ in enumerate(result.results):
                data = result.data(i)
                data["job_id"] = result.job_id
                if "counts" in data:
                    # Format to Counts object rather than hex dict
                    data["counts"] = result.get_counts(i)
                expr_result = result.results[i]
                if hasattr(expr_result, "header") and hasattr(expr_result.header, "metadata"):
                    data["metadata"] = expr_result.header.metadata
                data["shots"] = expr_result.shots
                data["meas_level"] = expr_result.meas_level
                if hasattr(expr_result, "meas_return"):
                    data["meas_return"] = expr_result.meas_return
                self._result_data.append(data)

    def _retrieve_data(self):
        """Retrieve job data if missing experiment data."""
        if self._result_data or not self._backend:
            return
        # Get job results if missing experiment data.
        retrieved_jobs = {}
        for jid, job in self._jobs.items():
            if job is None:
                try:
                    LOG.debug("Retrieving job from backend %s [Job ID: %s]", self._backend, jid)
                    job = self._backend.retrieve_job(jid)
                    retrieved_jobs[jid] = job
                except Exception:  # pylint: disable=broad-except
                    LOG.warning(
                        "Unable to retrieve data from job on backend %s [Job ID: %s]",
                        self._backend,
                        jid,
                    )
        # Add retrieved job objects to stored jobs and extract data
        for jid, job in retrieved_jobs.items():
            self._jobs[jid] = job
            if job.status() in JOB_FINAL_STATES:
                # Add job results synchronously
                self._add_job_data(job)
            else:
                # Add job results asynchronously
                self._add_job_future(job)

    def data(
        self,
        index: Optional[Union[int, slice, str]] = None,
    ) -> Union[Dict, List[Dict]]:
        """Return the experiment data at the specified index.

        Args:
            index: Index of the data to be returned.
                Several types are accepted for convenience:

                    * None: Return all experiment data.
                    * int: Specific index of the data.
                    * slice: A list slice of data indexes.
                    * str: ID of the job that produced the data.

        Returns:
            Experiment data.

        Raises:
            TypeError: If the input `index` has an invalid type.
        """
        self._retrieve_data()
        if index is None:
            return self._result_data.copy()
        if isinstance(index, (int, slice)):
            return self._result_data[index]
        if isinstance(index, str):
            return [data for data in self._result_data if data.get("job_id") == index]
        raise TypeError(f"Invalid index type {type(index)}.")

    @do_auto_save
    def add_figures(
        self,
        figures,
        figure_names=None,
        overwrite=False,
        save_figure=None,
    ) -> Union[str, List[str]]:
        """Add the experiment figure.

        Args:
            figures (str or bytes or pyplot.Figure or list): Paths of the figure
                files or figure data.
            figure_names (str or list): Names of the figures. If ``None``, use the figure file
                names, if given, or a generated name. If `figures` is a list, then
                `figure_names` must also be a list of the same length or ``None``.
            overwrite (bool): Whether to overwrite the figure if one already exists with
                the same name.
            save_figure (bool): Whether to save the figure in the database. If ``None``,
                the ``auto-save`` attribute is used.

        Returns:
            str or list:
                Figure names.

        Raises:
            ExperimentEntryExists: If the figure with the same name already exists,
                and `overwrite=True` is not specified.
            ValueError: If an input parameter has an invalid value.
        """
        if figure_names is not None and not isinstance(figure_names, list):
            figure_names = [figure_names]
        if not isinstance(figures, list):
            figures = [figures]
        if figure_names is not None and len(figures) != len(figure_names):
            raise ValueError(
                "The parameter figure_names must be None or a list of "
                "the same size as the parameter figures."
            )

        added_figs = []
        for idx, figure in enumerate(figures):
            if figure_names is None:
                if isinstance(figure, str):
                    fig_name = figure
                else:
                    fig_name = (
                        f"{self.experiment_type}_"
                        f"Fig-{len(self._figures)}_"
                        f"Exp-{self.experiment_id[:8]}.svg"
                    )
            else:
                fig_name = figure_names[idx]

            if not fig_name.endswith(".svg"):
                LOG.info("File name %s does not have an SVG extension. A '.svg' is added.")
                fig_name += ".svg"

            existing_figure = fig_name in self._figures
            if existing_figure and not overwrite:
                raise ExperimentEntryExists(
                    f"A figure with the name {fig_name} for this experiment "
                    f"already exists. Specify overwrite=True if you "
                    f"want to overwrite it."
                )
            # figure_data = None
            if isinstance(figure, str):
                with open(figure, "rb") as file:
                    figure = file.read()

            # check whether the figure is already wrapped, meaning it came from a sub-experiment
            if isinstance(figure, FigureData):
                figure_data = figure.copy(new_name=fig_name)

            else:
                figure_metadata = {"qubits": self.metadata.get("physical_qubits")}
                figure_data = FigureData(figure=figure, name=fig_name, metadata=figure_metadata)

            self._figures[fig_name] = figure_data

            save = save_figure if save_figure is not None else self.auto_save
            if save and self._service:
                if isinstance(figure, pyplot.Figure):
                    figure = plot_to_svg_bytes(figure)
                self._service.create_or_update_figure(
                    experiment_id=self.experiment_id,
                    figure=figure,
                    figure_name=fig_name,
                    create=not existing_figure,
                )
            added_figs.append(fig_name)

        return added_figs if len(added_figs) != 1 else added_figs[0]

    @do_auto_save
    def delete_figure(
        self,
        figure_key: Union[str, int],
    ) -> str:
        """Add the experiment figure.

        Args:
            figure_key: Name or index of the figure.

        Returns:
            Figure name.

        Raises:
            ExperimentEntryNotFound: If the figure is not found.
        """
        if isinstance(figure_key, int):
            figure_key = self._figures.keys()[figure_key]
        elif figure_key not in self._figures:
            raise ExperimentEntryNotFound(f"Figure {figure_key} not found.")

        del self._figures[figure_key]
        self._deleted_figures.append(figure_key)

        if self._service and self.auto_save:
            with service_exception_to_warning():
                self.service.delete_figure(experiment_id=self.experiment_id, figure_name=figure_key)
            self._deleted_figures.remove(figure_key)

        return figure_key

    def figure(
        self,
        figure_key: Union[str, int],
        file_name: Optional[str] = None,
    ) -> Union[int, bytes]:
        """Retrieve the specified experiment figure.

        Args:
            figure_key: Name or index of the figure.
            file_name: Name of the local file to save the figure to. If ``None``,
                the content of the figure is returned instead.

        Returns:
            The size of the figure if `file_name` is specified. Otherwise the
            content of the figure in bytes.

        Raises:
            ExperimentEntryNotFound: If the figure cannot be found.
        """
        if isinstance(figure_key, int):
            if figure_key < 0 or figure_key >= len(self._figures.keys()):
                raise ExperimentEntryNotFound(f"Figure {figure_key} not found.")
            figure_key = self._figures.keys()[figure_key]

        figure_data = self._figures.get(figure_key, None)
        if figure_data is None and self.service:
            figure = self.service.figure(experiment_id=self.experiment_id, figure_name=figure_key)
            figure_data = FigureData(figure=figure, name=figure_key)
            self._figures[figure_key] = figure_data

        if figure_data is None:
            raise ExperimentEntryNotFound(f"Figure {figure_key} not found.")

        if file_name:
            with open(file_name, "wb") as output:
                num_bytes = output.write(figure_data.figure)
                return num_bytes
        return figure_data

    @do_auto_save
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

        for result in results:
            self._analysis_results[result.result_id] = result

            with contextlib.suppress(ExperimentDataError):
                result.service = self.service
                result.auto_save = self.auto_save

            if self.auto_save and self._service:
                result.save()

    @do_auto_save
    def delete_analysis_result(
        self,
        result_key: Union[int, str],
    ) -> str:
        """Delete the analysis result.

        Args:
            result_key: ID or index of the analysis result to be deleted.

        Returns:
            Analysis result ID.

        Raises:
            ExperimentEntryNotFound: If analysis result not found.
        """

        if isinstance(result_key, int):
            result_key = self._analysis_results.keys()[result_key]
        else:
            # Retrieve from DB if needed.
            result_key = self.analysis_results(result_key, block=False).result_id

        del self._analysis_results[result_key]
        self._deleted_analysis_results.append(result_key)

        if self._service and self.auto_save:
            with service_exception_to_warning():
                self.service.delete_analysis_result(result_id=result_key)
            self._deleted_analysis_results.remove(result_key)

        return result_key

    def _retrieve_analysis_results(self, refresh: bool = False):
        """Retrieve service analysis results.

        Args:
            refresh: Retrieve the latest analysis results from the server, if
                an experiment service is available.
        """
        # Get job results if missing experiment data.
        if self.service and (not self._analysis_results or refresh):
            retrieved_results = self.service.analysis_results(
                experiment_id=self.experiment_id, limit=None, json_decoder=self._json_decoder
            )
            for result in retrieved_results:
                result_id = result.result_id

                self._analysis_results[result_id] = AnalysisResult(service=self.service)
                self._analysis_results[result_id].set_data(result)
                self._analysis_results[result_id]._created_in_db = True

    def analysis_results(
        self,
        index: Optional[Union[int, slice, str]] = None,
        refresh: bool = False,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> Union[AnalysisResult, List[AnalysisResult]]:
        """Return analysis results associated with this experiment.

        Args:
            index: Index of the analysis result to be returned.
                Several types are accepted for convenience:

                    * None: Return all analysis results.
                    * int: Specific index of the analysis results.
                    * slice: A list slice of indexes.
                    * str: ID or name of the analysis result.
            refresh: Retrieve the latest analysis results from the server, if
                an experiment service is available.
            block: If True block for any analysis callbacks to finish running.
            timeout: max time in seconds to wait for analysis callbacks to finish running.

        Returns:
            Analysis results for this experiment.

        Raises:
            TypeError: If the input `index` has an invalid type.
            ExperimentEntryNotFound: If the entry cannot be found.
        """
        if block:
            self._wait_for_futures(
                self._analysis_futures.values(), name="analysis", timeout=timeout
            )
        self._retrieve_analysis_results(refresh=refresh)
        if index is None:
            return self._analysis_results.values()

        def _make_not_found_message(index: Union[int, slice, str]) -> str:
            """Helper to make error message for index not found"""
            msg = [f"Analysis result {index} not found."]
            errors = self.errors()
            if errors:
                msg.append(f"Errors: {errors}")
            return "\n".join(msg)

        if isinstance(index, int):
            if index >= len(self._analysis_results.values()):
                raise ExperimentEntryNotFound(_make_not_found_message(index))
            return self._analysis_results.values()[index]
        if isinstance(index, slice):
            results = self._analysis_results.values()[index]
            if not results:
                raise ExperimentEntryNotFound(_make_not_found_message(index))
            return results
        if isinstance(index, str):
            # Check by result ID
            if index in self._analysis_results:
                return self._analysis_results[index]
            # Check by name
            filtered = [
                result for result in self._analysis_results.values() if result.name == index
            ]
            if not filtered:
                raise ExperimentEntryNotFound(_make_not_found_message(index))
            if len(filtered) == 1:
                return filtered[0]
            else:
                return filtered

        raise TypeError(f"Invalid index type {type(index)}.")

    # Save and load from the database

    def save_metadata(self) -> None:
        """Save this experiments metadata to a database service.

        .. note::
            This method does not save analysis results nor figures.
            Use :meth:`save` for general saving of all experiment data.

            See :meth:`qiskit.providers.experiment.IBMExperimentService.create_experiment`
            for fields that are saved.
        """
        self._save_experiment_metadata()
        for data in self.child_data():
            data.save_metadata()

    def _save_experiment_metadata(self, suppress_errors: bool = True) -> None:
        """Save this experiments metadata to a database service.
        Args:
            suppress_errors: should the method catch exceptions (true) or
            pass them on, potentially aborting the experiemnt (false)
        Raises:
            QiskitError: If the save to the database failed
        .. note::
            This method does not save analysis results nor figures.
            Use :meth:`save` for general saving of all experiment data.

            See :meth:`qiskit.providers.experiment.IBMExperimentService.create_experiment`
            for fields that are saved.
        """
        if not self._service:
            LOG.warning(
                "Experiment cannot be saved because no experiment service is available. "
                "An experiment service is available, for example, "
                "when using an IBM Quantum backend."
            )
            return
        try:
            handle_metadata_separately = self._metadata_too_large()
            if handle_metadata_separately:
                metadata = self._db_data.metadata
                self._db_data.metadata = {}

            self.service.create_or_update_experiment(
                self._db_data, json_encoder=self._json_encoder, create=not self._created_in_db
            )
            self._created_in_db = True

            if handle_metadata_separately:
                self.service.file_upload(
                    self._db_data.experiment_id, self._metadata_filename, metadata
                )
                self._db_data.metadata = metadata

        except Exception as ex:  # pylint: disable=broad-except
            # Don't automatically fail the experiment just because its data cannot be saved.
            LOG.error("Unable to save the experiment data: %s", traceback.format_exc())
            if not suppress_errors:
                raise QiskitError(f"Experiment data save failed\nError Message:\n{str(ex)}") from ex

    def _metadata_too_large(self):
        """Determines whether the metadata should be stored in a separate file"""
        # currently the entire POST JSON request body is limited by default to 100kb
        return sys.getsizeof(self.metadata) > 10000

    def save(self, suppress_errors: bool = True) -> None:
        """Save the experiment data to a database service.

        Args:
            suppress_errors: should the method catch exceptions (true) or
            pass them on, potentially aborting the experiemnt (false)
        .. note::
            This saves the experiment metadata, all analysis results, and all
            figures. Depending on the number of figures and analysis results this
            operation could take a while.

            To only update a previously saved experiments metadata (eg for
            additional tags or notes) use :meth:`save_metadata`.
        """
        # TODO - track changes
        if not self._service:
            LOG.warning(
                "Experiment cannot be saved because no experiment service is available. "
                "An experiment service is available, for example, "
                "when using an IBM Quantum backend."
            )
            return

        self._save_experiment_metadata(suppress_errors=suppress_errors)
        if not self._created_in_db:
            LOG.warning("Could not save experiment metadata to DB, aborting experiment save")
            return

        for result in self._analysis_results.values():
            result.save(suppress_errors=suppress_errors)

        for result in self._deleted_analysis_results.copy():
            with service_exception_to_warning():
                self._service.delete_analysis_result(result_id=result)
            self._deleted_analysis_results.remove(result)

        with self._figures.lock:
            for name, figure in self._figures.items():
                if figure is None:
                    continue
                # currently only the figure and its name are stored in the database
                if isinstance(figure, FigureData):
                    figure = figure.figure
                    LOG.debug("Figure metadata is currently not saved to the database")
                if isinstance(figure, pyplot.Figure):
                    figure = plot_to_svg_bytes(figure)
                self._service.create_or_update_figure(
                    experiment_id=self.experiment_id, figure=figure, figure_name=name
                )

        for name in self._deleted_figures.copy():
            with service_exception_to_warning():
                self._service.delete_figure(experiment_id=self.experiment_id, figure_name=name)
            self._deleted_figures.remove(name)

        if not self.service.local and self.verbose:
            print(
                "You can view the experiment online at "
                f"https://quantum-computing.ibm.com/experiments/{self.experiment_id}"
            )
        # handle children, but without additional prints
        for data in self._child_data.values():
            original_verbose = data.verbose
            data.verbose = False
            data.save()
            data.verbose = original_verbose

    def jobs(self) -> List[Job]:
        """Return a list of jobs for the experiment"""
        return self._jobs.values()

    def cancel_jobs(self, ids: Optional[Union[str, List[str]]] = None) -> bool:
        """Cancel any running jobs.

        Args:
            ids: Job(s) to cancel. If None all non-finished jobs will be cancelled.

        Returns:
            True if the specified jobs were successfully cancelled
            otherwise false.
        """
        if isinstance(ids, str):
            ids = [ids]

        with self._jobs.lock:
            all_cancelled = True
            for jid, job in reversed(self._jobs.items()):
                if ids and jid not in ids:
                    # Skip cancelling this callback
                    continue
                if job and job.status() not in JOB_FINAL_STATES:
                    try:
                        job.cancel()
                        LOG.warning("Cancelled job [Job ID: %s]", jid)
                    except Exception as err:  # pylint: disable=broad-except
                        all_cancelled = False
                        LOG.warning("Unable to cancel job [Job ID: %s]:\n%s", jid, err)
                        continue

                # Remove done or cancelled job futures
                if jid in self._job_futures:
                    del self._job_futures[jid]

        return all_cancelled

    def cancel_analysis(self, ids: Optional[Union[str, List[str]]] = None) -> bool:
        """Cancel any queued analysis callbacks.

        .. note::
            A currently running analysis callback cannot be cancelled.

        Args:
            ids: Analysis callback(s) to cancel. If None all non-finished
                 analysis will be cancelled.

        Returns:
            True if the specified analysis callbacks were successfully
            cancelled otherwise false.
        """
        if isinstance(ids, str):
            ids = [ids]

        # Lock analysis futures so we can't add more while trying to cancel
        with self._analysis_futures.lock:
            all_cancelled = True
            not_running = []
            for cid, callback in reversed(self._analysis_callbacks.items()):
                if ids and cid not in ids:
                    # Skip cancelling this callback
                    continue

                # Set event to cancel callback
                callback.event.set()

                # Check for running callback that can't be cancelled
                if callback.status == AnalysisStatus.RUNNING:
                    all_cancelled = False
                    LOG.warning(
                        "Unable to cancel running analysis callback [Experiment ID: %s]"
                        "[Analysis Callback ID: %s]",
                        self.experiment_id,
                        cid,
                    )
                else:
                    not_running.append(cid)

            # Wait for completion of other futures cancelled via event.set
            waited = futures.wait([self._analysis_futures[cid] for cid in not_running], timeout=1)
            # Get futures that didn't raise exception
            for fut in waited.done:
                if fut.done() and not fut.exception():
                    cid = fut.result()[0]
                    if cid in self._analysis_futures:
                        del self._analysis_futures[cid]

        return all_cancelled

    def cancel(self) -> bool:
        """Attempt to cancel any running jobs and queued analysis callbacks.

        .. note::
            A running analysis callback cannot be cancelled.

        Returns:
            True if all jobs and analysis are successfully cancelled, otherwise false.
        """
        # Cancel analysis first since it is queued on jobs, then cancel jobs
        # otherwise there can be a race issue when analysis starts running
        # as soon as jobs are cancelled
        analysis_cancelled = self.cancel_analysis()
        jobs_cancelled = self.cancel_jobs()
        return analysis_cancelled and jobs_cancelled

    def block_for_results(self, timeout: Optional[float] = None) -> "ExperimentData":
        """Block until all pending jobs and analysis callbacks finish.

        Args:
            timeout: Timeout in seconds for waiting for results.

        Returns:
            The experiment data with finished jobs and post-processing.
        """
        start_time = time.time()
        with self._job_futures.lock and self._analysis_futures.lock:
            # Lock threads to get all current job and analysis futures
            # at the time of function call and then release the lock
            job_ids = self._job_futures.keys()
            job_futs = self._job_futures.values()
            analysis_ids = self._analysis_futures.keys()
            analysis_futs = self._analysis_futures.values()

        # Wait for futures
        self._wait_for_futures(job_futs + analysis_futs, name="jobs and analysis", timeout=timeout)

        # Clean up done job futures
        num_jobs = len(job_ids)
        for jid, fut in zip(job_ids, job_futs):
            if (fut.done() and not fut.exception()) or fut.cancelled():
                if jid in self._job_futures:
                    del self._job_futures[jid]
                    num_jobs -= 1

        # Clean up done analysis futures
        num_analysis = len(analysis_ids)
        for cid, fut in zip(analysis_ids, analysis_futs):
            if (fut.done() and not fut.exception()) or fut.cancelled():
                if cid in self._analysis_futures:
                    del self._analysis_futures[cid]
                    num_analysis -= 1

        # Check if more futures got added while this function was running
        # and block recursively. This could happen if an analysis callback
        # spawns another callback or creates more jobs
        if len(self._job_futures) > num_jobs or len(self._analysis_futures) > num_analysis:
            time_taken = time.time() - start_time
            if timeout is not None:
                timeout = max(0, timeout - time_taken)
            return self.block_for_results(timeout=timeout)

        return self

    def _wait_for_futures(
        self, futs: List[futures.Future], name: str = "futures", timeout: Optional[float] = None
    ) -> bool:
        """Wait for jobs to finish running.

        Args:
            futs: Job or analysis futures to wait for.
            name: type name for future for logger messages.
            timeout: The length of time to wait for all jobs
                     before returning False.

        Returns:
            True if all jobs finished. False if timeout time was reached
            or any jobs were cancelled or had an exception.
        """
        waited = futures.wait(futs, timeout=timeout)
        value = True

        # Log futures still running after timeout
        if waited.not_done:
            LOG.info(
                "Waiting for %s timed out before completion [Experiment ID: %s].",
                name,
                self.experiment_id,
            )
            value = False

        # Check for futures that were cancelled or errorred
        excepts = ""
        for fut in waited.done:
            ex = fut.exception()
            if ex:
                excepts += "\n".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
                value = False
            elif fut.cancelled():
                LOG.debug(
                    "%s was cancelled before completion [Experiment ID: %s]",
                    name,
                    self.experiment_id,
                )
                value = False
            elif not fut.result()[1]:
                # The job/analysis did not succeed, and the failure reflects in the second
                # returned value of _add_job_data/_run_analysis_callback. See details in Issue #866.
                value = False
        if excepts:
            LOG.error(
                "%s raised exceptions [Experiment ID: %s]:%s", name, self.experiment_id, excepts
            )

        return value

    def status(self) -> ExperimentStatus:
        """Return the experiment status.

        Possible return values for :class:`.ExperimentStatus` are

        * :attr:`~.ExperimentStatus.EMPTY` - experiment data is empty
        * :attr:`~.ExperimentStatus.INITIALIZING` - experiment jobs are being initialized
        * :attr:`~.ExperimentStatus.QUEUED` - experiment jobs are queued
        * :attr:`~.ExperimentStatus.RUNNING` - experiment jobs is actively running
        * :attr:`~.ExperimentStatus.CANCELLED` - experiment jobs or analysis has been cancelled
        * :attr:`~.ExperimentStatus.POST_PROCESSING` - experiment analysis is actively running
        * :attr:`~.ExperimentStatus.DONE` - experiment jobs and analysis have successfully run
        * :attr:`~.ExperimentStatus.ERROR` - experiment jobs or analysis incurred an error

        .. note::

            If an experiment has status :attr:`~.ExperimentStatus.ERROR`
            there may still be pending or running jobs. In these cases it
            may be beneficial to call :meth:`cancel_jobs` to terminate these
            remaining jobs.

        Returns:
            The experiment status.
        """
        if all(
            len(container) == 0
            for container in [
                self._result_data,
                self._jobs,
                self._job_futures,
                self._analysis_callbacks,
                self._analysis_futures,
                self._figures,
                self._analysis_results,
            ]
        ):
            return ExperimentStatus.EMPTY

        # Return job status is job is not DONE
        try:
            return {
                JobStatus.INITIALIZING: ExperimentStatus.INITIALIZING,
                JobStatus.QUEUED: ExperimentStatus.QUEUED,
                JobStatus.VALIDATING: ExperimentStatus.VALIDATING,
                JobStatus.RUNNING: ExperimentStatus.RUNNING,
                JobStatus.CANCELLED: ExperimentStatus.CANCELLED,
                JobStatus.ERROR: ExperimentStatus.ERROR,
            }[self.job_status()]
        except KeyError:
            pass

        # Return analysis status if Done, cancelled or error
        try:
            return {
                AnalysisStatus.DONE: ExperimentStatus.DONE,
                AnalysisStatus.CANCELLED: ExperimentStatus.CANCELLED,
                AnalysisStatus.ERROR: ExperimentStatus.ERROR,
            }[self.analysis_status()]
        except KeyError:
            return ExperimentStatus.POST_PROCESSING

    def job_status(self) -> JobStatus:
        """Return the experiment job execution status.

        Possible return values for :class:`.JobStatus` are

        * :attr:`~.JobStatus.ERROR` - if any job incurred an error
        * :attr:`~.JobStatus.CANCELLED` - if any job is cancelled.
        * :attr:`~.JobStatus.RUNNING` - if any job is still running.
        * :attr:`~.JobStatus.QUEUED` - if any job is queued.
        * :attr:`~.JobStatus.VALIDATING` - if any job is being validated.
        * :attr:`~.JobStatus.INITIALIZING` - if any job is being initialized.
        * :attr:`~.JobStatus.DONE` - if all jobs are finished.

        .. note::

            If an experiment has status :attr:`~.JobStatus.ERROR` or
            :attr:`~.JobStatus.CANCELLED` there may still be pending or
            running jobs. In these cases it may be beneficial to call
            :meth:`cancel_jobs` to terminate these remaining jobs.

        Returns:
            The job execution status.
        """
        statuses = set()
        with self._jobs.lock:

            # No jobs present
            if not self._jobs:
                return JobStatus.DONE

            statuses = set()
            for job in self._jobs.values():
                if job:
                    statuses.add(job.status())

        # If any jobs are in non-DONE state return that state
        for stat in [
            JobStatus.ERROR,
            JobStatus.CANCELLED,
            JobStatus.RUNNING,
            JobStatus.QUEUED,
            JobStatus.VALIDATING,
            JobStatus.INITIALIZING,
        ]:
            if stat in statuses:
                return stat

        return JobStatus.DONE

    def analysis_status(self) -> AnalysisStatus:
        """Return the data analysis post-processing status.

        Possible return values for :class:`.AnalysisStatus` are

        * :attr:`~.AnalysisStatus.ERROR` - if any analysis callback incurred an error
        * :attr:`~.AnalysisStatus.CANCELLED` - if any analysis callback is cancelled.
        * :attr:`~.AnalysisStatus.RUNNING` - if any analysis callback is actively running.
        * :attr:`~.AnalysisStatus.QUEUED` - if any analysis callback is queued.
        * :attr:`~.AnalysisStatus.DONE` - if all analysis callbacks have successfully run.

        Returns:
            Then analysis status.
        """
        statuses = set()
        for status in self._analysis_callbacks.values():
            statuses.add(status.status)

        for stat in [
            AnalysisStatus.ERROR,
            AnalysisStatus.CANCELLED,
            AnalysisStatus.RUNNING,
            AnalysisStatus.QUEUED,
        ]:
            if stat in statuses:
                return stat

        return AnalysisStatus.DONE

    def job_errors(self) -> str:
        """Return any errors encountered in job execution."""
        errors = []

        # Get any job errors
        for job in self._jobs.values():
            if job and job.status() == JobStatus.ERROR:
                if hasattr(job, "error_message"):
                    error_msg = job.error_message()
                else:
                    error_msg = ""
                errors.append(f"\n[Job ID: {job.job_id()}]: {error_msg}")

        # Get any job futures errors:
        for jid, fut in self._job_futures.items():
            if fut and fut.done() and fut.exception():
                ex = fut.exception()
                errors.append(
                    f"[Job ID: {jid}]"
                    "\n".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
                )
        return "".join(errors)

    def analysis_errors(self) -> str:
        """Return any errors encountered during analysis callbacks."""
        errors = []

        # Get any callback errors
        for cid, callback in self._analysis_callbacks.items():
            if callback.status == AnalysisStatus.ERROR:
                errors.append(f"\n[Analysis Callback ID: {cid}]: {callback.error_msg}")

        return "".join(errors)

    def errors(self) -> str:
        """Return errors encountered during job and analysis execution.

        .. note::
            To display only job or analysis errors use the
            :meth:`job_errors` or :meth:`analysis_errors` methods.

        Returns:
            Experiment errors.
        """
        return self.job_errors() + self.analysis_errors()

    # Children handling

    def add_child_data(self, experiment_data: ExperimentData):
        """Add child experiment data to the current experiment data"""
        experiment_data.parent_id = self.experiment_id
        self._child_data[experiment_data.experiment_id] = experiment_data
        self.metadata["child_data_ids"] = self._child_data.keys()

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

    @classmethod
    def load(cls, experiment_id: str, service: IBMExperimentService) -> "ExperimentData":
        """Load a saved experiment data from a database service.

        Args:
            experiment_id: Experiment ID.
            service: the database service.

        Returns:
            The loaded experiment data.
        """
        data = service.experiment(experiment_id, json_decoder=cls._json_decoder)
        if service.experiment_has_file(experiment_id, cls._metadata_filename):
            metadata = service.file_download(experiment_id, cls._metadata_filename)
            data.metadata.update(metadata)
        expdata = cls(service=service, db_data=data)

        # Retrieve data and analysis results
        # Maybe this isn't necessary but the repr of the class should
        # be updated to show correct number of results including remote ones
        expdata._retrieve_data()
        expdata._retrieve_analysis_results()

        # mark it as existing in the DB
        expdata._created_in_db = True

        child_data_ids = expdata.metadata.pop("child_data_ids", [])
        child_data = [ExperimentData.load(child_id, service) for child_id in child_data_ids]
        expdata._set_child_data(child_data)

        return expdata

    def copy(self, copy_results: bool = True) -> "ExperimentData":
        """Make a copy of the experiment data with a new experiment ID.

        Args:
            copy_results: If True copy the analysis results and figures
                          into the returned container, along with the
                          experiment data and metadata. If False only copy
                          the experiment data and metadata.

        Returns:
            A copy of the experiment data object with the same data
            but different IDs.

        .. note:
            If analysis results and figures are copied they will also have
            new result IDs and figure names generated for the copies.

            This method can not be called from an analysis callback. It waits
            for analysis callbacks to complete before copying analysis results.
        """
        new_instance = ExperimentData(
            backend=self.backend,
            service=self.service,
            parent_id=self.parent_id,
            job_ids=self.job_ids,
            child_data=list(self._child_data.values()),
            verbose=self.verbose,
        )
        new_instance._db_data = self._db_data.copy()
        new_instance._db_data.experiment_id = str(
            uuid.uuid4()
        )  # different id for copied experiment
        if self.experiment is None:
            new_instance._experiment = None
        else:
            new_instance._experiment = self.experiment.copy()

        LOG.debug(
            "Copying experiment data [Experiment ID: %s]: %s",
            self.experiment_id,
            new_instance.experiment_id,
        )

        # Copy basic properties and metadata

        new_instance._jobs = self._jobs.copy_object()
        new_instance._auto_save = self._auto_save
        new_instance._extra_data = self._extra_data

        # Copy circuit result data and jobs
        with self._result_data.lock:  # Hold the lock so no new data can be added.
            new_instance._result_data = self._result_data.copy_object()
            for jid, fut in self._job_futures.items():
                if not fut.done():
                    new_instance._add_job_future(new_instance._jobs[jid])

        # If not copying results return the object
        if not copy_results:
            return new_instance

        # Copy results and figures.
        # This requires analysis callbacks to finish
        self._wait_for_futures(self._analysis_futures.values(), name="analysis")
        with self._analysis_results.lock:
            new_instance._analysis_results = ThreadSafeOrderedDict()
            new_instance.add_analysis_results([result.copy() for result in self.analysis_results()])
        with self._figures.lock:
            new_instance._figures = ThreadSafeOrderedDict()
            new_instance.add_figures(self._figures.values())

        # Recursively copy child data
        child_data = [data.copy(copy_results=copy_results) for data in self.child_data()]
        new_instance._set_child_data(child_data)
        return new_instance

    def _set_child_data(self, child_data: List[ExperimentData]):
        """Set child experiment data for the current experiment."""
        self._child_data = ThreadSafeOrderedDict()
        for data in child_data:
            self.add_child_data(data)
        self._db_data.metadata["child_data_ids"] = self._child_data.keys()

    def _set_service(self, service: IBMExperimentService) -> None:
        """Set the service to be used for storing experiment data,
           to this experiment itself and its descendants.

        Args:
            service: Service to be used.

        Raises:
            ExperimentDataError: If an experiment service is already being used.
        """
        if self._service:
            raise ExperimentDataError("An experiment service is already being used.")
        self._service = service
        for result in self._analysis_results.values():
            result.service = service
        with contextlib.suppress(Exception):
            self.auto_save = self._service.options.get("auto_save", False)
        for data in self.child_data():
            data._set_service(service)

    def add_tags_recursive(self, tags2add: List[str]) -> None:
        """Add tags to this experiment itself and its descendants

        Args:
            tags2add - the tags that will be added to the existing tags
        """
        self.tags += tags2add
        for data in self._child_data.values():
            data.add_tags_recursive(tags2add)

    def remove_tags_recursive(self, tags2remove: List[str]) -> None:
        """Remove tags from this experiment itself and its descendants

        Args:
            tags2remove - the tags that will be removed from the existing tags
        """
        self.tags = [x for x in self.tags if x not in tags2remove]
        for data in self._child_data.values():
            data.remove_tags_recursive(tags2remove)

    # represetnation and serialization

    def __repr__(self):
        out = f"{type(self).__name__}({self.experiment_type}"
        out += f", {self.experiment_id}"
        if self.parent_id:
            out += f", parent_id={self.parent_id}"
        if self.tags:
            out += f", tags={self.tags}"
        if self.job_ids:
            out += f", job_ids={self.job_ids}"
        if self.share_level:
            out += f", share_level={self.share_level}"
        if self.metadata:
            out += f", metadata=<{len(self.metadata)} items>"
        if self.figure_names:
            out += f", figure_names={self.figure_names}"
        if self.notes:
            out += f", notes={self.notes}"
        if self._extra_data:
            for key, val in self._extra_data.items():
                out += f", {key}={repr(val)}"
        out += ")"
        return out

    def __getattr__(self, name: str) -> Any:
        try:
            return self._extra_data[name]
        except KeyError:
            # pylint: disable=raise-missing-from
            raise AttributeError("Attribute %s is not defined" % name)

    def _safe_serialize_jobs(self):
        """Return serializable object for stored jobs"""
        # Since Job objects are not serializable this removes
        # them from the jobs dict and returns {job_id: None}
        # that can be used to retrieve jobs from a service after loading
        jobs = ThreadSafeOrderedDict()
        with self._jobs.lock:
            for jid in self._jobs.keys():
                jobs[jid] = None
        return jobs

    def _safe_serialize_figures(self):
        """Return serializable object for stored figures"""
        # Convert any MPL figures into SVG images before serializing
        figures = ThreadSafeOrderedDict()
        with self._figures.lock:
            for name, figure in self._figures.items():
                if isinstance(figure, pyplot.Figure):
                    figures[name] = plot_to_svg_bytes(figure)
                else:
                    figures[name] = figure
        return figures

    def __json_encode__(self):
        if any(not fut.done() for fut in self._job_futures.values()):
            raise QiskitError(
                "Not all experiment jobs have finished. Jobs must be "
                "cancelled or done to serialize experiment data."
            )
        if any(not fut.done() for fut in self._analysis_futures.values()):
            raise QiskitError(
                "Not all experiment analysis has finished. Analysis must be "
                "cancelled or done to serialize experiment data."
            )
        json_value = {
            "_db_data": self._db_data,
            "_analysis_results": self._analysis_results,
            "_analysis_callbacks": self._analysis_callbacks,
            "_deleted_figures": self._deleted_figures,
            "_deleted_analysis_results": self._deleted_analysis_results,
            "_result_data": self._result_data,
            "_extra_data": self._extra_data,
            "_created_in_db": self._created_in_db,
            "_figures": self._safe_serialize_figures(),  # Convert figures to SVG
            "_jobs": self._safe_serialize_jobs(),  # Handle non-serializable objects
            "_experiment": self._experiment,
            "_child_data": self._child_data,
        }
        # the attribute self._service in charge of the connection and communication with the
        #  experiment db. It doesn't have meaning in the json format so there is no need to serialize
        #  it.
        for att in ["_service", "_backend"]:
            json_value[att] = None
            value = getattr(self, att)
            if value is not None:
                LOG.info("%s cannot be JSON serialized", str(type(value)))

        return json_value

    @classmethod
    def __json_decode__(cls, value):
        ret = cls()
        for att, att_val in value.items():
            setattr(ret, att, att_val)
        return ret

    def __getstate__(self):
        if any(not fut.done() for fut in self._job_futures.values()):
            LOG.warning(
                "Not all job futures have finished."
                " Data from running futures will not be serialized."
            )
        if any(not fut.done() for fut in self._analysis_futures.values()):
            LOG.warning(
                "Not all analysis callbacks have finished."
                " Results from running callbacks will not be serialized."
            )

        state = self.__dict__.copy()

        # Remove non-pickleable attributes
        for key in ["_job_futures", "_analysis_futures", "_analysis_executor", "_monitor_executor"]:
            del state[key]

        # Convert figures to SVG
        state["_figures"] = self._safe_serialize_figures()

        # Handle partially pickleable attributes
        state["_jobs"] = self._safe_serialize_jobs()

        return state

    @staticmethod
    def get_service_from_backend(backend):
        """Initializes the server from the backend data"""
        db_url = "https://auth.quantum-computing.ibm.com/api"
        try:
            credentials = backend._provider.credentials
            service = IBMExperimentService(token=credentials.token, url=db_url)
            return service
        except Exception:  # pylint: disable=broad-except
            return None

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Initialize non-pickled attributes
        self._job_futures = ThreadSafeOrderedDict()
        self._analysis_futures = ThreadSafeOrderedDict()
        self._analysis_executor = futures.ThreadPoolExecutor(max_workers=1)

    def __str__(self):
        line = 51 * "-"
        n_res = len(self._analysis_results)
        status = self.status()
        ret = line
        ret += f"\nExperiment: {self.experiment_type}"
        ret += f"\nExperiment ID: {self.experiment_id}"
        if self._db_data.parent_id:
            ret += f"\nParent ID: {self._db_data.parent_id}"
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
        ret += f"\nData: {len(self._result_data)}"
        ret += f"\nAnalysis Results: {n_res}"
        ret += f"\nFigures: {len(self._figures)}"
        return ret


@contextlib.contextmanager
def service_exception_to_warning():
    """Convert an exception raised by experiment service to a warning."""
    try:
        yield
    except Exception:  # pylint: disable=broad-except
        LOG.warning("Experiment service operation failed: %s", traceback.format_exc())


class ExperimentStatus(enum.Enum):
    """Class for experiment status enumerated type."""

    EMPTY = "experiment data is empty"
    INITIALIZING = "experiment jobs are being initialized"
    VALIDATING = "experiment jobs are validating"
    QUEUED = "experiment jobs are queued"
    RUNNING = "experiment jobs is actively running"
    CANCELLED = "experiment jobs or analysis has been cancelled"
    POST_PROCESSING = "experiment analysis is actively running"
    DONE = "experiment jobs and analysis have successfully run"
    ERROR = "experiment jobs or analysis incurred an error"

    def __json_encode__(self):
        return self.name

    @classmethod
    def __json_decode__(cls, value):
        return cls.__members__[value]  # pylint: disable=unsubscriptable-object


class AnalysisStatus(enum.Enum):
    """Class for analysis callback status enumerated type."""

    QUEUED = "analysis callback is queued"
    RUNNING = "analysis callback is actively running"
    CANCELLED = "analysis callback has been cancelled"
    DONE = "analysis callback has successfully run"
    ERROR = "analysis callback incurred an error"

    def __json_encode__(self):
        return self.name

    @classmethod
    def __json_decode__(cls, value):
        return cls.__members__[value]  # pylint: disable=unsubscriptable-object


@dataclasses.dataclass
class AnalysisCallback:
    """Dataclass for analysis callback status"""

    name: str = ""
    callback_id: str = ""
    status: AnalysisStatus = AnalysisStatus.QUEUED
    error_msg: Optional[str] = None
    event: Event = dataclasses.field(default_factory=Event)

    def __getstate__(self):
        # We need to remove the Event object from state when pickling
        # since events are not pickleable
        state = self.__dict__
        state["event"] = None
        return state

    def __json_encode__(self):
        return self.__getstate__()
