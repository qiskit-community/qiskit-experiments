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

"""Stored data class."""

import warnings
import logging
import dataclasses
import uuid
import enum
from typing import Optional, List, Any, Union, Callable, Dict, Tuple
import copy
from concurrent import futures
from functools import wraps
import traceback
import contextlib
from collections import deque
from datetime import datetime
import numpy as np

from matplotlib import pyplot
from qiskit import QiskitError
from qiskit.providers import Job, BaseJob, Backend, BaseBackend, Provider
from qiskit.result import Result
from qiskit.providers.jobstatus import JobStatus, JOB_FINAL_STATES
from qiskit_experiments.framework.json import ExperimentEncoder, ExperimentDecoder

from .database_service import DatabaseServiceV1
from .exceptions import DbExperimentDataError, DbExperimentEntryNotFound, DbExperimentEntryExists
from .db_analysis_result import DbAnalysisResultV1 as DbAnalysisResult
from .utils import (
    save_data,
    qiskit_version,
    plot_to_svg_bytes,
    ThreadSafeOrderedDict,
    ThreadSafeList,
)

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


class DbExperimentData:
    """Base common type for all versioned DbExperimentData classes.

    Note this class should not be inherited from directly, it is intended
    to be used for type checking. When implementing a custom DbExperimentData class,
    you should use the versioned classes as the parent class and not this class
    directly.
    """

    version = 0


class DbExperimentDataV1(DbExperimentData):
    """Class to define and handle experiment data stored in a database.

    This class serves as a container for experiment related data to be stored
    in a database, which may include experiment metadata, analysis results,
    and figures. It also provides methods used to interact with the database,
    such as storing into and retrieving from the database.
    """

    version = 1
    verbose = True  # Whether to print messages to the standard output.
    _metadata_version = 1
    _job_executor = futures.ThreadPoolExecutor()

    _json_encoder = ExperimentEncoder
    _json_decoder = ExperimentDecoder

    def __init__(
        self,
        experiment_type: Optional[str] = "Unknown",
        backend: Optional[Union[Backend, BaseBackend]] = None,
        service: Optional[DatabaseServiceV1] = None,
        experiment_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        job_ids: Optional[List[str]] = None,
        share_level: Optional[str] = None,
        metadata: Optional[Dict] = None,
        figure_names: Optional[List[str]] = None,
        notes: Optional[str] = None,
        **kwargs,
    ):
        """Initializes the DbExperimentData instance.

        Args:
            experiment_type: Experiment type.
            backend: Backend the experiment runs on.
            experiment_id: Experiment ID. One will be generated if not supplied.
            parent_id: The experiment ID of the parent experiment.
            tags: Tags to be associated with the experiment.
            job_ids: IDs of jobs submitted for the experiment.
            share_level: Whether this experiment can be shared with others. This
                is applicable only if the database service supports sharing. See
                the specific service provider's documentation on valid values.
            metadata: Additional experiment metadata.
            figure_names: Name of figures associated with this experiment.
            notes: Freeform notes about the experiment.
            **kwargs: Additional experiment attributes.
        """
        metadata = metadata or {}
        self._metadata = copy.deepcopy(metadata)
        self._source = self._metadata.pop(
            "_source",
            {
                "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
                "metadata_version": self._metadata_version,
                "qiskit_version": qiskit_version(),
            },
        )

        self._service = service
        if self.service is None:
            self._set_service_from_backend(backend)
        self._backend = backend
        self._auto_save = False

        self._id = experiment_id or str(uuid.uuid4())
        self._parent_id = parent_id
        self._type = experiment_type
        self._tags = tags or []
        self._share_level = share_level
        self._notes = notes or ""

        self._jobs = ThreadSafeOrderedDict(job_ids or [])
        self._job_futures = ThreadSafeOrderedDict()
        self._analysis_callbacks = ThreadSafeOrderedDict()
        self._analysis_futures = ThreadSafeOrderedDict()
        self._analysis_executor = futures.ThreadPoolExecutor(max_workers=1)

        self._data = ThreadSafeList()
        self._figures = ThreadSafeOrderedDict(figure_names or [])
        self._analysis_results = ThreadSafeOrderedDict()

        self._deleted_figures = deque()
        self._deleted_analysis_results = deque()

        self._created_in_db = False
        self._extra_data = kwargs

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

    def _set_service_from_backend(self, backend: Union[Backend, BaseBackend]) -> None:
        """Set the service to be used from the input backend.

        Args:
            backend: Backend whose provider may offer experiment service.
        """
        with contextlib.suppress(Exception):
            self._service = backend.provider().service("experiment")
            self._auto_save = self._service.preferences.get("auto_save", False)

    def add_data(
        self,
        data: Union[Result, List[Result], Job, List[Job], Dict, List[Dict]],
        timeout: Optional[float] = None,
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
            timeout: (Deprecated) Timeout waiting for job to finish, if `data` is a ``Job``.

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

        # Extract job data (Deprecated) and directly add non-job data
        jobs = []
        with self._data.lock:
            for datum in data:
                if isinstance(datum, (Job, BaseJob)):
                    jobs.append(datum)
                elif isinstance(datum, dict):
                    self._data.append(datum)
                elif isinstance(datum, Result):
                    self._add_result_data(datum)
                else:
                    raise TypeError(f"Invalid data type {type(datum)}.")

        # Remove after deprecation is finished
        if jobs:
            warnings.warn(
                "Passing Jobs to the `add_data` method is deprecated as of "
                "qiskit-experiments 0.3.0 and will be removed in the 0.4.0 release. "
                "Use the `add_jobs` method to add jobs instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if timeout is not None:
                warnings.warn(
                    "The `timeout` kwarg of is deprecated as of "
                    "qiskit-experiments 0.3.0 and will be removed in the 0.4.0 release. "
                    "Use the `add_jobs` method to add jobs with timeout.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            self.add_jobs(jobs, timeout=timeout)

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
        if isinstance(jobs, (Job, BaseJob)):
            jobs = [jobs]

        # Add futures for extracting finished job data
        timeout_ids = []
        for job in jobs:
            jid = job.job_id()
            if self.backend and self.backend.name() != job.backend().name():
                LOG.warning(
                    "Adding a job from a backend (%s) that is different "
                    "than the current backend (%s). "
                    "The new backend will be used, but "
                    "service is not changed if one already exists.",
                    job.backend(),
                    self.backend,
                )
            self._backend = job.backend()
            if not self._service:
                self._set_service_from_backend(self._backend)

            if jid in self._jobs:
                LOG.warning(
                    "Skipping duplicate job, a job with this ID already exists [Job ID: %s]", jid
                )
            else:
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
        job: Union[Job, BaseJob],
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
            LOG.warning("Addind data from job failed [Job ID: %s]", job.job_id())
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
                status=AnalysisStatus.QUEUED,
            )

            # Get futures to wait for before running analysis
            if self._analysis_futures:
                # If analysis futures are present wait for all previous analysis
                # futures to complete before running the latest one
                futs = self._analysis_futures.values()
            else:
                # Create a dummy future to wait for jobs to finish.
                # This allows subsequent analysis futures to be cancelled as they
                # will be queued by the ThreadPoolExecutor while this future
                # is still running
                futs = [self._analysis_executor.submit(self._wait_for_jobs)]

            # Add run analysis future
            self._analysis_futures[cid] = self._analysis_executor.submit(
                self._run_analysis_callback, futs, cid, callback, **kwargs
            )

    def _run_analysis_callback(
        self, futs: List[futures.Future], callback_id: str, callback: Callable, **kwargs
    ):
        """Run an analysis callback after specified futures have finished."""
        if callback_id not in self._analysis_callbacks:
            raise ValueError(f"No analysis callback with id {callback_id}")

        # Wait for previous futures to finish
        if futs:
            waited = futures.wait(futs)
            # Check all previous futures finished, and cancel analysis callback if not
            for fut in waited.done:
                if fut.cancelled() or fut.exception() or not fut.result()[1]:
                    self._analysis_callbacks[callback_id].status = AnalysisStatus.CANCELLED
                    LOG.warning("Cancelled analysis callback [Analysis ID: %s]", callback_id)
                    return callback_id, False

        # Run callback function
        try:
            self._analysis_callbacks[callback_id].status = AnalysisStatus.RUNNING
            LOG.debug(
                "Running analysis callback '%s' [Analysis ID: %s]",
                self._analysis_callbacks[callback_id].name,
                callback_id,
            )
            callback(self, **kwargs)
            self._analysis_callbacks[callback_id].status = AnalysisStatus.DONE
            LOG.debug("Analysis callback finished [Analysis ID: %s]", callback_id)
            return callback_id, True
        except Exception as ex:  # pylint: disable=broad-except
            self._analysis_callbacks[callback_id].status = AnalysisStatus.ERROR
            tb_text = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
            error_msg = f"Analysis callback failed [Analysis ID: {callback_id}]:\n{tb_text}"
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
        with self._data.lock:
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
                self._data.append(data)

    def _retrieve_data(self):
        """Retrieve job data if missing experiment data."""
        if self._data or not self._backend:
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
            return self._data.copy()
        if isinstance(index, (int, slice)):
            return self._data[index]
        if isinstance(index, str):
            return [data for data in self._data if data.get("job_id") == index]
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
            DbExperimentEntryExists: If the figure with the same name already exists,
                and `overwrite=True` is not specified.
            ValueError: If an input parameter has an invalid value.
        """
        if (
            isinstance(figures, list)
            and figure_names is not None
            and (not isinstance(figure_names, list) or len(figures) != len(figure_names))
        ):
            raise ValueError(
                "The parameter figure_names must be None or a list of "
                "the same size as the parameter figures."
            )
        if not isinstance(figures, list):
            figures = [figures]
        if figure_names is not None and not isinstance(figure_names, list):
            figure_names = [figure_names]

        added_figs = []
        for idx, figure in enumerate(figures):
            if figure_names is None:
                if isinstance(figure, str):
                    fig_name = figure
                else:
                    fig_name = (
                        f"figure_{self.experiment_id[:8]}_"
                        f"{datetime.now().isoformat()}_{len(self._figures)}.svg"
                    )
            else:
                fig_name = figure_names[idx]

            if not fig_name.endswith(".svg"):
                LOG.info("File name %s does not have an SVG extension. A '.svg' is added.")
                fig_name += ".svg"

            existing_figure = fig_name in self._figures
            if existing_figure and not overwrite:
                raise DbExperimentEntryExists(
                    f"A figure with the name {fig_name} for this experiment "
                    f"already exists. Specify overwrite=True if you "
                    f"want to overwrite it."
                )
            # figure_data = None
            if isinstance(figure, str):
                with open(figure, "rb") as file:
                    figure = file.read()

            self._figures[fig_name] = figure

            save = save_figure if save_figure is not None else self.auto_save
            if save and self._service:
                if isinstance(figure, pyplot.Figure):
                    figure = plot_to_svg_bytes(figure)
                data = {
                    "experiment_id": self.experiment_id,
                    "figure": figure,
                    "figure_name": fig_name,
                }
                save_data(
                    is_new=(not existing_figure),
                    new_func=self._service.create_figure,
                    update_func=self._service.update_figure,
                    new_data={},
                    update_data=data,
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
            DbExperimentEntryNotFound: If the figure is not found.
        """
        if isinstance(figure_key, int):
            figure_key = self._figures.keys()[figure_key]
        elif figure_key not in self._figures:
            raise DbExperimentEntryNotFound(f"Figure {figure_key} not found.")

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
            DbExperimentEntryNotFound: If the figure cannot be found.
        """
        if isinstance(figure_key, int):
            figure_key = self._figures.keys()[figure_key]

        figure_data = self._figures.get(figure_key, None)
        if figure_data is None and self.service:
            figure_data = self.service.figure(
                experiment_id=self.experiment_id, figure_name=figure_key
            )
            self._figures[figure_key] = figure_data

        if figure_data is None:
            raise DbExperimentEntryNotFound(f"Figure {figure_key} not found.")

        if file_name:
            with open(file_name, "wb") as output:
                num_bytes = output.write(figure_data)
                return num_bytes
        return figure_data

    @do_auto_save
    def add_analysis_results(
        self,
        results: Union[DbAnalysisResult, List[DbAnalysisResult]],
    ) -> None:
        """Save the analysis result.

        Args:
            results: Analysis results to be saved.
        """
        if not isinstance(results, list):
            results = [results]

        for result in results:
            self._analysis_results[result.result_id] = result

            with contextlib.suppress(DbExperimentDataError):
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
            DbExperimentEntryNotFound: If analysis result not found.
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
                result_id = result["result_id"]
                self._analysis_results[result_id] = DbAnalysisResult._from_service_data(result)

    def analysis_results(
        self,
        index: Optional[Union[int, slice, str]] = None,
        refresh: bool = False,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> Union[DbAnalysisResult, List[DbAnalysisResult]]:
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
            DbExperimentEntryNotFound: If the entry cannot be found.
        """
        if block:
            self._wait_for_analysis(timeout=timeout)
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
                raise DbExperimentEntryNotFound(_make_not_found_message(index))
            return self._analysis_results.values()[index]
        if isinstance(index, slice):
            results = self._analysis_results.values()[index]
            if not results:
                raise DbExperimentEntryNotFound(_make_not_found_message(index))
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
                raise DbExperimentEntryNotFound(_make_not_found_message(index))
            if len(filtered) == 1:
                return filtered[0]
            else:
                return filtered

        raise TypeError(f"Invalid index type {type(index)}.")

    def save_metadata(self) -> None:
        """Save this experiments metadata to a database service.

        .. note::
            This method does not save analysis results nor figures.
            Use :meth:`save` for general saving of all experiment data.

            See :meth:`qiskit.providers.experiment.DatabaseServiceV1.create_experiment`
            for fields that are saved.
        """
        self._save_experiment_metadata()

    def _save_experiment_metadata(self) -> None:
        """Save this experiments metadata to a database service.

        .. note::
            This method does not save analysis results nor figures.
            Use :meth:`save` for general saving of all experiment data.

            See :meth:`qiskit.providers.experiment.DatabaseServiceV1.create_experiment`
            for fields that are saved.
        """
        if not self._service:
            LOG.warning(
                "Experiment cannot be saved because no experiment service is available. "
                "An experiment service is available, for example, "
                "when using an IBM Quantum backend."
            )
            return

        if not self._backend:
            LOG.warning("Experiment cannot be saved because backend is missing.")
            return

        metadata = copy.deepcopy(self._metadata)
        metadata["_source"] = self._source

        update_data = {
            "experiment_id": self._id,
            "metadata": metadata,
            "job_ids": self.job_ids,
            "tags": self.tags,
            "notes": self.notes,
        }
        new_data = {
            "experiment_type": self._type,
            "backend_name": self._backend.name(),
            "provider": self._provider,
        }
        if self.share_level:
            update_data["share_level"] = self.share_level
        if self.parent_id:
            update_data["parent_id"] = self.parent_id

        self._created_in_db, _ = save_data(
            is_new=(not self._created_in_db),
            new_func=self._service.create_experiment,
            update_func=self._service.update_experiment,
            new_data=new_data,
            update_data=update_data,
            json_encoder=self._json_encoder,
        )

    def save(self) -> None:
        """Save the experiment data to a database service.

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

        self._save_experiment_metadata()
        for result in self._analysis_results.values():
            result.save()

        for result in self._deleted_analysis_results.copy():
            with service_exception_to_warning():
                self._service.delete_analysis_result(result_id=result)
            self._deleted_analysis_results.remove(result)

        with self._figures.lock:
            for name, figure in self._figures.items():
                if figure is None:
                    continue
                if isinstance(figure, pyplot.Figure):
                    figure = plot_to_svg_bytes(figure)
                data = {"experiment_id": self.experiment_id, "figure": figure, "figure_name": name}
                save_data(
                    is_new=True,
                    new_func=self._service.create_figure,
                    update_func=self._service.update_figure,
                    new_data={},
                    update_data=data,
                )

        for name in self._deleted_figures.copy():
            with service_exception_to_warning():
                self._service.delete_figure(experiment_id=self.experiment_id, figure_name=name)
            self._deleted_figures.remove(name)

        if self._created_in_db and self.verbose:
            print(
                "You can view the experiment online at "
                "https://quantum-computing.ibm.com/experiments/" + self.experiment_id
            )

    @classmethod
    def load(cls, experiment_id: str, service: DatabaseServiceV1) -> "DbExperimentDataV1":
        """Load a saved experiment data from a database service.

        Args:
            experiment_id: Experiment ID.
            service: the database service.

        Returns:
            The loaded experiment data.
        """
        service_data = service.experiment(experiment_id, json_decoder=cls._json_decoder)

        # Parse serialized metadata
        metadata = service_data.pop("metadata")

        # Initialize container
        expdata = DbExperimentDataV1(
            experiment_type=service_data.pop("experiment_type"),
            backend=service_data.pop("backend"),
            experiment_id=service_data.pop("experiment_id"),
            parent_id=service_data.pop("parent_id", None),
            tags=service_data.pop("tags"),
            job_ids=service_data.pop("job_ids"),
            share_level=service_data.pop("share_level"),
            metadata=metadata,
            figure_names=service_data.pop("figure_names"),
            notes=service_data.pop("notes"),
            **service_data,
        )

        if expdata.service is None:
            expdata.service = service

        # Retrieve data and analysis results
        # Maybe this isn't necessary but the repr of the class should
        # be updated to show correct number of results including remote ones
        expdata._retrieve_data()
        expdata._retrieve_analysis_results()

        # mark it as existing in the DB
        expdata._created_in_db = True
        return expdata

    def jobs(self) -> List[Job]:
        """Return a list of jobs for the experiment"""
        return self._jobs.values()

    def cancel_jobs(self, ids: Optional[List[str]] = None) -> bool:
        """Cancel any running jobs.

        Args:
            ids: Jobs to cancel. If None all non-finished jobs will be cancelled.

        Returns:
            True if the specified jobs were successfully cancelled
            otherwise false.
        """
        cancelled = True

        with self._jobs.lock:
            if ids is None:
                jobs = self._jobs.values()
            else:
                jobs = [self._jobs[jid] for jid in ids]

            for job in jobs:
                if job and job.status() not in JOB_FINAL_STATES:
                    try:
                        job.cancel()
                        LOG.debug("Cancelled job [Job ID: %s]", job.job_id())
                    except Exception as err:  # pylint: disable=broad-except
                        cancelled = False
                        LOG.warning("Unable to cancel job [Job ID: %s]:\n%s", job.job_id(), err)
        return cancelled

    def cancel_analysis(self, ids: Optional[List[str]] = None) -> bool:
        """Cancel any queued analysis callbacks.

        .. note::
            A currently running analysis callback cannot be cancelled.

        Args:
            ids: Analysis callbacks to cancel. If None all non-finished
                 analysis will be cancelled.

        Returns:
            True if the specified analysis callbacks were successfully
            cancelled otherwise false.
        """
        cancelled = True
        with self._analysis_futures.lock:
            if ids is None:
                ids = self._analysis_futures.keys()
                futs = self._analysis_futures.values()
            else:
                futs = [self._analysis_futures[i] for i in ids]
            for cid, fut in zip(ids, futs):
                if fut.done():
                    continue
                if fut.cancel():
                    self._analysis_callbacks[cid].status = AnalysisStatus.CANCELLED
                    LOG.info("Cancelled analysis callback [Analysis ID: %s]", cid)
                else:
                    cancelled = False
                    LOG.warning("Unable to cancel analysis callback [Analysis ID: %s]", cid)
        return cancelled

    def cancel(self) -> bool:
        """Attempt to cancel any running jobs and queued analysis callbacks.

        .. note::
            A running analysis callback cannot be cancelled.

        Returns:
            True if all jobs and analysis are successfully cancelled, otherwise false.
        """
        return self.cancel_jobs() and self.cancel_analysis()

    def block_for_results(self, timeout: Optional[float] = None) -> "DbExperimentDataV1":
        """Block until all pending jobs and analysis callbacks finish.

        Args:
            timeout: Timeout in seconds for waiting for results.

        Returns:
            The experiment data with finished jobs and post-processing.
        """
        if self._analysis_futures:
            self._wait_for_analysis(timeout)
        else:
            self._wait_for_jobs(timeout)
        self._removed_done_futures()
        return self

    def _wait_for_jobs(self, timeout: Optional[float] = None):
        """Wait for jobs to finish running"""
        waited = futures.wait(self._job_futures.values(), timeout=timeout)
        value = True
        # Log jobs still running after timeout
        if waited.not_done:
            LOG.info(
                "Waiting for jobs timed out before all jobs completed [Experiment ID: %s].",
                self.experiment_id,
            )
            value = False

        # Check for jobs that were cancelled or errorred
        excepts = ""
        for fut in waited.done:
            ex = fut.exception()
            if ex:
                excepts += "\n".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
                value = False
            elif fut.cancelled():
                LOG.debug(
                    "Job futures were cancelled before completion [Experiment ID: %s]",
                    self.experiment_id,
                )
                value = False
        if excepts:
            LOG.error(
                "Job futures raised exceptions [Experiment ID: %s]:%s", self.experiment_id, excepts
            )
        return "_wait_for_jobs", value

    def _wait_for_analysis(self, timeout: Optional[float] = None):
        """Wait for analysis callbacks to finish"""
        waited = futures.wait(self._analysis_futures.values(), timeout=timeout)
        # Log jobs still running after timeout
        if waited.not_done:
            LOG.info(
                "Waiting for analysis callbacks timed out before all "
                "callbacks completed [Experiment ID: %s].",
                self.experiment_id,
            )

        # Check for jobs that were cancelled or errorred
        excepts = ""
        for fut in waited.done:
            ex = fut.exception()
            if ex:
                excepts += "\n".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
            elif fut.cancelled():
                LOG.debug(
                    "Analysis callbacks were cancelled before completion [Experiment ID: %s]",
                    self.experiment_id,
                )
        if excepts:
            LOG.error(
                "Analysis callbacks raised exceptions [Experiment ID: %s]:%s",
                self.experiment_id,
                excepts,
            )

    def _removed_done_futures(self):
        """Remove futures that have finished"""

        def _keep_future(fut):
            return not fut.done() or fut.cancelled() or fut.exception()

        with self._analysis_futures.lock and self._job_futures.lock:
            not_done_analysis = [
                (cid, fut) for cid, fut in self._analysis_futures.items() if _keep_future(fut)
            ]
            self._analysis_futures = ThreadSafeOrderedDict(not_done_analysis)

            not_done_jobs = [
                (jid, fut) for jid, fut in self._job_futures.items() if _keep_future(fut)
            ]
            self._job_futures = ThreadSafeOrderedDict(not_done_jobs)

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
                self._data,
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
                errors.append(f"\n[Analysis ID: {cid}]: {callback.error_msg}")

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

    def copy(self, copy_results: bool = True) -> "DbExperimentDataV1":
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
        new_instance = self.__class__()
        LOG.debug(
            "Copying experiment data [Experiment ID: %s]: %s",
            self.experiment_id,
            new_instance.experiment_id,
        )

        # Copy basic properties and metadata
        new_instance._type = self.experiment_type
        new_instance._backend = self._backend
        new_instance._tags = self._tags
        new_instance._jobs = self._jobs.copy_object()
        new_instance._share_level = self._share_level
        new_instance._metadata = copy.deepcopy(self._metadata)
        new_instance._notes = self._notes
        new_instance._auto_save = self._auto_save
        new_instance._service = self._service
        new_instance._extra_data = self._extra_data

        # Copy circuit result data and jobs
        with self._data.lock:  # Hold the lock so no new data can be added.
            new_instance._data = self._data.copy_object()
            for jid, fut in self._job_futures.items():
                if not fut.done():
                    new_instance._add_job_future(new_instance._jobs[jid])

        # If not copying results return the object
        if not copy_results:
            return new_instance

        # Copy results and figures.
        # This requires analysis callbacks to finish
        self._wait_for_analysis()
        with self._analysis_results.lock:
            new_instance._analysis_results = ThreadSafeOrderedDict()
            new_instance.add_analysis_results([result.copy() for result in self.analysis_results()])
        with self._figures.lock:
            new_instance._figures = ThreadSafeOrderedDict()
            new_instance.add_figures(self._figures.values())

        return new_instance

    @property
    def tags(self) -> List[str]:
        """Return tags assigned to this experiment data.

        Returns:
            A list of tags assigned to this experiment data.

        """
        return self._tags

    @tags.setter
    def tags(self, new_tags: List[str]) -> None:
        """Set tags for this experiment."""
        if not isinstance(new_tags, list):
            raise DbExperimentDataError(
                f"The `tags` field of {type(self).__name__} must be a list."
            )
        self._tags = np.unique(new_tags).tolist()
        if self.auto_save:
            self.save_metadata()

    @property
    def metadata(self) -> Dict:
        """Return experiment metadata.

        Returns:
            Experiment metadata.
        """
        return self._metadata

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
        return self._id

    @property
    def parent_id(self) -> str:
        """Return parent experiment ID

        Returns:
            Parent ID.
        """
        return self._parent_id

    @property
    def job_ids(self) -> List[str]:
        """Return experiment job IDs.

        Returns: IDs of jobs submitted for this experiment.
        """
        return self._jobs.keys()

    @property
    def backend(self) -> Optional[Union[BaseBackend, Backend]]:
        """Return backend.

        Returns:
            Backend this experiment is for, or ``None`` if backend is unknown.
        """
        return self._backend

    @property
    def experiment_type(self) -> str:
        """Return experiment type.

        Returns:
            Experiment type.
        """
        return self._type

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
        return self._share_level

    @share_level.setter
    def share_level(self, new_level: str) -> None:
        """Set the experiment share level,
           only to this experiment and not to its descendants.

        Args:
            new_level: New experiment share level. Valid share levels are provider-
                specified. For example, IBM Quantum experiment service allows
                "public", "hub", "group", "project", and "private".
        """
        self._share_level = new_level
        if self.auto_save:
            self.save_metadata()

    @property
    def notes(self) -> str:
        """Return experiment notes.

        Returns:
            Experiment notes.
        """
        return self._notes

    @notes.setter
    def notes(self, new_notes: str) -> None:
        """Update experiment notes.

        Args:
            new_notes: New experiment notes.
        """
        self._notes = new_notes
        if self.auto_save:
            self.save_metadata()

    @property
    def service(self) -> Optional[DatabaseServiceV1]:
        """Return the database service.

        Returns:
            Service that can be used to access this experiment in a database.
        """
        return self._service

    @service.setter
    def service(self, service: DatabaseServiceV1) -> None:
        """Set the service to be used for storing experiment data

        Args:
            service: Service to be used.

        Raises:
            DbExperimentDataError: If an experiment service is already being used.
        """
        self._set_service(service)

    def _set_service(self, service: DatabaseServiceV1) -> None:
        """Set the service to be used for storing experiment data,
           to this experiment only and not to its descendants

        Args:
            service: Service to be used.

        Raises:
            DbExperimentDataError: If an experiment service is already being used.
        """
        if self._service:
            raise DbExperimentDataError("An experiment service is already being used.")
        self._service = service
        for result in self._analysis_results.values():
            result.service = service
        with contextlib.suppress(Exception):
            self.auto_save = self._service.options.get("auto_save", False)

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

    @property
    def source(self) -> Dict:
        """Return the class name and version."""
        return self._source

    def __repr__(self):
        out = f"{type(self).__name__}({self.experiment_type}"
        out += f", {self.experiment_id}"
        if self._parent_id:
            out += f", parent_id={self._parent_id}"
        if self._tags:
            out += f", tags={self._tags}"
        if self.job_ids:
            out += f", job_ids={self.job_ids}"
        if self._share_level:
            out += f", share_level={self._share_level}"
        if self._metadata:
            out += f", metadata=<{len(self._metadata)} items>"
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
        json_value = {}
        for att in [
            "_metadata",
            "_source",
            "_service",
            "_backend",
            "_id",
            "_parent_id",
            "_type",
            "_tags",
            "_share_level",
            "_notes",
            "_data",
            "_analysis_results",
            "_analysis_callbacks",
            "_deleted_figures",
            "_deleted_analysis_results",
            "_created_in_db",
            "_extra_data",
        ]:
            value = getattr(self, att)
            if value:
                json_value[att] = value

        # Convert figures to SVG
        json_value["_figures"] = self._safe_serialize_figures()

        # Handle non-serializable objects
        json_value["_jobs"] = self._safe_serialize_jobs()

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
        for key in ["_job_futures", "_analysis_futures", "_analysis_executor"]:
            del state[key]

        # Convert figures to SVG
        state["_figures"] = self._safe_serialize_figures()

        # Handle partially pickleable attributes
        state["_jobs"] = self._safe_serialize_jobs()

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Initialize non-pickled attributes
        self._job_futures = ThreadSafeOrderedDict()
        self._analysis_futures = ThreadSafeOrderedDict()
        self._analysis_executor = futures.ThreadPoolExecutor(max_workers=1)
