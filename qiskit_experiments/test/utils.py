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

"""Test utility functions."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from qiskit.providers.job import JobV1 as Job
from qiskit.providers.jobstatus import JobStatus
from qiskit.providers.backend import BackendV2 as Backend
from qiskit.result import Result


if TYPE_CHECKING:
    from qiskit_experiments.framework import Job as JobLike


class FakeProvider:
    """Dummy Provider class for test purposes only"""

    def __init__(self):
        self._jobs: dict[str, JobLike] = {}

    def add_job(self, job: JobLike):
        """Add job to provider"""
        self._jobs[job.job_id()] = job

    def job(self, job_id: str) -> JobLike:
        """Retrieve job by job ID"""
        return self._jobs[job_id]


class FakeJob(Job):
    """Fake job."""

    def __init__(
        self,
        backend: Backend,
        result: Result | None = None,
        status: JobStatus | None = None,
        cancel_callback: Callable | None = None,
        result_callback: Callable | None = None,
        status_callback: Callable | None = None,
    ):
        """Initialize FakeJob."""
        if result:
            job_id = result.job_id
        else:
            job_id = uuid.uuid4().hex
        super().__init__(backend, job_id)
        self._result = result
        self._status = status

        self._cancel_callback = cancel_callback
        self._result_callback = result_callback
        self._status_callback = status_callback

    def result(self):
        """Return job result."""
        if self._result_callback:
            return self._result_callback()
        return self._result

    def cancel(self):
        """Cancel the job"""
        self._status = JobStatus.CANCELLED
        if self._cancel_callback:
            self._cancel_callback()

    def submit(self):
        """Submit the job to the backend for execution."""
        pass

    @staticmethod
    def time_per_step() -> dict[str, datetime]:
        """Return the completion time."""
        return {"COMPLETED": datetime.now(timezone.utc)}

    def status(self) -> JobStatus:
        """Return the status of the job, among the values of ``JobStatus``."""
        if self._status_callback:
            return self._status_callback()

        if self._status is not None:
            return self._status
        if self._result:
            return JobStatus.DONE
        return JobStatus.RUNNING
