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

import uuid
from typing import Optional, Dict
from datetime import datetime, timezone

from qiskit.providers.job import JobV1 as Job
from qiskit.providers.jobstatus import JobStatus
from qiskit.providers.backend import BackendV1 as Backend
from qiskit.result import Result


class FakeJob(Job):
    """Fake job."""

    def __init__(self, backend: Backend, result: Optional[Result] = None):
        """Initialize FakeJob."""
        if result:
            job_id = result.job_id
        else:
            job_id = uuid.uuid4().hex
        super().__init__(backend, job_id)
        self._result = result

    def result(self):
        """Return job result."""
        return self._result

    def submit(self):
        """Submit the job to the backend for execution."""
        pass

    @staticmethod
    def time_per_step() -> Dict[str, datetime]:
        """Return the completion time."""
        return {"COMPLETED": datetime.now(timezone.utc)}

    def status(self) -> JobStatus:
        """Return the status of the job, among the values of ``JobStatus``."""
        if self._result:
            return JobStatus.DONE
        return JobStatus.RUNNING
