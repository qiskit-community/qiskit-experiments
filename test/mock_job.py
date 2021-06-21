# -*- coding: utf-8 -*-

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
Mock Job class for test backends
"""
import uuid
from qiskit.providers import JobV1 as Job
from qiskit.providers import JobStatus
from qiskit.result import Result


class MockJob(Job):
    """Mock Job class for tests"""

    def __init__(self, backend, result):
        if isinstance(result, dict):
            self._result = Result.from_dict(result)
        else:
            self._result = result
        super().__init__(backend, str(uuid.uuid4()))

    def submit(self):
        """Submit the job to the backend for execution."""
        pass

    def result(self):
        """Return the results of the job."""
        return self._result

    def cancel(self):
        """Attempt to cancel the job."""
        pass

    def status(self):
        """Return the status of the job, among the values of ``JobStatus``."""
        return JobStatus.DONE
