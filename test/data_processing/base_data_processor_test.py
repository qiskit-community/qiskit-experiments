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

"""Base class for data processor tests."""

from qiskit.test import QiskitTestCase
from qiskit.qobj.common import QobjExperimentHeader


class BaseDataProcessorTest(QiskitTestCase):
    """Define some basic setup functionality for data processor tests."""

    def setUp(self):
        """Define variables needed for most tests."""
        super().setUp()

        self.base_result_args = dict(
            backend_name="test_backend",
            backend_version="1.0.0",
            qobj_id="id-123",
            job_id="job-123",
            success=True,
        )

        self.header = QobjExperimentHeader(
            memory_slots=2,
            metadata={"experiment_type": "fake_test_experiment"},
        )
