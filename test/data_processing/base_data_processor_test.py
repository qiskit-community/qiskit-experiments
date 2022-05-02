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
from test.base import QiskitExperimentsTestCase
from test.fake_experiment import FakeExperiment

from typing import Any, List

from qiskit.result import Result

from qiskit.qobj.common import QobjExperimentHeader
from qiskit.result.models import ExperimentResultData, ExperimentResult
from qiskit_experiments.framework import ExperimentData


class BaseDataProcessorTest(QiskitExperimentsTestCase):
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

    def create_experiment_data(self, iq_data: List[Any], single_shot: bool = False):
        """Populate avg_iq_data to use it for testing.

        Args:
            iq_data: A List of IQ data.
            single_shot: Indicates if the data is single-shot or not.
        """
        results = []
        if not single_shot:
            for circ_data in iq_data:
                res = ExperimentResult(
                    success=True,
                    meas_level=1,
                    meas_return="avg",
                    data=ExperimentResultData(memory=circ_data),
                    header=self.header,
                    shots=1024,
                )
                results.append(res)
        else:
            for circ_data in iq_data:
                res = ExperimentResult(
                    success=True,
                    meas_level=1,
                    meas_return="single",
                    data=ExperimentResultData(memory=circ_data),
                    header=self.header,
                    shots=1024,
                )
                results.append(res)

        # pylint: disable=attribute-defined-outside-init
        self.iq_experiment = ExperimentData(FakeExperiment())
        self.iq_experiment.add_data(Result(results=results, **self.base_result_args))
