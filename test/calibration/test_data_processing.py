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

"""Data processor tests."""

from qiskit_experiments.calibration import DataProcessor
from qiskit_experiments.calibration.metadata import CalibrationMetadata
from qiskit_experiments.calibration.data_processing import SystemKernel, SystemDiscriminator
from qiskit.result import models
from qiskit.result import Result
from qiskit.test import QiskitTestCase
from qiskit.qobj.utils import MeasLevel


class DataProcessorTest(QiskitTestCase):
    """Class to test DataProcessor."""

    def setUp(self):
        self.base_result_args = dict(backend_name='test_backend',
                                     backend_version='1.0.0',
                                     qobj_id='id-123',
                                     job_id='job-123',
                                     success=True)

        super().setUp()

    def test_empty_processor(self):
        """Check that a DataProcessor without steps does nothing."""

        raw_counts = {'0x0': 4, '0x2': 10}
        data = models.ExperimentResultData(counts=dict(**raw_counts))
        exp_result = models.ExperimentResult(shots=14, success=True, meas_level=2, data=data)
        result = Result(results=[exp_result], **self.base_result_args)

        data_processor = DataProcessor()
        processed = data_processor.format_data(result, metadata=CalibrationMetadata(), index=0)
        self.assertEqual(processed.get_counts(0)['0'], 4)
        self.assertEqual(processed.get_counts(0)['10'], 10)

    def test_append_kernel(self):
        """Tests that we can add a kernel and a discriminator."""
        processor = DataProcessor()
        self.assertEqual(processor.meas_level(), MeasLevel.RAW)

        processor.append(SystemKernel())
        self.assertEqual(processor.meas_level(), MeasLevel.KERNELED)

        processor.append(SystemDiscriminator(None))
        self.assertEqual(processor.meas_level(), MeasLevel.CLASSIFIED)
