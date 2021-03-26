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

from qiskit.result.models import ExperimentResultData, ExperimentResult
from qiskit.result import Result
from qiskit.test import QiskitTestCase
from qiskit.qobj.utils import MeasLevel
from qiskit.qobj.common import QobjExperimentHeader
from qiskit_experiments import ExperimentData
from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.nodes import (Kernel, Discriminator,
                                                      ToReal, ToImag, Population)


class FakeKernel:
    """Fake kernel to test the data chain."""

    def kernel(self, data):
        """Fake kernel method"""
        return data

class FakeExperiment(BaseExperiment):
    """Fake experiment class for testing."""

    def __init__(self):
        """Initialise the fake experiment."""
        self._type = None
        super().__init__((0, ), 'fake_test_experiment')

    def circuits(self, backend=None, **circuit_options):
        """Fake circuits."""
        return []


class DataProcessorTest(QiskitTestCase):
    """Class to test DataProcessor."""

    def setUp(self):
        """Setup variables used for testing."""
        self.base_result_args = dict(backend_name='test_backend',
                                     backend_version='1.0.0',
                                     qobj_id='id-123',
                                     job_id='job-123',
                                     success=True)

        mem1 = ExperimentResultData(memory=[[[1103260.0, -11378508.0], [2959012.0, -16488753.0]],
                                            [[442170.0, -19283206.0], [-5279410.0, -15339630.0]],
                                            [[3016514.0, -14548009.0], [-3404756.0, -16743348.0]]])

        mem2 = ExperimentResultData(memory=[[[5131962.0, -16630257.0], [4438870.0, -13752518.0]],
                                            [[3415985.0, -16031913.0], [2942458.0, -15840465.0]],
                                            [[5199964.0, -14955998.0], [4030843.0, -14538923.0]]])

        header1 = QobjExperimentHeader(clbit_labels=[['meas', 0], ['meas', 1]],
                                       creg_sizes=[['meas', 2]], global_phase=0.0, memory_slots=2,
                                       metadata={'experiment_type': 'fake_test_experiment',
                                                 'x_values': 0.0})

        header2 = QobjExperimentHeader(clbit_labels=[['meas', 0], ['meas', 1]],
                                       creg_sizes=[['meas', 2]], global_phase=0.0, memory_slots=2,
                                       metadata={'experiment_type': 'fake_test_experiment',
                                                 'x_values': 1.0})

        res1 = ExperimentResult(shots=3, success=True, meas_level=1, data=mem1, header=header1)
        res2 = ExperimentResult(shots=3, success=True, meas_level=1, data=mem2, header=header2)

        self.result_lvl1 = Result(results=[res1, res2], **self.base_result_args)

        super().setUp()

    def test_empty_processor(self):
        """Check that a DataProcessor without steps does nothing."""

        raw_counts = {'0x0': 4, '0x2': 5}
        data = ExperimentResultData(counts=dict(**raw_counts))
        header = QobjExperimentHeader(metadata={'experiment_type': 'fake_test_experiment'})
        result_ = ExperimentResult(shots=9, success=True, meas_level=2, data=data, header=header)

        result = Result(results=[result_], **self.base_result_args)

        exp_data = ExperimentData(FakeExperiment())
        exp_data.add_data(result)

        data_processor = DataProcessor()
        data_processor.format_data(exp_data.data)
        self.assertEqual(exp_data.data[0]['counts']['0'], 4)
        self.assertEqual(exp_data.data[0]['counts']['10'], 5)

    def test_append_kernel(self):
        """Tests that we can add a kernel and a discriminator."""
        processor = DataProcessor()
        self.assertEqual(processor.meas_level(), MeasLevel.RAW)

        processor.append(Kernel(FakeKernel))
        self.assertEqual(processor.meas_level(), MeasLevel.KERNELED)

        processor.append(Discriminator(None))
        self.assertEqual(processor.meas_level(), MeasLevel.CLASSIFIED)

    def test_output_key(self):
        """Test that we can properly get the output key from the node."""
        processor = DataProcessor()
        self.assertEqual(processor.output_key(), 'counts')

        processor.append(Kernel(FakeKernel()))
        self.assertEqual(processor.output_key(), 'memory')

        processor.append(ToReal())
        self.assertEqual(processor.output_key(), 'memory')

        processor = DataProcessor()
        processor.append(Kernel(FakeKernel()))
        processor.append(Discriminator(None))
        self.assertEqual(processor.output_key(), 'counts')

        processor = DataProcessor()
        processor.append(Population())
        self.assertEqual(processor.output_key(), 'populations')

    def test_to_real(self):
        """Test scaling and conversion to real part."""
        processor = DataProcessor()
        processor.append(ToReal(scale=1e-3))
        self.assertEqual(processor.output_key(), 'memory')

        exp_data = ExperimentData(FakeExperiment())
        exp_data.add_data(self.result_lvl1)

        processor.format_data(exp_data.data[0])

        expected = {'memory': [[1103.26, 2959.012], [442.17, -5279.41], [3016.514, -3404.7560]],
                    'metadata': {'experiment_type': 'fake_test_experiment', 'x_values': 0.0}}

        self.assertEqual(exp_data.data[0], expected)

        # Test that we can average single-shots
        processor = DataProcessor()
        processor.append(ToReal(scale=1e-3, average=True))
        self.assertEqual(processor.output_key(), 'memory')

        exp_data = ExperimentData(FakeExperiment())
        exp_data.add_data(self.result_lvl1)

        processor.format_data(exp_data.data[0])

        expected = {'memory': [1520.6480000000001, -1908.3846666666666],
                    'metadata': {'experiment_type': 'fake_test_experiment', 'x_values': 0.0}}

        self.assertEqual(exp_data.data[0], expected)

    def test_to_imag(self):
        """Test that we can average the data."""
        processor = DataProcessor()
        processor.append(ToImag(scale=1e-3))
        self.assertEqual(processor.output_key(), 'memory')

        exp_data = ExperimentData(FakeExperiment())
        exp_data.add_data(self.result_lvl1)

        processor.format_data(exp_data.data[0])

        expected = {'memory': [[-11378.508, -16488.753],
                               [-19283.206000000002, -15339.630000000001],
                               [-14548.009, -16743.348]],
                    'metadata': {'experiment_type': 'fake_test_experiment', 'x_values': 0.0}}

        self.assertEqual(exp_data.data[0], expected)

        # Test that we can average single-shots
        processor = DataProcessor()
        processor.append(ToImag(scale=1e-3, average=True))
        self.assertEqual(processor.output_key(), 'memory')

        exp_data = ExperimentData(FakeExperiment())
        exp_data.add_data(self.result_lvl1)

        processor.format_data(exp_data.data[0])

        expected = {'memory': [-15069.907666666666, -16190.577],
                    'metadata': {'experiment_type': 'fake_test_experiment', 'x_values': 0.0}}

        self.assertEqual(exp_data.data[0], expected)
