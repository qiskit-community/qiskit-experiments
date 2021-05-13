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

import numpy as np

from qiskit.result.models import ExperimentResultData, ExperimentResult
from qiskit.result import Result
from qiskit.test import QiskitTestCase
from qiskit.qobj.common import QobjExperimentHeader
from qiskit_experiments import ExperimentData
from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.data_processing.nodes import (
    ToReal,
    ToRealAvg,
    ToImag,
    ToImagAvg,
    Probability,
)


class FakeExperiment(BaseExperiment):
    """Fake experiment class for testing."""

    def __init__(self):
        """Initialise the fake experiment."""
        self._type = None
        super().__init__((0,), "fake_test_experiment")

    def circuits(self, backend=None, **circuit_options):
        """Fake circuits."""
        return []


class DataProcessorTest(QiskitTestCase):
    """Class to test DataProcessor."""

    def setUp(self):
        """Setup variables used for testing."""
        self.base_result_args = dict(
            backend_name="test_backend",
            backend_version="1.0.0",
            qobj_id="id-123",
            job_id="job-123",
            success=True,
        )

        mem1 = ExperimentResultData(
            memory=[
                [[1103260.0, -11378508.0], [2959012.0, -16488753.0]],
                [[442170.0, -19283206.0], [-5279410.0, -15339630.0]],
                [[3016514.0, -14548009.0], [-3404756.0, -16743348.0]],
            ]
        )

        mem2 = ExperimentResultData(
            memory=[
                [[5131962.0, -16630257.0], [4438870.0, -13752518.0]],
                [[3415985.0, -16031913.0], [2942458.0, -15840465.0]],
                [[5199964.0, -14955998.0], [4030843.0, -14538923.0]],
            ]
        )

        header1 = QobjExperimentHeader(
            clbit_labels=[["meas", 0], ["meas", 1]],
            creg_sizes=[["meas", 2]],
            global_phase=0.0,
            memory_slots=2,
            metadata={"experiment_type": "fake_test_experiment", "x_values": 0.0},
        )

        header2 = QobjExperimentHeader(
            clbit_labels=[["meas", 0], ["meas", 1]],
            creg_sizes=[["meas", 2]],
            global_phase=0.0,
            memory_slots=2,
            metadata={"experiment_type": "fake_test_experiment", "x_values": 1.0},
        )

        res1 = ExperimentResult(shots=3, success=True, meas_level=1, data=mem1, header=header1)
        res2 = ExperimentResult(shots=3, success=True, meas_level=1, data=mem2, header=header2)

        self.result_lvl1 = Result(results=[res1, res2], **self.base_result_args)

        raw_counts = {"0x0": 4, "0x2": 6}
        data = ExperimentResultData(counts=dict(**raw_counts))
        header = QobjExperimentHeader(
            metadata={"experiment_type": "fake_test_experiment"},
            clbit_labels=[["c", 0], ["c", 1]],
            creg_sizes=[["c", 2]],
            n_qubits=2,
            memory_slots=2,
        )
        res = ExperimentResult(shots=9, success=True, meas_level=2, data=data, header=header)
        self.exp_data_lvl2 = ExperimentData(FakeExperiment())
        self.exp_data_lvl2.add_data(Result(results=[res], **self.base_result_args))

        super().setUp()

    def test_empty_processor(self):
        """Check that a DataProcessor without steps does nothing."""
        data_processor = DataProcessor("counts")

        datum = data_processor(self.exp_data_lvl2.data[0])
        self.assertEqual(datum, {"00": 4, "10": 6})

        datum, history = data_processor.call_with_history(self.exp_data_lvl2.data[0])
        self.assertEqual(datum, {"00": 4, "10": 6})
        self.assertEqual(history, [])

    def test_to_real(self):
        """Test scaling and conversion to real part."""
        processor = DataProcessor("memory", [ToReal(scale=1e-3)])

        exp_data = ExperimentData(FakeExperiment())
        exp_data.add_data(self.result_lvl1)

        new_data = processor(exp_data.data[0])

        expected_old = {
            "memory": [
                [[1103260.0, -11378508.0], [2959012.0, -16488753.0]],
                [[442170.0, -19283206.0], [-5279410.0, -15339630.0]],
                [[3016514.0, -14548009.0], [-3404756.0, -16743348.0]],
            ],
            "metadata": {"experiment_type": "fake_test_experiment", "x_values": 0.0},
        }

        expected_new = np.array([[1103.26, 2959.012], [442.17, -5279.41], [3016.514, -3404.7560]])

        self.assertEqual(exp_data.data[0], expected_old)
        self.assertTrue(np.allclose(new_data, expected_new))

        # Test that we can call with history.
        new_data, history = processor.call_with_history(exp_data.data[0])

        self.assertEqual(exp_data.data[0], expected_old)
        self.assertTrue(np.allclose(new_data, expected_new))

        self.assertEqual(history[0][0], "ToReal")
        self.assertTrue(np.allclose(history[0][1], expected_new))

    def test_to_imag(self):
        """Test that we can average the data."""
        processor = DataProcessor("memory")
        processor.append(ToImag(scale=1e-3))

        exp_data = ExperimentData(FakeExperiment())
        exp_data.add_data(self.result_lvl1)

        new_data = processor(exp_data.data[0])

        expected_old = {
            "memory": [
                [[1103260.0, -11378508.0], [2959012.0, -16488753.0]],
                [[442170.0, -19283206.0], [-5279410.0, -15339630.0]],
                [[3016514.0, -14548009.0], [-3404756.0, -16743348.0]],
            ],
            "metadata": {"experiment_type": "fake_test_experiment", "x_values": 0.0},
        }

        expected_new = np.array(
            [
                [-11378.508, -16488.753],
                [-19283.206000000002, -15339.630000000001],
                [-14548.009, -16743.348],
            ]
        )

        self.assertEqual(exp_data.data[0], expected_old)
        self.assertTrue(np.allclose(new_data, expected_new))

        # Test that we can call with history.
        new_data, history = processor.call_with_history(exp_data.data[0])
        self.assertEqual(exp_data.data[0], expected_old)
        self.assertTrue(np.allclose(new_data, expected_new))

        self.assertEqual(history[0][0], "ToImag")
        self.assertTrue(np.allclose(history[0][1], expected_new))

    def test_populations(self):
        """Test that counts are properly converted to a population."""

        processor = DataProcessor("counts")
        processor.append(Probability("00"))

        new_data = processor(self.exp_data_lvl2.data[0])

        self.assertEqual(new_data[0], 0.4)
        self.assertEqual(new_data[1], 0.4 * (1 - 0.4) / 10)

    def test_validation(self):
        """Test the validation mechanism."""

        for validate, error in [(False, AttributeError), (True, DataProcessorError)]:
            processor = DataProcessor("counts")
            processor.append(Probability("00", validate=validate))

            with self.assertRaises(error):
                processor({"counts": [0, 1, 2]})


class TestIQSingleAvg(QiskitTestCase):
    """Test the IQ data processing nodes single and average."""

    def setUp(self):
        """Setup some IQ data."""
        super().setUp()

        self.base_result_args = dict(
            backend_name="test_backend",
            backend_version="1.0.0",
            qobj_id="id-123",
            job_id="job-123",
            success=True,
        )

        mem_avg = ExperimentResultData(
            memory=[[-539698.0, -153030784.0], [5541283.0, -160369600.0]]
        )
        mem_single = ExperimentResultData(
            memory=[
                [[-56470872.0, -136691568.0], [-53407256.0, -176278624.0]],
                [[-34623272.0, -151247824.0], [-36650644.0, -170559312.0]],
                [[42658720.0, -153054640.0], [29689970.0, -174671824.0]],
                [[-47387248.0, -177826640.0], [-62149124.0, -165909728.0]],
                [[-51465408.0, -148338000.0], [23157112.0, -165826736.0]],
                [[51426688.0, -142703104.0], [34330920.0, -185572592.0]],
            ]
        )

        header = QobjExperimentHeader(
            metadata={"experiment_type": "fake_test_experiment"},
            clbit_labels=[["c", 0], ["c", 1]],
            creg_sizes=[["c", 2]],
            n_qubits=2,
            memory_slots=2,
        )
        res_single = ExperimentResult(
            shots=3,
            success=True,
            meas_level=1,
            meas_return="single",
            data=mem_single,
            header=header,
        )
        res_avg = ExperimentResult(
            shots=6, success=True, meas_level=1, meas_return="avg", data=mem_avg, header=header
        )

        # result_single = Result(results=[res_single], **self.base_result_args)
        # result_avg = Result(results=[res_avg], **self.base_result_args)

        self.exp_data_single = ExperimentData(FakeExperiment())
        self.exp_data_single.add_data(Result(results=[res_single], **self.base_result_args))

        self.exp_data_avg = ExperimentData(FakeExperiment())
        self.exp_data_avg.add_data(Result(results=[res_avg], **self.base_result_args))

    def test_avg_and_single(self):
        """Test that the different nodes process the data correctly."""

        real_single = DataProcessor("memory", [ToReal(scale=1)])
        imag_single = DataProcessor("memory", [ToImag(scale=1)])
        real_avg = DataProcessor("memory", [ToRealAvg(scale=1)])
        imag_avg = DataProcessor("memory", [ToImagAvg(scale=1)])

        # Test the real single shot node
        new_data = real_single(self.exp_data_single.data[0])
        expected = np.array(
            [
                [-56470872.0, -53407256.0],
                [-34623272.0, -36650644.0],
                [42658720.0, 29689970.0],
                [-47387248.0, -62149124.0],
                [-51465408.0, 23157112.0],
                [51426688.0, 34330920.0],
            ]
        )
        self.assertTrue(np.allclose(new_data, expected))

        with self.assertRaises(DataProcessorError):
            real_single(self.exp_data_avg.data[0])

        # Test the imaginary single shot node
        new_data = imag_single(self.exp_data_single.data[0])
        expected = np.array(
            [
                [-136691568.0, -176278624.0],
                [-151247824.0, -170559312.0],
                [-153054640.0, -174671824.0],
                [-177826640.0, -165909728.0],
                [-148338000.0, -165826736.0],
                [-142703104.0, -185572592.0],
            ]
        )
        self.assertTrue(np.allclose(new_data, expected))

        # Test the real average node
        new_data = real_avg(self.exp_data_avg.data[0])
        self.assertTrue(np.allclose(new_data, np.array([-539698.0, 5541283.0])))

        # Test the imaginary average node
        new_data = imag_avg(self.exp_data_avg.data[0])
        self.assertTrue(np.allclose(new_data, np.array([-153030784.0, -160369600.0])))

        with self.assertRaises(DataProcessorError):
            real_avg(self.exp_data_single.data[0])
