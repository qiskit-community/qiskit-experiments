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

from qiskit_experiments import ExperimentData
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from test.data_processing.fake_experiment import FakeExperiment, BaseDataProcessorTest
from qiskit_experiments.data_processing.nodes import (
    ToReal,
    ToImag,
    Probability,
)


class DataProcessorTest(BaseDataProcessorTest):
    """Class to test DataProcessor."""

    def setUp(self):
        """Setup variables used for testing."""
        super().setUp()

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

        res1 = ExperimentResult(shots=3, success=True, meas_level=1, data=mem1, header=self.header)
        res2 = ExperimentResult(shots=3, success=True, meas_level=1, data=mem2, header=self.header)

        self.result_lvl1 = Result(results=[res1, res2], **self.base_result_args)

        raw_counts = {"0x0": 4, "0x2": 6}
        data = ExperimentResultData(counts=dict(**raw_counts))
        res = ExperimentResult(shots=9, success=True, meas_level=2, data=data, header=self.header)
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
            "metadata": {"experiment_type": "fake_test_experiment"},
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
            "metadata": {"experiment_type": "fake_test_experiment"},
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
