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

# pylint: disable=unbalanced-tuple-unpacking

from test.data_processing.fake_experiment import FakeExperiment, BaseDataProcessorTest

from typing import Any, List
import numpy as np

from qiskit.result.models import ExperimentResultData, ExperimentResult
from qiskit.result import Result
from qiskit.test import QiskitTestCase
from qiskit_experiments.experiment_data import ExperimentData
from qiskit_experiments.data_processing.nodes import SVD, AverageData
from qiskit_experiments.data_processing.data_processor import DataProcessor


class TestAveraging(QiskitTestCase):
    """Test the averaging nodes."""

    def test_simple(self):
        """Simple test of averaging."""

        datum = np.array([[1, 2], [3, 4]])

        node = AverageData(axis=1)
        self.assertTrue(np.allclose(node(datum)[0], np.array([1.5, 3.5])))
        self.assertTrue(np.allclose(node(datum)[1], np.array([0.5, 0.5]) / np.sqrt(2)))

        node = AverageData(axis=0)
        self.assertTrue(np.allclose(node(datum)[0], np.array([2.0, 3.0])))
        self.assertTrue(np.allclose(node(datum)[1], np.array([1.0, 1.0]) / np.sqrt(2)))


class TestSVD(BaseDataProcessorTest):
    """Test the SVD nodes."""

    def create_experiment(self, iq_data: List[Any], single_shot: bool = False):
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
            res = ExperimentResult(
                success=True,
                meas_level=1,
                meas_return="single",
                data=ExperimentResultData(memory=iq_data),
                header=self.header,
                shots=1024,
            )
            results.append(res)

        # pylint: disable=attribute-defined-outside-init
        self.iq_experiment = ExperimentData(FakeExperiment())
        self.iq_experiment.add_data(Result(results=results, **self.base_result_args))

    def test_simple_data(self):
        """
        A simple setting where the IQ data of qubit 0 is oriented along (1,1) and
        the IQ data of qubit 1 is oriented along (1,-1).
        """

        iq_data = [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [-1.0, 1.0]], [[-1.0, -1.0], [1.0, -1.0]]]

        self.create_experiment(iq_data)

        iq_svd = SVD()
        iq_svd.train([datum["memory"] for datum in self.iq_experiment.data()])

        # qubit 0 IQ data is oriented along (1,1)
        self.assertTrue(np.allclose(iq_svd._main_axes[0], np.array([-1, -1]) / np.sqrt(2)))

        # qubit 1 IQ data is oriented along (1, -1)
        self.assertTrue(np.allclose(iq_svd._main_axes[1], np.array([-1, 1]) / np.sqrt(2)))

        processed, _ = iq_svd(np.array([[1, 1], [1, -1]]))
        expected = np.array([-1, -1]) / np.sqrt(2)
        self.assertTrue(np.allclose(processed, expected))

        processed, _ = iq_svd(np.array([[2, 2], [2, -2]]))
        self.assertTrue(np.allclose(processed, expected * 2))

        # Check that orthogonal data gives 0.
        processed, _ = iq_svd(np.array([[1, -1], [1, 1]]))
        expected = np.array([0, 0])
        self.assertTrue(np.allclose(processed, expected))

    def test_svd(self):
        """Use IQ data gathered from the hardware."""

        # This data is primarily oriented along the real axis with a slight tilt.
        # The is a large offset in the imaginary dimension when comparing qubits
        # 0 and 1.
        iq_data = [
            [[-6.20601501e14, -1.33257051e15], [-1.70921324e15, -4.05881657e15]],
            [[-5.80546502e14, -1.33492509e15], [-1.65094637e15, -4.05926942e15]],
            [[-4.04649069e14, -1.33191056e15], [-1.29680377e15, -4.03604815e15]],
            [[-2.22203874e14, -1.30291309e15], [-8.57663429e14, -3.97784973e15]],
            [[-2.92074029e13, -1.28578530e15], [-9.78824053e13, -3.92071056e15]],
            [[1.98056981e14, -1.26883024e15], [3.77157017e14, -3.87460328e15]],
            [[4.29955888e14, -1.25022995e15], [1.02340118e15, -3.79508679e15]],
            [[6.38981344e14, -1.25084614e15], [1.68918514e15, -3.78961044e15]],
            [[7.09988897e14, -1.21906634e15], [1.91914171e15, -3.73670664e15]],
            [[7.63169115e14, -1.20797552e15], [2.03772603e15, -3.74653863e15]],
        ]

        self.create_experiment(iq_data)

        iq_svd = SVD()
        iq_svd.train([datum["memory"] for datum in self.iq_experiment.data()])

        self.assertTrue(np.allclose(iq_svd._main_axes[0], np.array([-0.99633018, -0.08559302])))
        self.assertTrue(np.allclose(iq_svd._main_axes[1], np.array([-0.99627747, -0.0862044])))

    def test_svd_error(self):
        """Test the error formula of the SVD."""

        iq_svd = SVD()
        iq_svd._main_axes = np.array([[1.0, 0.0]])
        iq_svd._scales = [1.0]
        iq_svd._means = [[0.0, 0.0]]

        # Since the axis is along the real part the imaginary error is irrelevant.
        processed, error = iq_svd([[1.0, 0.2]], [[0.2, 0.1]])
        self.assertEqual(processed, np.array([1.0]))
        self.assertEqual(error, np.array([0.2]))

        # Since the axis is along the real part the imaginary error is irrelevant.
        processed, error = iq_svd([[1.0, 0.2]], [[0.2, 0.3]])
        self.assertEqual(processed, np.array([1.0]))
        self.assertEqual(error, np.array([0.2]))

        # Title the axis to an angle of 36.9... degrees
        iq_svd._main_axes = np.array([[0.8, 0.6]])
        processed, error = iq_svd([[1.0, 0.0]], [[0.2, 0.3]])
        cos_ = np.cos(np.arctan(0.6 / 0.8))
        sin_ = np.sin(np.arctan(0.6 / 0.8))
        self.assertEqual(processed, np.array([cos_]))
        expected_error = np.sqrt((0.2 * cos_) ** 2 + (0.3 * sin_) ** 2)
        self.assertEqual(error, np.array([expected_error]))

    def test_train_svd_processor(self):
        """Test that we can train a DataProcessor with an SVD."""

        processor = DataProcessor("memory", [SVD()])

        self.assertFalse(processor.is_trained)

        iq_data = [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [-1.0, 1.0]], [[-1.0, -1.0], [1.0, -1.0]]]
        self.create_experiment(iq_data)

        processor.train(self.iq_experiment.data())

        self.assertTrue(processor.is_trained)

        # Check that we can use the SVD
        iq_data = [[[2, 2], [2, -2]]]
        self.create_experiment(iq_data)

        processed, _ = processor(self.iq_experiment.data(0))
        expected = np.array([-2, -2]) / np.sqrt(2)
        self.assertTrue(np.allclose(processed, expected))

    def test_iq_averaging(self):
        """Test averaging of IQ-data."""

        iq_data = [
            [[-6.20601501e14, -1.33257051e15], [-1.70921324e15, -4.05881657e15]],
            [[-5.80546502e14, -1.33492509e15], [-1.65094637e15, -4.05926942e15]],
            [[-4.04649069e14, -1.33191056e15], [-1.29680377e15, -4.03604815e15]],
            [[-2.22203874e14, -1.30291309e15], [-8.57663429e14, -3.97784973e15]],
            [[-2.92074029e13, -1.28578530e15], [-9.78824053e13, -3.92071056e15]],
            [[1.98056981e14, -1.26883024e15], [3.77157017e14, -3.87460328e15]],
            [[4.29955888e14, -1.25022995e15], [1.02340118e15, -3.79508679e15]],
            [[6.38981344e14, -1.25084614e15], [1.68918514e15, -3.78961044e15]],
            [[7.09988897e14, -1.21906634e15], [1.91914171e15, -3.73670664e15]],
            [[7.63169115e14, -1.20797552e15], [2.03772603e15, -3.74653863e15]],
        ]

        self.create_experiment(iq_data, single_shot=True)

        avg_iq = AverageData()

        avg_datum, error = avg_iq(self.iq_experiment.data(0)["memory"])

        expected_avg = np.array([[8.82943876e13, -1.27850527e15], [1.43410186e14, -3.89952402e15]])

        expected_std = np.array(
            [[5.07650185e14, 4.44664719e13], [1.40522641e15, 1.22326831e14]]
        ) / np.sqrt(10)

        self.assertTrue(np.allclose(avg_datum, expected_avg))
        self.assertTrue(np.allclose(error, expected_std))
