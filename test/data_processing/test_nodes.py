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

from typing import Any, List
import numpy as np

from qiskit.result.models import ExperimentResultData, ExperimentResult
from qiskit.result import Result
from qiskit_experiments.experiment_data import ExperimentData
from qiskit_experiments.data_processing.nodes import SVDAvg
from test.data_processing.fake_experiment import FakeExperiment, BaseDataProcessorTest


class TestSVD(BaseDataProcessorTest):
    """Test the SVD nodes."""

    def setUp(self):
        """Setup experiment data."""
        super().setUp()

    def create_experiment(self, iq_data: List[Any]):
        """Populate avg_iq_data to use it for testing.

        Args:
            iq_data: A List of IQ data.
        """

        results = []
        for circ_data in iq_data:
            res = ExperimentResult(
                success=True,
                meas_level=1,
                data=ExperimentResultData(memory=circ_data),
                header=self.header,
                shots=1024
            )
            results.append(res)

        self.avg_iq_data = ExperimentData(FakeExperiment())
        self.avg_iq_data.add_data(Result(results=results, **self.base_result_args))

    def test_simple_data(self):
        """
        A simple setting where the IQ data of qubit 0 is oriented along (1,1) and
        the IQ data of qubit 1 is oriented along (1,-1).
        """

        iq_data = [
            [[0., 0.], [0., 0.]],
            [[1., 1.], [-1., 1.]],
            [[-1., -1.], [1., -1.]]
        ]

        self.create_experiment(iq_data)

        print([datum["memory"] for datum in self.avg_iq_data.data])

        iq_svd = SVDAvg(validate=False)
        iq_svd.train([datum["memory"] for datum in self.avg_iq_data.data])

        # qubit 0 IQ data is oriented along (1,1)
        self.assertTrue(np.allclose(iq_svd._main_axes[0], np.array([1,1]) / np.sqrt(2)))

        # qubit 1 IQ data is oriented along (1, -1)
        self.assertTrue(np.allclose(iq_svd._main_axes[1], np.array([1, -1]) / np.sqrt(2)))

        processed = iq_svd(np.array([[1,1], [1, -1]]))
        expected = np.array([1,1])/np.sqrt(2)
        self.assertTrue(np.allclose(processed, expected))

        processed = iq_svd(np.array([[2,2], [2, -2]]))
        self.assertTrue(np.allclose(processed, expected*2))

        # Check that orthogonal data gives 0.
        processed = iq_svd(np.array([[1, -1], [1, 1]]))
        expected = np.array([0,0])
        self.assertTrue(np.allclose(processed, expected))

    def test_svd(self):
        """Use IQ data gathered from the hardware."""

        # This data is primarily oriented along the real axis with a slight tilt.
        # The is a large offset in the imaginary dimension when comparing qubits
        # 0 and 1.
        iq_data = [
            [[-6.20601501e+14, -1.33257051e+15], [-1.70921324e+15, -4.05881657e+15]],
            [[-5.80546502e+14, -1.33492509e+15], [-1.65094637e+15, -4.05926942e+15]],
            [[-4.04649069e+14, -1.33191056e+15], [-1.29680377e+15, -4.03604815e+15]],
            [[-2.22203874e+14, -1.30291309e+15], [-8.57663429e+14, -3.97784973e+15]],
            [[-2.92074029e+13, -1.28578530e+15], [-9.78824053e+13, -3.92071056e+15]],
            [[1.98056981e+14, -1.26883024e+15], [3.77157017e+14, -3.87460328e+15]],
            [[4.29955888e+14, -1.25022995e+15], [1.02340118e+15, -3.79508679e+15]],
            [[6.38981344e+14, -1.25084614e+15], [1.68918514e+15, -3.78961044e+15]],
            [[7.09988897e+14, -1.21906634e+15], [1.91914171e+15, -3.73670664e+15]],
            [[7.63169115e+14, -1.20797552e+15], [2.03772603e+15, -3.74653863e+15]]
        ]

        self.create_experiment(iq_data)

        iq_svd = SVDAvg(validate=False)
        iq_svd.train([datum["memory"] for datum in self.avg_iq_data.data])

        self.assertTrue(np.allclose(iq_svd._main_axes[0], np.array([0.99633018, 0.08559302])))
        self.assertTrue(np.allclose(iq_svd._main_axes[1], np.array([0.99627747, 0.0862044])))
