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

import warnings
from typing import Any, Dict, List
from test.base import QiskitExperimentsTestCase

import json
import numpy as np
from uncertainties import unumpy as unp, ufloat

from qiskit_experiments.data_processing.nodes import (
    DiscriminatorNode,
    MemoryToCounts,
    SVD,
    ToAbs,
    AverageData,
    MinMaxNormalize,
    Probability,
    MarginalizeCounts,
    RestlessToCounts,
    RestlessToIQ,
)
from qiskit_experiments.data_processing import DataProcessor
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.data_processing.discriminator import BaseDiscriminator
from qiskit_experiments.framework.json import ExperimentDecoder, ExperimentEncoder
from . import BaseDataProcessorTest


class TestAveraging(BaseDataProcessorTest):
    """Test the averaging nodes."""

    def test_simple(self):
        """Simple test of averaging. Standard error of mean is generated."""
        with np.errstate(invalid="ignore"):
            # Setting std_devs to NaN will trigger floating point exceptions
            # which we can ignore. See https://stackoverflow.com/q/75656026
            datum = unp.uarray([[1, 2], [3, 4], [5, 6]], np.full((3, 2), np.nan))

        node = AverageData(axis=1)
        processed_data = node(data=datum)

        np.testing.assert_array_almost_equal(
            unp.nominal_values(processed_data),
            np.array([1.5, 3.5, 5.5]),
        )
        np.testing.assert_array_almost_equal(
            unp.std_devs(processed_data),
            np.array([0.5, 0.5, 0.5]) / np.sqrt(2),
        )

        node = AverageData(axis=0)
        processed_data = node(data=datum)

        np.testing.assert_array_almost_equal(
            unp.nominal_values(processed_data),
            np.array([3.0, 4.0]),
        )
        np.testing.assert_array_almost_equal(
            unp.std_devs(processed_data),
            np.array([1.632993161855452, 1.632993161855452]) / np.sqrt(3),
        )

    def test_with_error(self):
        """Compute error propagation. This is quadratic sum divided by samples."""
        datum = unp.uarray(
            [[1, 2, 3, 4, 5, 6]],
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
        )

        node = AverageData(axis=1)
        processed_data = node(data=datum)

        self.assertAlmostEqual(processed_data[0].nominal_value, 3.5)
        # sqrt(0.1**2 + 0.2**2 + ... + 0.6**2) / 6
        self.assertAlmostEqual(processed_data[0].std_dev, 0.15898986690282427)

    def test_with_error_partly_non_error(self):
        """Compute error propagation. Some elements have no error."""
        with np.errstate(invalid="ignore"):
            # Setting std_devs to NaN will trigger floating point exceptions
            # which we can ignore. See https://stackoverflow.com/q/75656026
            datum = unp.uarray(
                [
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6],
                ],
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    [np.nan, 0.2, 0.3, 0.4, 0.5, 0.6],
                ],
            )

        node = AverageData(axis=1)
        processed_data = node(data=datum)

        self.assertAlmostEqual(processed_data[0].nominal_value, 3.5)
        # sqrt(0.1**2 + 0.2**2 + ... + 0.6**2) / 6
        self.assertAlmostEqual(processed_data[0].std_dev, 0.15898986690282427)

        self.assertAlmostEqual(processed_data[1].nominal_value, 3.5)
        # sqrt((0.1 - 0.35)**2 + (0.2 - 0.35)**2 + ... + (0.6 - 0.35)**2) / 6
        self.assertAlmostEqual(processed_data[1].std_dev, 0.6972166887783964)

    def test_iq_averaging(self):
        """Test averaging of IQ-data."""

        # This data represents IQ data for a single quantum circuit with 10 shots and 2 slots.
        iq_data = np.array(
            [
                [
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
            ],
            dtype=float,
        )
        iq_std = np.full_like(iq_data, np.nan)

        with np.errstate(invalid="ignore"):
            # Setting std_devs to NaN will trigger floating point exceptions
            # which we can ignore. See https://stackoverflow.com/q/75656026
            self.create_experiment_data(unp.uarray(iq_data, iq_std), single_shot=True)

        avg_iq = AverageData(axis=0)
        processed_data = avg_iq(data=np.asarray(self.iq_experiment.data(0)["memory"]))

        expected_avg = np.array([[8.82943876e13, -1.27850527e15], [1.43410186e14, -3.89952402e15]])
        expected_std = np.array(
            [[5.07650185e14, 4.44664719e13], [1.40522641e15, 1.22326831e14]]
        ) / np.sqrt(10)

        np.testing.assert_array_almost_equal(
            unp.nominal_values(processed_data),
            expected_avg,
            decimal=-8,
        )
        np.testing.assert_array_almost_equal(
            unp.std_devs(processed_data),
            expected_std,
            decimal=-8,
        )

    def test_json(self):
        """Check if the node is serializable."""
        node = AverageData(axis=3)
        self.assertRoundTripSerializable(node)


class TestToAbs(QiskitExperimentsTestCase):
    """Test the ToAbs node."""

    def test_simple(self):
        """Simple test to check the it runs."""

        data = [
            [[ufloat(2.0, np.nan), ufloat(2.0, np.nan)]],
            [[ufloat(1.0, np.nan), ufloat(2.0, np.nan)]],
            [[ufloat(2.0, 0.2), ufloat(3.0, 0.3)]],
        ]

        processed = ToAbs()(np.array(data))

        val = np.sqrt(2**2 + 3**2)
        val_err = np.sqrt(2**2 * 0.2**2 + 2**2 * 0.3**2) / val

        expected = np.array(
            [
                [ufloat(np.sqrt(8), np.nan)],
                [ufloat(np.sqrt(5), np.nan)],
                [ufloat(val, val_err)],
            ]
        )

        np.testing.assert_array_almost_equal(
            unp.nominal_values(processed),
            unp.nominal_values(expected),
            decimal=-8,
        )

        with np.errstate(invalid="ignore"):
            # Setting std_devs to NaN will trigger floating point exceptions
            # which we can ignore. See https://stackoverflow.com/q/75656026
            np.testing.assert_array_almost_equal(
                unp.std_devs(processed),
                unp.std_devs(expected),
                decimal=-8,
            )


class TestNormalize(QiskitExperimentsTestCase):
    """Test the normalization node."""

    def test_simple(self):
        """Simple test of normalization node."""

        data = np.array([1.0, 2.0, 3.0, 3.0])
        error = np.array([0.1, 0.2, 0.3, 0.3])

        expected_data = np.array([0.0, 0.5, 1.0, 1.0])
        expected_error = np.array([0.05, 0.1, 0.15, 0.15])

        node = MinMaxNormalize()

        processed_data = node(data=data)
        np.testing.assert_array_almost_equal(
            unp.nominal_values(processed_data),
            expected_data,
        )

        processed_data = node(data=unp.uarray(nominal_values=data, std_devs=error))
        np.testing.assert_array_almost_equal(
            unp.nominal_values(processed_data),
            expected_data,
        )
        np.testing.assert_array_almost_equal(
            unp.std_devs(processed_data),
            expected_error,
        )

    def test_json(self):
        """Check if the node is serializable."""
        node = MinMaxNormalize()
        self.assertRoundTripSerializable(node)


class TestSVD(BaseDataProcessorTest):
    """Test the SVD nodes."""

    def test_simple_data(self):
        """
        A simple setting where the IQ data of qubit 0 is oriented along (1,1) and
        the IQ data of qubit 1 is oriented along (1,-1).
        """
        iq_data = [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [-1.0, 1.0]], [[-1.0, -1.0], [1.0, -1.0]]]

        self.create_experiment_data(iq_data)

        iq_svd = SVD()
        iq_svd.train(np.asarray([datum["memory"] for datum in self.iq_experiment.data()]))

        # qubit 0 IQ data is oriented along (1,1)
        np.testing.assert_array_almost_equal(
            iq_svd.parameters.main_axes[0], np.array([1, 1]) / np.sqrt(2)
        )

        # qubit 1 IQ data is oriented along (1, -1)
        np.testing.assert_array_almost_equal(
            iq_svd.parameters.main_axes[1], np.array([1, -1]) / np.sqrt(2)
        )

        # This is n_circuit = 1, n_slot = 2, the input shape should be [1, 2, 2]
        # Then the output shape will be [1, 2] by reducing the last dimension
        processed_data = iq_svd(np.array([[[1, 1], [1, -1]]]))
        np.testing.assert_array_almost_equal(
            unp.nominal_values(processed_data),
            np.array([[1, 1]]) / np.sqrt(2),
        )

        processed_data = iq_svd(np.array([[[2, 2], [2, -2]]]))
        np.testing.assert_array_almost_equal(
            unp.nominal_values(processed_data),
            2 * np.array([[1, 1]]) / np.sqrt(2),
        )

        # Check that orthogonal data gives 0.
        processed_data = iq_svd(np.array([[[1, -1], [1, 1]]]))
        np.testing.assert_array_almost_equal(
            unp.nominal_values(processed_data),
            np.array([[0, 0]]),
        )

    def test_svd_on_averaged(self):
        """Use IQ data gathered from the hardware."""
        # This data is primarily oriented along the real axis with a slight tilt.
        # There is a large offset in the imaginary dimension when comparing qubits
        # 0 and 1. The data below is averaged IQ data on two qubits.
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

        self.create_experiment_data(iq_data)

        iq_svd = SVD()
        iq_svd.train(np.asarray([datum["memory"] for datum in self.iq_experiment.data()]))

        np.testing.assert_array_almost_equal(
            iq_svd.parameters.main_axes[0], np.array([0.99633018, 0.08559302])
        )
        np.testing.assert_array_almost_equal(
            iq_svd.parameters.main_axes[1], np.array([0.99627747, 0.0862044])
        )

    def test_on_single_shot(self):
        """Test the SVD node on single shot data."""

        # The data has the shape
        iq_data = [
            # Circuit no. 1, 5 shots
            [
                [[-84858304.0, -111158232.0]],
                [[-92671216.0, -74032944.0]],
                [[-74049176.0, -22372804.0]],
                [[-87495592.0, -72437616.0]],
                [[-52787048.0, -63746976.0]],
            ],
            # Circuit no. 2, 5 shots
            [
                [[-70452328.0, -91318008.0]],
                [[-82281464.0, -72478736.0]],
                [[-107760368.0, -77817680.0]],
                [[-47410012.0, -48451952.0]],
                [[68308432.0, -72074976.0]],
            ],
            # Circuit no. 3, 5 shots
            [
                [[47855768.0, -52185604.0]],
                [[-64009220.0, -79507104.0]],
                [[51899032.0, -80737864.0]],
                [[118873272.0, -43621036.0]],
                [[24438894.0, -84970704.0]],
            ],
        ]

        self.create_experiment_data(iq_data, single_shot=True)

        iq_svd = SVD()
        iq_svd.train(np.asarray([datum["memory"] for datum in self.iq_experiment.data()]))

        processed_data = iq_svd(np.array(iq_data))

        # Test the output of the axis
        self.assertEqual(len(iq_svd.parameters.main_axes), 1)
        self.assertTrue(np.allclose(iq_svd.parameters.main_axes[0], [0.92727304, 0.37438577]))

        # Test the output data
        self.assertEqual(processed_data.shape, (3, 5, 1))
        test_values = np.array(processed_data[0].flatten(), dtype=float)
        expected = np.array(
            [-0.4982860, -0.4383349, -0.10852355, -0.38971727, -0.07045186], dtype=float
        )
        self.assertTrue(np.allclose(test_values, expected, atol=1e-06))

        # Test in a data processor, will catch, e.g., unumpy issues
        data_processor = DataProcessor("memory", [SVD()])
        data_processor.train(self.iq_experiment.data())
        processed_data = data_processor(self.iq_experiment.data())
        self.assertEqual(processed_data.shape, (3, 5, 1))

    def test_svd_error(self):
        """Test the error formula of the SVD."""
        # This is n_circuit = 1, n_slot = 1, the input shape should be [1, 1, 2]
        # Then the output shape will be [1, 1] by reducing the last dimension

        iq_svd = SVD()
        iq_svd.set_parameters(
            main_axes=np.array([[1.0, 0.0]]), scales=[1.0], i_means=[0.0], q_means=[0.0]
        )

        # Since the axis is along the real part the imaginary error is irrelevant.
        processed_data = iq_svd(unp.uarray(nominal_values=[[[1.0, 0.2]]], std_devs=[[[0.2, 0.1]]]))
        np.testing.assert_array_almost_equal(unp.nominal_values(processed_data), np.array([[1.0]]))
        np.testing.assert_array_almost_equal(unp.std_devs(processed_data), np.array([[0.2]]))

        # Since the axis is along the real part the imaginary error is irrelevant.
        processed_data = iq_svd(unp.uarray(nominal_values=[[[1.0, 0.2]]], std_devs=[[[0.2, 0.3]]]))
        np.testing.assert_array_almost_equal(unp.nominal_values(processed_data), np.array([[1.0]]))
        np.testing.assert_array_almost_equal(unp.std_devs(processed_data), np.array([[0.2]]))

        # Tilt the axis to an angle of 36.9... degrees
        iq_svd.set_parameters(main_axes=np.array([[0.8, 0.6]]))

        processed_data = iq_svd(unp.uarray(nominal_values=[[[1.0, 0.0]]], std_devs=[[[0.2, 0.3]]]))
        cos_ = np.cos(np.arctan(0.6 / 0.8))
        sin_ = np.sin(np.arctan(0.6 / 0.8))
        np.testing.assert_array_almost_equal(
            unp.nominal_values(processed_data),
            np.array([[cos_]]),
        )
        np.testing.assert_array_almost_equal(
            unp.std_devs(processed_data),
            np.array([[np.sqrt((0.2 * cos_) ** 2 + (0.3 * sin_) ** 2)]]),
        )

    def test_json(self):
        """Check if the node is serializable."""
        node = SVD()
        self.assertRoundTripSerializable(node)

    def test_json_trained(self):
        """Check if the trained node is serializable."""
        node = SVD()
        node.set_parameters(
            main_axes=np.array([[1.0, 2.0]]), scales=[1.0], i_means=[2.0], q_means=[3.0]
        )
        self.assertRoundTripSerializable(node)

        loaded_node = json.loads(json.dumps(node, cls=ExperimentEncoder), cls=ExperimentDecoder)
        self.assertTrue(loaded_node.is_trained)


class FakeDiscriminator(BaseDiscriminator):
    """A fake discriminator class for testing."""

    def __init__(self, threshold: float):
        """
        Args:
            threshold: The I threshold above which a shot is a 1.
        """
        self._threshold = threshold

    def predict(self, data: List[List[float]]) -> List[str]:
        """Discriminate the data"""
        return ["1" if iq[0] > self._threshold else "0" for iq in data]

    def config(self) -> Dict[str, Any]:
        """Config method."""
        return {}

    def is_trained(self) -> bool:
        """This test discriminator is always trained."""
        return True


class FakeQutritDiscriminator(BaseDiscriminator):
    """A fake qutrit discriminator class for testing."""

    def predict(self, data: List[List[float]]) -> List[str]:
        """Discriminate the data. I < 0 then |0>"""
        labels = []
        for iq in data:
            if iq[0] < 0:
                labels.append("0")
            else:
                labels.append("1" if iq[1] > 0 else "2")

        return labels

    def config(self) -> Dict[str, Any]:
        """Config method."""
        return {}

    def is_trained(self) -> bool:
        """This test discriminator is always trained."""
        return True


class TestDiscriminator(BaseDataProcessorTest):
    """Test the discriminator node."""

    def test_averaged_data(self):
        """Test that an error is raised when we try to discriminate averaged data."""
        # IQ data with dimension 3, 2, 2, i.e. 3 circuits, 2 qubits, and IQ point.
        iq_data = [[[0.2, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]], [[-1.0, -1.0], [1.0, -1.0]]]

        self.create_experiment_data(iq_data)

        discriminator = DiscriminatorNode(FakeDiscriminator(0.1))
        data = np.asarray([datum["memory"] for datum in self.iq_experiment.data()])
        with self.assertRaises(DataProcessorError):
            discriminator(data)

    def test_single_shot_data(self):
        """Test that we can discriminate single-shot data."""

        # The data has the shape
        iq_data = [
            # Circuit no. 1, 5 shots, 3 qubits
            [
                [[0.8, -1.0], [0.1, 0.3], [-0.3, 0.4]],
                [[0.2, -1.0], [-0.5, 0.3], [-0.2, 0.4]],
                [[-0.8, -1.0], [-0.1, 0.3], [-0.3, 0.4]],
                [[-0.5, -1.0], [-0.2, 0.3], [-0.9, 0.4]],
                [[-0.8, -1.0], [0.2, 0.3], [0.3, 0.4]],
            ],
            # Circuit no. 2, 5 shots, 3 qubits
            [
                [[-0.3, -1.0], [0.1, 0.3], [0.9, 0.4]],
                [[0.8, -1.0], [0.9, 0.3], [0.3, 0.4]],
                [[-0.7, -1.0], [0.1, 0.3], [0.1, 0.4]],
                [[-0.8, -1.0], [0.7, 0.3], [0.3, 0.4]],
                [[-0.8, -1.0], [-0.1, 0.3], [0.2, 0.4]],
            ],
            # Circuit no. 3, 5 shots, 3 qubits
            [
                [[-0.8, -1.0], [0.1, 0.3], [-0.2, 0.4]],
                [[0.2, -1.0], [0.1, 0.3], [0.1, 0.4]],
                [[-0.8, -1.0], [0.1, 0.3], [-0.3, 0.4]],
                [[0.8, -1.0], [-0.1, 0.3], [0.2, 0.4]],
                [[0.8, -1.0], [-0.1, 0.3], [-0.3, 0.4]],
            ],
        ]

        self.create_experiment_data(iq_data, single_shot=True)
        data = np.asarray([datum["memory"] for datum in self.iq_experiment.data()])
        discriminator = DiscriminatorNode(FakeDiscriminator(0.0))
        classified = discriminator(data)

        expected = [
            ["011", "001", "000", "000", "110"],  # Circuit no. 1
            ["110", "111", "110", "110", "100"],  # Circuit no. 2
            ["010", "111", "010", "101", "001"],  # Circuit no. 3
        ]

        self.assertListEqual(classified.tolist(), expected)

        # Now test the follow-on node that converts to count dicts.
        make_counts = MemoryToCounts()
        counts = make_counts(classified)
        expected = np.array(
            [
                {"011": 1, "001": 1, "000": 2, "110": 1},
                {"110": 3, "111": 1, "100": 1},
                {"010": 2, "111": 1, "101": 1, "001": 1},
            ]
        )
        for idx, counts_dict in enumerate(counts):
            self.assertDictEqual(counts_dict, expected[idx])

    def test_discriminator_list(self):
        """Test that we can discriminate using a list of discriminators.

        Here, the first discriminator will always return 1, the second one will give either 0
        or 1 and the last one with a threshold of 10 will always return 0.
        """

        # The data has the shape
        iq_data = [
            # Circuit no. 1, 5 shots, 3 qubits
            [
                [[0.8, -1.1], [0.1, 0.3], [-0.3, 0.4]],
                [[0.2, -1.0], [-0.5, 0.3], [-0.2, 0.4]],
                [[-0.8, -1.0], [-0.1, 0.3], [-0.3, 0.4]],
                [[-0.5, -1.0], [-0.2, 0.3], [-0.9, 0.4]],
                [[-0.8, -1.0], [0.2, 0.3], [0.3, 0.4]],
            ],
            # Circuit no. 2, 5 shots, 3 qubits
            [
                [[-0.3, -1.0], [0.1, 0.3], [0.9, 0.4]],
                [[0.8, -1.2], [0.9, 0.3], [0.3, 0.4]],
                [[-0.7, -1.0], [0.1, 0.3], [0.1, 0.4]],
                [[-0.8, -1.0], [0.7, 0.3], [0.3, 0.4]],
                [[-0.8, -1.0], [-0.1, 0.3], [0.2, 0.4]],
            ],
            # Circuit no. 3, 5 shots, 3 qubits
            [
                [[-0.8, -1.0], [0.1, 0.3], [-0.2, 0.4]],
                [[0.2, -1.0], [0.1, 0.3], [0.1, 0.4]],
                [[-0.8, -1.0], [0.1, 0.3], [-0.3, 0.4]],
                [[0.8, -1.0], [-0.1, 0.3], [0.2, 0.4]],
                [[0.8, -1.0], [-0.1, 0.3], [-0.3, 0.4]],
            ],
        ]

        self.create_experiment_data(iq_data, single_shot=True)
        data = np.asarray([datum["memory"] for datum in self.iq_experiment.data()])
        thresholds = [-10, 0, 10]
        discriminator = DiscriminatorNode(
            [FakeDiscriminator(threshold) for threshold in thresholds]
        )
        classified = discriminator(data)

        expected = [
            ["011", "001", "001", "001", "011"],  # Circuit no. 1
            ["011", "011", "011", "011", "001"],  # Circuit no. 2
            ["011", "011", "011", "001", "001"],  # Circuit no. 3
        ]

        self.assertListEqual(classified.tolist(), expected)

    def test_qutrit(self):
        """Test a qutrit discriminator."""

        iq_data = [
            [[[0.8, -1.0]], [[0.2, -1.0]], [[-0.8, -1.0]]],  # Circuit no. 1, 3 shots, 1 qutrit
            [[[-0.3, -1.0]], [[0.8, -1.0]], [[-0.7, -1.0]]],  # Circuit no. 2, 3 shots, 3 qubits
            [[[-0.8, -1.0]], [[0.2, -1.0]], [[0.8, 1.0]]],  # Circuit no. 3, 3 shots, 3 qubits
        ]

        expected = [["2", "2", "0"], ["0", "2", "0"], ["0", "2", "1"]]

        self.create_experiment_data(iq_data, single_shot=True)
        data = np.asarray([datum["memory"] for datum in self.iq_experiment.data()])
        discriminator = DiscriminatorNode(FakeQutritDiscriminator())
        classified = discriminator(data)
        self.assertListEqual(classified.tolist(), expected)


class TestMemoryToCounts(QiskitExperimentsTestCase):
    """The test data formatting."""

    def test_memory_to_counts_format(self):
        """Test that an error is raised if the data has length one."""

        node = MemoryToCounts()
        with self.assertRaises(DataProcessorError):
            node(np.array(["0", "1", "1", "0", "0"]))

    def test_memory_to_counts(self):
        """Test  simple counting."""

        node = MemoryToCounts()
        output = node(np.array([["0", "0", "1"], ["10", "11", "11"]]))
        self.assertEqual(output[0], {"0": 2, "1": 1})
        self.assertEqual(output[1], {"10": 1, "11": 2})


class TestMarginalize(QiskitExperimentsTestCase):
    """Test the marginalization node."""

    def test_marginalize(self):
        """Test the counts marginalization."""
        node = MarginalizeCounts(qubits_to_keep={0, 1})

        data = np.array(
            [
                {"010": 1, "110": 10, "100": 100},
                {"111": 1, "110": 10, "100": 100},
            ]
        )

        processed_data = node(data)

        self.assertEqual(processed_data[0], {"10": 11, "00": 100})
        self.assertEqual(processed_data[1], {"11": 1, "10": 10, "00": 100})

    def test_json(self):
        """Check if the node is serializable."""
        node = MarginalizeCounts(qubits_to_keep={0, 1})
        self.assertRoundTripSerializable(node)


class TestProbability(QiskitExperimentsTestCase):
    """Test probability computation."""

    def test_variance_not_zero(self):
        """Test if finite variance is computed at max or min probability."""
        node = Probability(outcome="1")

        data = {"1": 1024, "0": 0}
        processed_data = node(data=np.asarray([data]))
        self.assertGreater(unp.std_devs(processed_data), 0.0)
        self.assertLessEqual(unp.nominal_values(processed_data), 1.0)

        data = {"1": 0, "0": 1024}
        processed_data = node(data=np.asarray([data]))
        self.assertGreater(unp.std_devs(processed_data), 0.0)
        self.assertGreater(unp.nominal_values(processed_data), 0.0)

    def test_probability_balanced(self):
        """Test if p=0.5 is returned when counts are balanced and prior is flat."""
        node = Probability(outcome="1")

        # balanced counts with a flat prior will yield p = 0.5
        data = {"1": 512, "0": 512}
        processed_data = node(data=np.asarray([data]))
        self.assertAlmostEqual(unp.nominal_values(processed_data), 0.5)

    def test_json(self):
        """Check if the node is serializable."""
        node = Probability(outcome="00", alpha_prior=0.2)
        self.assertRoundTripSerializable(node)


class TestRestless(QiskitExperimentsTestCase):
    """Test the restless measurements node."""

    def test_restless_classify_1(self):
        """Test the classification of restless shots for two single-qubit shots.
        This example corresponds to running two single-qubit circuits without qubit reset where
        the first and second circuit would be, e.g. an X gate and an identity gate, respectively.
        We measure the qubit in the 1 state for the first circuit and measure 1 again for the
        second circuit. The second shot is reclassified as a 0 since there was no state change."""
        previous_shot = "1"
        shot = "1"

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Restless data processing nodes will be removed.*",
                category=DeprecationWarning,
            )
            restless_classified_shot = RestlessToCounts._restless_classify(shot, previous_shot)
        self.assertEqual(restless_classified_shot, "0")

    def test_restless_classify_2(self):
        """Test the classification of restless shots for two eight-qubit shots.
        In this example we run two eight qubit circuits. The first circuit applies an
        X, X, Id, Id, Id, X, X and Id gate, the second an Id, Id, X, Id, Id, X, Id and Id gate
        to qubits one to eight, respectively."""
        previous_shot = "11000110"
        shot = "11100010"

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Restless data processing nodes will be removed.*",
                category=DeprecationWarning,
            )
            restless_classified_shot = RestlessToCounts._restless_classify(shot, previous_shot)
        self.assertEqual(restless_classified_shot, "00100100")

    def test_restless_process_1(self):
        """Test that a single-qubit restless memory is correctly post-processed.
        This example corresponds to running an X gate and a SX gate with four shots
        in an ideal restless setting."""
        n_qubits = 1
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Restless data processing nodes will be removed.*",
                category=DeprecationWarning,
            )
            node = RestlessToCounts(n_qubits)

        data = [["0x1", "0x1", "0x0", "0x0"], ["0x0", "0x1", "0x1", "0x0"]]
        processed_data = node(data=np.array(data))
        # time-ordered data: ["1", "0", "1", "1", "0", "1", "0", "0"]
        # classification: ["1", "1", "1", "0", "1", "1", "1", "0"]
        expected_data = np.array([{"1": 4}, {"1": 2, "0": 2}])
        self.assertTrue(processed_data.all() == expected_data.all())

    def test_restless_process_2(self):
        """Test if a two-qubit restless memory is correctly post-processed.
        This example corresponds to running two two-qubit circuits in an ideal restless setting.
        The first circuit applies an X gate to the first and a SX gate to the second qubit. The
        second circuit applies two identity gates."""
        n_qubits = 2
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Restless data processing nodes will be removed.*",
                category=DeprecationWarning,
            )
            node = RestlessToCounts(n_qubits)

        data = [["0x3", "0x1", "0x2", "0x0"], ["0x3", "0x1", "0x2", "0x0"]]
        processed_data = node(data=np.array(data))
        # time-ordered data: ["11", "11", "01", "01", "10", "10", "00", "00"]
        # classification: ["11", "00", "10", "00", "11", "00", "10", "00"]
        expected_data = np.array([{"10": 2, "11": 2}, {"00": 4}])
        self.assertTrue(processed_data.all() == expected_data.all())

    def test_restless_iq_process(self):
        """Test restless IQ data processing."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Restless data processing nodes will be removed.*",
                category=DeprecationWarning,
            )
            node = RestlessToIQ()

        # The shape of the IQ data is (2, 2, 1, 2) corresponding to
        # 2 circuits, 2 shots, 1 qubit and the respective IQ-point [I, Q].
        data = [[[[1, -2]], [[1, 3]]], [[[6, -4]], [[-3, 1]]]]
        # time-ordered data: [[[1, -2]], [[6, -4]], [[1, 3]], [[-3, 1]]]
        # subtraction: [[[1, -2]], [[5, -2]], [[-5, 7]], [[-4, -2]]]
        # absolute value: [[[1, -2]], [[5, 2]], [[5, 7]], [[4, 2]]]
        # sorted by circuit: [[[[1, -2]], [[5, 7]]], [[[5, 2]], [[4, 2]]]]
        expected_data = np.array([[[[1, -2]], [[5, 7]]], [[[5, 2]], [[4, 2]]]])
        processed_data = node(data=np.array(data))
        self.assertTrue(processed_data.all() == expected_data.all())

    def test_restless_iq_process_2q(self):
        """Test two-qubit restless IQ data processing."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Restless data processing nodes will be removed.*",
                category=DeprecationWarning,
            )
            node = RestlessToIQ()

        # The shape of the IQ data is (2, 2, 2, 2) corresponding to
        # 2 circuits, 2 shots, 2 qubits and the respective IQ-point [I, Q].
        data = [[[[1, -2], [2, 5]], [[1, 3], [4, 2]]], [[[6, -4], [-8, 2]], [[-3, 1], [0, 3]]]]
        # time-ordered data: [[[1, -2], [2, 5]], [[6, -4], [-8, 2]], [[1, 3], [4, 2]], [[-3, 1], [0, 3]]]
        # subtraction: [[[1, -2], [2, 5]], [[5, -2], [-10, -3]], [[-5, 7], [12, 0]], [[-4, -2], [-4, 1]]]
        # absolute value: [[[1, -2], [2, 5]], [[5, 2], [10, 3]], [[5, 7], [12, 0]], [[4, 2], [4, 1]]]
        # sorted by circuit: [[[[1, -2], [2, 5]], [[5, 7], [12, 0]]],
        # [[[5, 2], [10, 3]], [[4, 2], [12, 0]]]]
        expected_data = np.array(
            [[[[1, -2], [2, 5]], [[5, 7], [12, 0]]], [[[5, 2], [10, 3]], [[4, 2], [12, 0]]]]
        )
        processed_data = node(data=np.array(data))
        self.assertTrue(processed_data.all() == expected_data.all())
