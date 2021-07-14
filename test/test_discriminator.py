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

"""Test version string generation."""

from typing import Tuple
import numpy as np

from qiskit import QuantumCircuit
from qiskit.qobj.utils import MeasLevel
from qiskit.test import QiskitTestCase

from qiskit_experiments.test.mock_iq_backend import MockIQBackend
from qiskit_experiments.measurement.discriminator import (
    Discriminator,
)
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.nodes import Probability, Discriminator


class DiscriminatorBackend(MockIQBackend):
    """
    A simple backend that generates gaussian data for discriminator tests
    """

    def __init__(
        self,
        iq_cluster_centers: Tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0),
        iq_cluster_width: float = 0.2,
    ):
        """
        Initialize the discriminator backend
        """
        super().__init__(iq_cluster_centers, iq_cluster_width)

    def _compute_probability(self, circuit: QuantumCircuit) -> float:
        """Returns the probability based on the frequency."""
        if circuit.data[-1][0].name == "x":
            return 1
        elif circuit.data[-1][0].name == "measure":
            return 0


class TestDiscriminator(QiskitTestCase):
    """Class to test the discriminator."""

    def test_single_qubit(self):
        """Test the discriminator works on one qubit."""
        backend = DiscriminatorBackend()
        exp = Discriminator(1)
        exp.set_run_options(meas_level=MeasLevel.KERNELED)
        res = exp.run(backend, shots=10, meas_return="single").analysis_result(0)
        self.assertEqual(res["success"], True)
        self.assertAlmostEqual(res["coef"][0][0], 0.30699667)
        self.assertAlmostEqual(res["coef"][0][1], 4.2800415)
        self.assertAlmostEqual(res["intercept"][0], 4.72491722)

    def test_single_qubit_qda(self):
        """Test that the QDA discriminator works on one qubit."""
        backend = DiscriminatorBackend()
        exp = Discriminator(1)
        exp.set_analysis_options(discriminator_type="QDA")
        exp.set_run_options(meas_level=MeasLevel.KERNELED)
        res = exp.run(backend, shots=10, meas_return="single").analysis_result(0)
        self.assertEqual(res["success"], True)

        self.assertTrue(
            np.allclose(
                res["rotations"][0],
                [[-0.59700674, -0.80223622], [0.80223622, -0.59700674]],
            )
        )

    def test_parallel_discriminator(self):
        """Test the discriminator works correctly on multiple qubits."""

        pass

    def test_discriminator_data_processor_node(self):
        """Test that the discriminator works in the data processing chain."""
        backend = DiscriminatorBackend()
        exp = Discriminator(1)
        exp.set_run_options(meas_level=MeasLevel.KERNELED)
        lda_res = exp.run(backend, shots=100)
        print(lda_res)
        processor = DataProcessor("memory", [Discriminator(lda_res)])
        processor.append(Probability("0"))
        datum = processor(lda_res.data(0))
        self.assertTrue(np.allclose(datum, (0.53, 0.04990991885387112)))
