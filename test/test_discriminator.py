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
from qiskit_experiments.measurement.discriminator.twoleveldiscriminator_experiment import (
    TwoLevelDiscriminator,
)
from qiskit_experiments.data_processing.nodes import Probability, TwoLevelDiscriminate
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.composite import ParallelExperiment


class TwoLevelDiscriminatorBackend(MockIQBackend):
    """
    A simple backend that generates gaussian data for TwoLevelDiscriminator tests
    """

    def __init__(
        self,
        iq_cluster_centers: Tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0),
        iq_cluster_width: float = 1.5,
    ):
        """
        Initialize the TwoLevelDiscriminator backend
        """
        super().__init__(iq_cluster_centers, iq_cluster_width)
        self.configuration().basis_gates = ["x"]

    def _compute_probability(self, circuit: QuantumCircuit) -> float:
        """Returns the probability based on the frequency."""
        if circuit.data[0][0].name == "x":
            return 1
        elif circuit.data[0][0].name == "barrier":
            return 0


class TestTwoLevelDiscriminator(QiskitTestCase):
    """Class to test the TwoLevelDiscriminator."""

    def test_single_qubit(self):
        """Test the default LDA TwoLevelDiscriminator works on one qubit."""
        backend = TwoLevelDiscriminatorBackend()
        exp = TwoLevelDiscriminator(1)
        res = exp.run(backend, shots=1000).analysis_results(0)
        self.assertEqual(res["success"], True)
        self.assertAlmostEqual(res["coef"][0][0], 0.9051186)
        self.assertAlmostEqual(res["coef"][0][1], 0.87117249)
        self.assertAlmostEqual(res["intercept"][0], 0.04186)

    def test_single_qubit_qda(self):
        """Test that the QDA TwoLevelDiscriminator works on one qubit."""
        backend = TwoLevelDiscriminatorBackend()
        exp = TwoLevelDiscriminator(1)
        exp.set_analysis_options(Discriminator_type="QDA")
        res = exp.run(backend, shots=1000).analysis_results(0)
        self.assertEqual(res["success"], True)

        self.assertTrue(
            np.allclose(
                res["rotations"][0],
                [[-0.43811802, 0.89891746], [-0.89891746, -0.43811802]],
            )
        )
        self.assertTrue(
            np.allclose(
                res["rotations"][1],
                [[-0.17039714, 0.98537547], [0.98537547, 0.17039714]],
            )
        )

    def test_discriminate_data_processor_node(self):
        """Test that the TwoLevelDiscriminator works in the data processing chain."""
        backend = TwoLevelDiscriminatorBackend()
        exp = TwoLevelDiscriminator(1)
        exp.set_analysis_options(Discriminator_type="LDA")
        lda_res = exp.run(backend, shots=1000)
        processor = DataProcessor("memory", [TwoLevelDiscriminate(lda_res)])
        processor.append(Probability("0"))
        datum = processor(lda_res.data(0))
        self.assertTrue(np.allclose(datum, (0.821, 0.012122664723566351)))

        backend = TwoLevelDiscriminatorBackend()
        exp = TwoLevelDiscriminator(1)
        exp.set_analysis_options(Discriminator_type="QDA")
        qda_res = exp.run(backend, shots=1000)
        qda_processor = DataProcessor("memory", [TwoLevelDiscriminate(qda_res)])
        qda_processor.append(Probability("0"))
        datum = qda_processor(qda_res.data(0))
        self.assertTrue(np.allclose(datum, (0.819, 0.012175343937647102)))

    def test_parallel_TwoLevelDiscriminator(self):
        """Test the TwoLevelDiscriminator data processor works correctly on multiple qubits."""
        backend = TwoLevelDiscriminatorBackend()
        par_exp = ParallelExperiment([TwoLevelDiscriminator(0), TwoLevelDiscriminator(1)])
        par_exp.set_run_options(meas_level=MeasLevel.KERNELED, meas_return="single")
        par_expdata = par_exp.run(backend, shots=1000)
        processor = DataProcessor("memory", [TwoLevelDiscriminate(par_expdata)])
        processor.append(Probability("01"))
        datum = processor([par_expdata.data(0), par_expdata.data(1)])
        self.assertAlmostEqual(datum[0], 0.682)

    def test_training_TwoLevelDiscriminator(self):
        """Test that the discrminator node can be trained."""
        pass
