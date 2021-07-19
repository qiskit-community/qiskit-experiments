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
from qiskit_experiments.measurement.discriminator import Discriminator
from qiskit_experiments.data_processing.nodes import Probability, Discriminate
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.composite import ParallelExperiment


class DiscriminatorBackend(MockIQBackend):
    """
    A simple backend that generates gaussian data for discriminator tests
    """

    def __init__(
        self,
        iq_cluster_centers: Tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0),
        iq_cluster_width: float = 1.5,
    ):
        """
        Initialize the discriminator backend
        """
        super().__init__(iq_cluster_centers, iq_cluster_width)
        self.configuration().basis_gates = ["x"]

    def _compute_probability(self, circuit: QuantumCircuit) -> float:
        """Returns the probability based on the frequency."""
        if circuit.data[0][0].name == "x":
            return 1
        elif circuit.data[0][0].name == "barrier":
            return 0


class TestDiscriminator(QiskitTestCase):
    """Class to test the discriminator."""

    def test_single_qubit(self):
        """Test the default LDA discriminator works on one qubit."""
        backend = DiscriminatorBackend()
        exp = Discriminator(1)
        exp.set_run_options(meas_level=MeasLevel.KERNELED)
        res = exp.run(backend, shots=1000, meas_return="single").analysis_result(0)
        self.assertEqual(res["success"], True)
        self.assertAlmostEqual(res["coef"][0][0], 0.9051186)
        self.assertAlmostEqual(res["coef"][0][1], 0.87117249)
        self.assertAlmostEqual(res["intercept"][0], 0.04186)

    def test_single_qubit_qda(self):
        """Test that the QDA discriminator works on one qubit."""
        backend = DiscriminatorBackend()
        exp = Discriminator(1)
        exp.set_analysis_options(discriminator_type="QDA")
        exp.set_run_options(meas_level=MeasLevel.KERNELED)
        res = exp.run(backend, shots=1000, meas_return="single").analysis_result(0)
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
        """Test that the discriminator works in the data processing chain."""
        backend = DiscriminatorBackend()
        exp = Discriminator(1)
        exp.set_analysis_options(discriminator_type="LDA")
        exp.set_run_options(meas_level=MeasLevel.KERNELED)
        lda_res = exp.run(backend, shots=1000)
        processor = DataProcessor("memory", [Discriminate(lda_res)])
        processor.append(Probability("0"))
        datum = processor(lda_res.data(0))
        self.assertTrue(np.allclose(datum, (0.821, 0.012122664723566351)))

        backend = DiscriminatorBackend()
        exp = Discriminator(1)
        exp.set_analysis_options(discriminator_type="QDA")
        exp.set_run_options(meas_level=MeasLevel.KERNELED)
        qda_res = exp.run(backend, shots=1000)
        qda_processor = DataProcessor("memory", [Discriminate(qda_res)])
        qda_processor.append(Probability("0"))
        datum = qda_processor(qda_res.data(0))
        self.assertTrue(np.allclose(datum, (0.819, 0.012175343937647102)))

    def test_parallel_discriminator(self):
        """Test the discriminator data processor works correctly on multiple qubits."""
        backend = DiscriminatorBackend()
        par_exp = ParallelExperiment([Discriminator(0), Discriminator(1)])
        par_exp.set_run_options(meas_level=MeasLevel.KERNELED, meas_return="single")
        par_expdata = par_exp.run(backend, shots=1000)
        processor = DataProcessor("memory", [Discriminate(par_expdata)])
        processor.append(Probability("01"))
        datum = processor([par_expdata.data(0), par_expdata.data(1)])
        self.assertAlmostEqual(datum[0], 0.682)
