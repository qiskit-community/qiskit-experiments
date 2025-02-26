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

"""Test Rabi amplitude Experiment class."""
from test.base import QiskitExperimentsTestCase
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.qobj.utils import MeasLevel
from qiskit_aer import AerSimulator

from qiskit_experiments.framework import ExperimentData

from qiskit_experiments.curve_analysis.standard_analysis.oscillation import OscillationAnalysis
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.nodes import Probability
from qiskit_experiments.curve_analysis import ParameterRepr


class TestOscillationAnalysis(QiskitExperimentsTestCase):
    """Class to test the fitting."""

    def simulate_experiment_data(self, thetas, amplitudes, shots=1024):
        """Generate experiment data for Rx rotations with an arbitrary amplitude calibration."""
        circuits = []
        for theta in thetas:
            qc = QuantumCircuit(1)
            qc.rx(theta, 0)
            qc.measure_all()
            circuits.append(qc)

        sim = AerSimulator()
        circuits = transpile(circuits, sim)
        job = sim.run(circuits, shots=shots, seed_simulator=10)
        result = job.result()
        data = [
            {
                "counts": self._add_uncertainty(result.get_counts(i)),
                "metadata": {
                    "xval": amplitudes[i],
                    "meas_level": MeasLevel.CLASSIFIED,
                    "meas_return": "avg",
                },
            }
            for i, theta in enumerate(thetas)
        ]
        return data

    @staticmethod
    def _add_uncertainty(counts):
        """Ensure that we always have a non-zero sigma in the test."""
        for label in ["0", "1"]:
            if label not in counts:
                counts[next(iter(counts))] -= 1
                counts[label] = 1

        return counts

    def test_good_analysis(self):
        """Test the Rabi analysis."""
        experiment_data = ExperimentData()

        thetas = np.linspace(-np.pi, np.pi, 31)
        amplitudes = np.linspace(-0.25, 0.25, 31)
        expected_rate, test_tol = 2.0, 0.2

        experiment_data.add_data(self.simulate_experiment_data(thetas, amplitudes, shots=400))

        data_processor = DataProcessor("counts", [Probability(outcome="1")])

        analysis = OscillationAnalysis()
        analysis.set_options(
            result_parameters=[ParameterRepr("freq", "rabi_rate")],
        )

        experiment_data = analysis.run(
            experiment_data, data_processor=data_processor, plot=False
        ).block_for_results()

        result = experiment_data.analysis_results("rabi_rate")
        self.assertEqual(result.quality, "good")
        self.assertAlmostEqual(result.value, expected_rate, delta=test_tol)

    def test_bad_analysis(self):
        """Test the Rabi analysis."""
        experiment_data = ExperimentData()

        # Change rotation angle with square root of amplitude so that
        # population versus amplitude will not be sinusoidal and the fit will
        # be bad.
        thetas = np.sqrt(np.linspace(0.0, 4 * np.pi**2, 31))
        amplitudes = np.linspace(0.0, 0.95, 31)

        experiment_data.add_data(self.simulate_experiment_data(thetas, amplitudes, shots=200))

        data_processor = DataProcessor("counts", [Probability(outcome="1")])

        analysis = OscillationAnalysis()
        analysis.set_options(
            result_parameters=[ParameterRepr("freq", "rabi_rate")],
        )
        experiment_data = analysis.run(
            experiment_data,
            data_processor=data_processor,
            plot=False,
        ).block_for_results()

        result = experiment_data.analysis_results("rabi_rate")

        self.assertEqual(result.quality, "bad")
