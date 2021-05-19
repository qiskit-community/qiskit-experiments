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

import numpy as np

from qiskit import QuantumCircuit, execute
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.test import QiskitTestCase

from qiskit_experiments import ExperimentData
from qiskit_experiments.calibration.experiments.rabi import RabiAnalysis
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.nodes import Probability


class TestRabiAnalysis(QiskitTestCase):
    """Class to test the fitting."""

    def simulate_experiment_data(self, thetas, amplitudes, shots=1024):
        """Generate experiment data for Rx rotations with an arbitrary amplitude calibration."""
        circuits = []
        for theta in thetas:
            qc = QuantumCircuit(1)
            qc.rx(theta, 0)
            qc.measure_all()
            circuits.append(qc)

        sim = QasmSimulatorPy()
        result = execute(circuits, sim, shots=shots, seed_simulator=10).result()
        data = [
            {"counts": self._add_uncertainty(result.get_counts(i)),
             "metadata": {"xval": amplitudes[i]}} for i, theta in enumerate(thetas)
        ]
        return data

    def _add_uncertainty(self, counts):
        """Ensure that we always have a non-zero sigma in the test."""
        for label in ["0", "1"]:
            if label not in counts:
                counts[label] = 1

        return counts

    def test_analysis(self):
        """Test the Rabi analysis."""
        experiment_data = ExperimentData()

        thetas = np.linspace(-1.5*np.pi, 1.5*np.pi, 51)
        amplitudes = np.linspace(-0.95, 0.95, 51)

        data = self.simulate_experiment_data(thetas, amplitudes)
        experiment_data.add_data(data)

        rabi_analysis = RabiAnalysis()

        data_processor = DataProcessor("counts", [Probability(outcome="1")])

        result = rabi_analysis.run(experiment_data, data_processor=data_processor, plot=False)

        print(result)