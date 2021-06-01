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

import numpy as np
from qiskit.test import QiskitTestCase
from qiskit import QuantumCircuit, execute
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit_experiments.analysis.curve_fitting import curve_fit, multi_curve_fit, process_curve_data
from qiskit_experiments.analysis.data_processing import (
    level2_probability,
    mean_xy_data,
    multi_mean_xy_data,
)


class TestCurveFitting(QiskitTestCase):
    """Test curve fitting functions."""

    def simulate_experiment_data(self, thetas, shots=1024):
        """Generate experiment data for Ry rotations"""
        circuits = []
        for theta in thetas:
            qc = QuantumCircuit(1)
            qc.ry(theta, 0)
            qc.measure_all()
            circuits.append(qc)

        sim = QasmSimulatorPy()
        result = execute(circuits, sim, shots=shots, seed_simulator=10).result()
        data = [
            {"counts": result.get_counts(i), "metadata": {"xval": theta}}
            for i, theta in enumerate(thetas)
        ]
        return data

    @property
    def objective0(self):
        """Objective function for P0"""

        def func0(x, omega):
            return np.cos(omega * x) ** 2

        return np.vectorize(func0)

    @property
    def objective1(self):
        """Objective function for P0"""

        def func1(x, omega):
            return np.sin(omega * x) ** 2

        return np.vectorize(func1)

    @staticmethod
    def data_processor_p0(data):
        """Return P(0) probabilities"""
        return level2_probability(data, "0")

    @staticmethod
    def data_processor_p1(data):
        """Return P(1) probabilities"""
        return level2_probability(data, "1")

    def test_process_curve_data(self):
        """Test version string generation."""
        thetas = thetas = np.linspace(0.5, 4 * np.pi - 0.5, 20)
        data = self.simulate_experiment_data(thetas)
        xdata, ydata, _ = process_curve_data(data, data_processor=self.data_processor_p0)

        xdiff = thetas - xdata
        ydiff = self.objective0(xdata, 0.5) - ydata
        self.assertTrue(np.allclose(xdiff, 0))
        self.assertTrue(np.allclose(ydiff, 0, atol=0.05))

    def test_curve_fit(self):
        """Test curve_fit function"""
        thetas = thetas = np.linspace(0.5, 4 * np.pi - 0.5, 20)
        data = self.simulate_experiment_data(thetas)
        xdata, ydata, sigma = process_curve_data(data, data_processor=self.data_processor_p0)
        p0 = [0.6]
        bounds = ([0], [2])
        sol = curve_fit(self.objective0, xdata, ydata, p0, sigma=sigma, bounds=bounds)
        self.assertTrue(abs(sol["popt"][0] - 0.5) < 0.05)

    def test_multi_curve_fit(self):
        """Test multi_curve_fit function"""
        thetas = thetas = np.linspace(0.5, 4 * np.pi - 0.5, 20)
        data = self.simulate_experiment_data(thetas)
        xdata0, ydata0, sigma0 = process_curve_data(data, data_processor=self.data_processor_p0)
        xdata1, ydata1, sigma1 = process_curve_data(data, data_processor=self.data_processor_p1)

        # Combine curve data
        xdata = np.concatenate([xdata0, xdata1])
        series = np.concatenate([np.zeros(len(xdata0)), np.ones(len(xdata1))])
        ydata = np.concatenate([ydata0, ydata1])
        sigma = np.concatenate([sigma0, sigma1])

        p0 = [0.6]
        bounds = ([0], [2])
        sol = multi_curve_fit(
            [self.objective0, self.objective1], series, xdata, ydata, p0, sigma=sigma, bounds=bounds
        )
        self.assertTrue(abs(sol["popt"][0] - 0.5) < 0.05)

    def test_mean_xy_data(self):
        """Test mean_xy_data function"""
        # pylint: disable=unbalanced-tuple-unpacking
        x = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 5, 5])
        y = np.array([1, 2, 3, 8, 10, 50, 60, 10, 11, 17, 10, 10, 10, 10])
        x_mean, y_mean, y_sigma = mean_xy_data(x, y, method="sample")

        expected_x_mean = np.array([1, 2, 3, 4, 5])
        expected_y_mean = np.array([2, 32, 10.5, 17, 10])
        expected_y_sigma = np.sqrt(np.array([2 / 3, 542, 1 / 4, 0, 0]))
        self.assertTrue(np.allclose(expected_x_mean, x_mean))
        self.assertTrue(np.allclose(expected_y_mean, y_mean))
        self.assertTrue(np.allclose(expected_y_sigma, y_sigma))

        sigma = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        x_mean, y_mean, y_sigma = mean_xy_data(x, y, sigma, method="iwv")
        expected_y_mean = np.array([1.34693878, 23.31590234, 10.44137931, 17.0, 10.0])
        expected_y_sigma = np.array([0.85714286, 2.57610543, 5.97927455, 10.0, 6.17470935])
        self.assertTrue(np.allclose(expected_x_mean, x_mean))
        self.assertTrue(np.allclose(expected_y_mean, y_mean))
        self.assertTrue(np.allclose(expected_y_sigma, y_sigma))

        x = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        y = np.array([2, 6, 100, 200, 17, 50, 60, 70])
        series = np.array([0, 0, 1, 1, 0, 1, 1, 1])
        series, x_mean, y_mean, y_sigma = multi_mean_xy_data(series, x, y, method="sample")
        expected_x_mean = np.array([1, 2, 1, 2])
        expected_y_mean = np.array([4, 17, 150, 60])
        expected_y_sigma = np.sqrt(np.array([4.0, 0.0, 2500.0, 66.66666667]))
        expected_series = np.array([0, 0, 1, 1])
        self.assertTrue(np.allclose(expected_x_mean, x_mean))
        self.assertTrue(np.allclose(expected_y_mean, y_mean))
        self.assertTrue(np.allclose(expected_y_sigma, y_sigma))
        self.assertTrue(np.allclose(expected_series, series))
