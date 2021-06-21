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

"""Spectroscopy tests."""

from typing import Tuple
from test.mock_iq_backend import MockIQBackend
import numpy as np

from qiskit import QuantumCircuit
from qiskit.qobj.utils import MeasLevel
from qiskit.test import QiskitTestCase

from qiskit_experiments.characterization.qubit_spectroscopy import QubitSpectroscopy
from qiskit_experiments.analysis import get_opt_value


class SpectroscopyBackend(MockIQBackend):
    """A simple and primitive backend to test spectroscopy experiments."""

    def __init__(
        self,
        line_width: float = 2e6,
        freq_offset: float = 0.0,
        iq_cluster_centers: Tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0),
        iq_cluster_width: float = 0.2,
    ):
        """Initialize the spectroscopy backend."""

        self.__configuration__["basis_gates"] = ["spec"]

        self._linewidth = line_width
        self._freq_offset = freq_offset

        super().__init__(iq_cluster_centers, iq_cluster_width)

    def _compute_probability(self, circuit: QuantumCircuit) -> float:
        """Returns the probability based on the frequency."""
        set_freq = float(circuit.data[0][0].params[0])
        delta_freq = set_freq - self._freq_offset
        return np.exp(-(delta_freq ** 2) / (2 * self._linewidth ** 2))


class TestQubitSpectroscopy(QiskitTestCase):
    """Test spectroscopy experiment."""

    def test_spectroscopy_end2end_classified(self):
        """End to end test of the spectroscopy experiment."""

        backend = SpectroscopyBackend(line_width=2e6)

        spec = QubitSpectroscopy(3, np.linspace(-10.0, 10.0, 21), unit="MHz")
        spec.set_run_options(meas_level=MeasLevel.CLASSIFIED)
        result = spec.run(backend).analysis_result(0)

        value = get_opt_value(result, "freq")

        self.assertTrue(abs(value) < 1e6)
        self.assertTrue(result["success"])
        self.assertEqual(result["quality"], "computer_good")

        # Test if we find still find the peak when it is shifted by 5 MHz.
        backend = SpectroscopyBackend(line_width=2e6, freq_offset=5.0e6)

        spec = QubitSpectroscopy(3, np.linspace(-10.0, 10.0, 21), unit="MHz")
        spec.set_run_options(meas_level=MeasLevel.CLASSIFIED)
        result = spec.run(backend).analysis_result(0)

        value = get_opt_value(result, "freq")

        self.assertTrue(value < 5.1e6)
        self.assertTrue(value > 4.9e6)
        self.assertEqual(result["quality"], "computer_good")

    def test_spectroscopy_end2end_kerneled(self):
        """End to end test of the spectroscopy experiment on IQ data."""

        backend = SpectroscopyBackend(line_width=2e6)

        spec = QubitSpectroscopy(3, np.linspace(-10.0, 10.0, 21), unit="MHz")
        result = spec.run(backend).analysis_result(0)

        value = get_opt_value(result, "freq")

        self.assertTrue(abs(value) < 1e6)
        self.assertTrue(result["success"])
        self.assertEqual(result["quality"], "computer_good")

        # Test if we find still find the peak when it is shifted by 5 MHz.
        backend = SpectroscopyBackend(line_width=2e6, freq_offset=5.0e6)

        spec = QubitSpectroscopy(3, np.linspace(-10.0, 10.0, 21), unit="MHz")
        result = spec.run(backend).analysis_result(0)

        value = get_opt_value(result, "freq")

        self.assertTrue(value < 5.1e6)
        self.assertTrue(value > 4.9e6)
        self.assertEqual(result["quality"], "computer_good")

        spec.set_run_options(meas_return="avg")
        result = spec.run(backend).analysis_result(0)

        value = get_opt_value(result, "freq")

        self.assertTrue(value < 5.1e6)
        self.assertTrue(value > 4.9e6)
        self.assertEqual(result["quality"], "computer_good")
