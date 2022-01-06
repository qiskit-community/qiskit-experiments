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
from test.base import QiskitExperimentsTestCase
from typing import Tuple
import numpy as np

from qiskit import QuantumCircuit
from qiskit.qobj.utils import MeasLevel

from qiskit_experiments.library import QubitSpectroscopy, EFSpectroscopy
from qiskit_experiments.test.mock_iq_backend import MockIQBackend


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

        super().__init__(iq_cluster_centers, iq_cluster_width)

        self.configuration().basis_gates = ["x"]

        self._linewidth = line_width
        self._freq_offset = freq_offset

        super().__init__(iq_cluster_centers, iq_cluster_width)

    def _compute_probability(self, circuit: QuantumCircuit) -> float:
        """Returns the probability based on the frequency."""
        freq_shift = next(iter(circuit.calibrations["Spec"]))[1][0]
        delta_freq = freq_shift - self._freq_offset
        return np.exp(-(delta_freq ** 2) / (2 * self._linewidth ** 2))


class TestQubitSpectroscopy(QiskitExperimentsTestCase):
    """Test spectroscopy experiment."""

    def test_spectroscopy_end2end_classified(self):
        """End to end test of the spectroscopy experiment."""

        backend = SpectroscopyBackend(line_width=2e6)
        qubit = 1
        freq01 = backend.defaults().qubit_freq_est[qubit]
        frequencies = np.linspace(freq01 - 10.0e6, freq01 + 10.0e6, 21)

        spec = QubitSpectroscopy(qubit, frequencies)
        spec.set_run_options(meas_level=MeasLevel.CLASSIFIED)
        expdata = spec.run(backend)
        result = expdata.analysis_results(1)
        value = result.value.value

        self.assertTrue(4.999e9 < value < 5.001e9)
        self.assertEqual(result.quality, "good")

        # Test if we find still find the peak when it is shifted by 5 MHz.
        backend = SpectroscopyBackend(line_width=2e6, freq_offset=5.0e6)

        spec = QubitSpectroscopy(qubit, frequencies)
        spec.set_run_options(meas_level=MeasLevel.CLASSIFIED)
        expdata = spec.run(backend)
        result = expdata.analysis_results(1)
        value = result.value.value

        self.assertTrue(5.0049e9 < value < 5.0051e9)
        self.assertEqual(result.quality, "good")

    def test_spectroscopy_end2end_kerneled(self):
        """End to end test of the spectroscopy experiment on IQ data."""

        backend = SpectroscopyBackend(line_width=2e6)
        qubit = 0
        freq01 = backend.defaults().qubit_freq_est[qubit]
        frequencies = np.linspace(freq01 - 10.0e6, freq01 + 10.0e6, 21)

        spec = QubitSpectroscopy(qubit, frequencies)
        expdata = spec.run(backend)
        result = expdata.analysis_results(1)
        value = result.value.value

        self.assertTrue(freq01 - 2e6 < value < freq01 + 2e6)
        self.assertEqual(result.quality, "good")

        # Test if we find still find the peak when it is shifted by 5 MHz.
        backend = SpectroscopyBackend(line_width=2e6, freq_offset=5.0e6)

        spec = QubitSpectroscopy(qubit, frequencies)
        expdata = spec.run(backend)
        result = expdata.analysis_results(1)
        value = result.value.value

        self.assertTrue(freq01 + 3e6 < value < freq01 + 8e6)
        self.assertEqual(result.quality, "good")

        spec.set_run_options(meas_return="avg")
        expdata = spec.run(backend)
        result = expdata.analysis_results(1)
        value = result.value.value

        self.assertTrue(freq01 + 3e6 < value < freq01 + 8e6)
        self.assertEqual(result.quality, "good")

    def test_spectroscopy12_end2end_classified(self):
        """End to end test of the spectroscopy experiment with an x pulse."""

        backend = SpectroscopyBackend(line_width=2e6)
        qubit = 0
        freq01 = backend.defaults().qubit_freq_est[qubit]
        frequencies = np.linspace(freq01 - 10.0e6, freq01 + 10.0e6, 21)

        # Note that the backend is not sophisticated enough to simulate an e-f
        # transition so we run the test with g-e.
        spec = EFSpectroscopy(qubit, frequencies)
        spec.backend = backend
        spec.set_run_options(meas_level=MeasLevel.CLASSIFIED)
        expdata = spec.run(backend)
        result = expdata.analysis_results(1)
        value = result.value.value

        self.assertTrue(freq01 - 2e6 < value < freq01 + 2e6)
        self.assertEqual(result.quality, "good")

        # Test the circuits
        circ = spec.circuits()[0]
        self.assertEqual(circ.data[0][0].name, "x")
        self.assertEqual(circ.data[1][0].name, "Spec")

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = QubitSpectroscopy(1, np.linspace(100, 150, 20) * 1e6)
        loaded_exp = QubitSpectroscopy.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.experiments_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = QubitSpectroscopy(1, np.linspace(int(100e6), int(150e6), int(20e6)))
        self.assertRoundTripSerializable(exp, self.experiments_equiv)
