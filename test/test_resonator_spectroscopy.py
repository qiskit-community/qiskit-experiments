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

"""Spectroscopy tests for resonator spectroscop experiment."""

from test.base import QiskitExperimentsTestCase
from typing import Tuple
import numpy as np
from ddt import ddt, data

from qiskit import QuantumCircuit

from qiskit_experiments.library import ResonatorSpectroscopy
from qiskit_experiments.test.mock_iq_backend import MockIQBackend


class ResonatorSpectroscopyBackend(MockIQBackend):
    """A simple and primitive backend to test spectroscopy experiments."""

    def __init__(
        self,
        line_width: float = 2e6,
        freq_offset: float = 0.0,
        iq_cluster_centers: Tuple[float, float, float, float] = (-1.0, 0.0, 0.0, 0.0),
        iq_cluster_width: float = 0.2,
    ):
        """Initialize the spectroscopy backend."""

        super().__init__(iq_cluster_centers, iq_cluster_width)

        self._linewidth = line_width
        self._freq_offset = freq_offset
        self._configuration.timing_constraints = {"granularity": 16}

    def _compute_probability(self, circuit: QuantumCircuit) -> float:
        """Returns the probability based on the frequency."""
        freq_shift = next(iter(circuit.calibrations["measure"].values())).blocks[0].frequency
        delta_freq = freq_shift - self._freq_offset

        return np.abs(1 / (1 + 2.0j * delta_freq / self._linewidth))

    def _iq_phase(self, circuit: QuantumCircuit) -> float:
        """Add a phase to the IQ point depending on how far we are from the resonance.

        This will cause the IQ points to rotate around in the IQ plane when we approach the
        resonance which introduces and extra complication that the data processor needs to
        properly handle.
        """
        freq_shift = next(iter(circuit.calibrations["measure"].values())).blocks[0].frequency
        delta_freq = freq_shift - self._freq_offset

        return delta_freq / self._linewidth


@ddt
class TestResonatorSpectroscopy(QiskitExperimentsTestCase):
    """Tests for the resonator spectroscopy experiment."""

    @data(-5e6, -2e6, 0, 1e6, 3e6)
    def test_end_to_end(self, freq_shift):
        """Test the experiment from end to end."""

        qubit = 1
        backend = ResonatorSpectroscopyBackend(freq_offset=freq_shift)
        res_freq = backend.defaults().meas_freq_est[qubit]

        frequencies = np.linspace(res_freq - 20e6, res_freq + 20e6, 51)
        spec = ResonatorSpectroscopy(qubit, backend=backend, frequencies=frequencies)

        expdata = spec.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results(1)
        self.assertRoundTripSerializable(result.value, check_func=self.ufloat_equiv)

        self.assertAlmostEqual(result.value.n, res_freq + freq_shift, delta=0.1e6)
        self.assertEqual(str(result.device_components[0]), f"R{qubit}")

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = ResonatorSpectroscopy(1, frequencies=np.linspace(100, 150, 20) * 1e6)
        loaded_exp = ResonatorSpectroscopy.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = ResonatorSpectroscopy(1, frequencies=np.linspace(int(100e6), int(150e6), int(20e6)))
        self.assertRoundTripSerializable(exp, self.json_equiv)
