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

"""Test the fine frequency characterization and calibration experiments."""

from test.base import QiskitExperimentsTestCase
import numpy as np
from ddt import ddt, data

from qiskit_experiments.library import FineFrequency
from qiskit_experiments.framework import BackendData
from qiskit_experiments.test.mock_iq_backend import MockIQBackend
from qiskit_experiments.test.mock_iq_helpers import MockIQFineFreqHelper as FineFreqHelper


@ddt
class TestFineFreqEndToEnd(QiskitExperimentsTestCase):
    """Test the fine freq experiment."""

    def setUp(self):
        """Setup for the test."""
        super().setUp()

        self.sx_duration = 160

    @data(-0.5e6, -0.1e6, 0.1e6, 0.5e6)
    def test_end_to_end(self, freq_shift):
        """Test the experiment end to end."""
        exp_helper = FineFreqHelper(sx_duration=self.sx_duration, freq_shift=freq_shift)
        backend = MockIQBackend(exp_helper)
        exp_helper.dt = BackendData(backend).dt

        freq_exp = FineFrequency([0], 160, backend)

        expdata = freq_exp.run(shots=100)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results("d_theta")
        d_theta = result.value.n
        dt = BackendData(backend).dt
        d_freq = d_theta / (2 * np.pi * self.sx_duration * dt)

        tol = 0.01e6

        self.assertAlmostEqual(d_freq, freq_shift, delta=tol)
        self.assertEqual(result.quality, "good")

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = FineFrequency([0], 160)
        loaded_exp = FineFrequency.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqualExtended(exp, loaded_exp)

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = FineFrequency([0], 160)
        self.assertRoundTripSerializable(exp)

    def test_circuits_roundtrip_serializable(self):
        """Test circuits serialization of the experiment."""
        exp = FineFrequency([0], 160)
        self.assertRoundTripSerializable(exp._transpiled_circuits())
