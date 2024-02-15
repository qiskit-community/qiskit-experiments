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

from qiskit import pulse
from qiskit_ibm_runtime.fake_provider import FakeArmonkV2

from qiskit_experiments.library import (
    FineFrequency,
    FineFrequencyCal,
)
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.framework import BackendData
from qiskit_experiments.test.mock_iq_backend import MockIQBackend
from qiskit_experiments.test.mock_iq_helpers import MockIQFineFreqHelper as FineFreqHelper


@ddt
class TestFineFreqEndToEnd(QiskitExperimentsTestCase):
    """Test the fine freq experiment."""

    def setUp(self):
        """Setup for the test."""
        super().setUp()
        self.inst_map = pulse.InstructionScheduleMap()

        self.sx_duration = 160

        with pulse.build(name="sx") as sx_sched:
            pulse.play(pulse.Gaussian(self.sx_duration, 0.5, 40), pulse.DriveChannel(0))

        self.inst_map.add("sx", 0, sx_sched)

        self.cals = Calibrations.from_backend(FakeArmonkV2(), libraries=[FixedFrequencyTransmon()])

    @data(-0.5e6, -0.1e6, 0.1e6, 0.5e6)
    def test_end_to_end(self, freq_shift):
        """Test the experiment end to end."""
        exp_helper = FineFreqHelper(sx_duration=self.sx_duration, freq_shift=freq_shift)
        backend = MockIQBackend(exp_helper)
        exp_helper.dt = BackendData(backend).dt

        freq_exp = FineFrequency([0], 160, backend)
        freq_exp.set_transpile_options(inst_map=self.inst_map)

        expdata = freq_exp.run(shots=100)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results("d_theta")
        d_theta = result.value.n
        dt = BackendData(backend).dt
        d_freq = d_theta / (2 * np.pi * self.sx_duration * dt)

        tol = 0.01e6

        self.assertAlmostEqual(d_freq, freq_shift, delta=tol)
        self.assertEqual(result.quality, "good")

    def test_calibration_version(self):
        """Test the calibration version of the experiment."""

        exp_helper = FineFreqHelper(sx_duration=self.sx_duration, freq_shift=0.1e6)
        backend = MockIQBackend(exp_helper)
        exp_helper.dt = BackendData(backend).dt

        fine_freq = FineFrequencyCal([0], self.cals, backend)
        armonk_freq = BackendData(FakeArmonkV2()).drive_freqs[0]

        freq_before = self.cals.get_parameter_value("drive_freq", 0)

        self.assertAlmostEqual(freq_before, armonk_freq)

        expdata = fine_freq.run()
        self.assertExperimentDone(expdata)

        freq_after = self.cals.get_parameter_value("drive_freq", 0)

        # Test equality up to 10kHz on a 100 kHz shift
        self.assertAlmostEqual(freq_after, armonk_freq + exp_helper.freq_shift, delta=1e4)

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
