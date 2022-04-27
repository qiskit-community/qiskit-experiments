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

from qiskit.test.mock import FakeArmonk
import qiskit.pulse as pulse

from qiskit_experiments.library import (
    FineFrequency,
    FineFrequencyCal,
)
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.test.mock_iq_backend import MockFineFreq


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

        self.cals = Calibrations.from_backend(FakeArmonk(), libraries=[FixedFrequencyTransmon()])

    @data(-0.5e6, -0.1e6, 0.1e6, 0.5e6)
    def test_end_to_end(self, freq_shift):
        """Test the experiment end to end."""

        backend = MockFineFreq(freq_shift, sx_duration=self.sx_duration)

        freq_exp = FineFrequency(0, 160, backend)
        freq_exp.set_transpile_options(inst_map=self.inst_map)

        expdata = freq_exp.run(shots=100)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results(1)
        d_theta = result.value.n
        dt = backend.configuration().dt
        d_freq = d_theta / (2 * np.pi * self.sx_duration * dt)

        tol = 0.01e6

        self.assertAlmostEqual(d_freq, freq_shift, delta=tol)
        self.assertEqual(result.quality, "good")

    def test_calibration_version(self):
        """Test the calibration version of the experiment."""

        freq_shift = 0.1e6
        backend = MockFineFreq(freq_shift, sx_duration=self.sx_duration)

        fine_freq = FineFrequencyCal(0, self.cals, backend)
        armonk_freq = FakeArmonk().defaults().qubit_freq_est[0]

        freq_before = self.cals.get_parameter_value(self.cals.__drive_freq_parameter__, 0)

        self.assertAlmostEqual(freq_before, armonk_freq)

        expdata = fine_freq.run()
        self.assertExperimentDone(expdata)

        freq_after = self.cals.get_parameter_value(self.cals.__drive_freq_parameter__, 0)

        # Test equality up to 10kHz on a 100 kHz shift
        self.assertAlmostEqual(freq_after, armonk_freq + freq_shift, delta=1e4)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = FineFrequency(0, 160)
        loaded_exp = FineFrequency.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = FineFrequency(0, 160)
        self.assertRoundTripSerializable(exp, self.json_equiv)
