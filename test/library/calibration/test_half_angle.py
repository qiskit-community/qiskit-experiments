# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test rough amplitude calibration experiment classes."""

from test.base import QiskitExperimentsTestCase

from qiskit import pulse
from qiskit.circuit import Parameter

from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.library import HalfAngleCal
from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend


class TestHalfAngleCal(QiskitExperimentsTestCase):
    """A class to test the half angle calibration experiments."""

    def setUp(self):
        """Setup the tests."""
        super().setUp()
        library = FixedFrequencyTransmon()

        self.backend = SingleTransmonTestBackend(noise=False, atol=1e-3)
        self.cals = Calibrations.from_backend(self.backend, libraries=[library])

    def test_amp_parameter_error(self):
        """Test that setting cal_parameter_name to amp raises an error"""
        with self.assertRaises(CalibrationError):
            HalfAngleCal([0], self.cals, cal_parameter_name="amp")

    def test_angle_parameter_missing_error(self):
        """Test that default cal_parameter_name with no matching parameter raises an error"""
        cals_no_angle = Calibrations()
        dur = Parameter("dur")
        amp = Parameter("amp")
        sigma = Parameter("σ")
        beta = Parameter("β")
        drive = pulse.DriveChannel(Parameter("ch0"))

        with pulse.build(name="sx") as sx:
            pulse.play(pulse.Drag(dur, amp, sigma, beta), drive)

        cals_no_angle.add_schedule(sx, num_qubits=1)
        with self.assertRaises(CalibrationError):
            HalfAngleCal([0], cals_no_angle)

    def test_circuits_roundtrip_serializable(self):
        """Test circuits serialization of the experiment."""
        exp = HalfAngleCal([0], self.cals, backend=self.backend)
        self.assertRoundTripSerializable(exp._transpiled_circuits())
