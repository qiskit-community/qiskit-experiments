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

"""Test the fine amplitude calibration experiment."""

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.pulse import DriveChannel, Drag
import qiskit.pulse as pulse

from qiskit_experiments.calibration.fine_amplitude import FineAmplitude
from qiskit_experiments.test.mock_iq_backend import MockFineAmp
from qiskit_experiments.exceptions import CalibrationError


class TestFineAmpEndToEnd(QiskitTestCase):
    """Test the drag experiment."""

    def setUp(self):
        """Setup some schedules."""
        super().setUp()

        with pulse.build(name="xp") as xp:
            pulse.play(Drag(duration=160, amp=0.208519, sigma=40, beta=0.0), DriveChannel(0))

        self.x_plus = xp

    def test_end_to_end_under_rotation(self):
        """Test the experiment end to end."""

        amp_cal = FineAmplitude(0)
        amp_cal.set_schedule(
            schedule=self.x_plus, angle_per_gate=np.pi, add_xp_circuit=True, add_sx=True
        )
        amp_cal.set_analysis_options(number_guesses=11)

        backend = MockFineAmp(-np.pi * 0.07, np.pi, "xp")

        result = amp_cal.run(backend).analysis_result(-1)

        d_theta = result["popt"][result["popt_keys"].index("d_theta")]

        tol = 0.04

        self.assertTrue(abs(d_theta - backend.angle_error) < tol)
        self.assertEqual(result["quality"], "computer_good")

    def test_end_to_end_over_rotation(self):
        """Test the experiment end to end."""

        amp_cal = FineAmplitude(0)
        amp_cal.set_schedule(
            schedule=self.x_plus, angle_per_gate=np.pi, add_xp_circuit=True, add_sx=True
        )
        amp_cal.set_analysis_options(number_guesses=6)

        backend = MockFineAmp(np.pi * 0.07, np.pi, "xp")

        result = amp_cal.run(backend).analysis_result(-1)

        d_theta = result["popt"][result["popt_keys"].index("d_theta")]

        tol = 0.04

        self.assertTrue(abs(d_theta - backend.angle_error) < tol)
        self.assertEqual(result["quality"], "computer_good")

    def test_zero_angle_per_gate(self):
        """Test that we cannot set angle per gate to zero."""
        amp_cal = FineAmplitude(0)

        with self.assertRaises(CalibrationError):
            amp_cal.set_schedule(
                schedule=self.x_plus, angle_per_gate=0.0, add_xp_circuit=True, add_sx=True
            )


class TestFineAmplitudeCircuits(QiskitTestCase):
    """Test the circuits."""

    def setUp(self):
        """Setup some schedules."""
        super().setUp()

        with pulse.build(name="xp") as xp:
            pulse.play(Drag(duration=160, amp=0.208519, sigma=40, beta=0.0), DriveChannel(0))

        with pulse.build(name="x90p") as x90p:
            pulse.play(Drag(duration=160, amp=0.208519, sigma=40, beta=0.0), DriveChannel(0))

        self.x_plus = xp
        self.x_90_plus = x90p

    def test_xp(self):
        """Test a circuit with xp."""

        amp_cal = FineAmplitude(0)
        amp_cal.set_schedule(
            schedule=self.x_plus, angle_per_gate=np.pi, add_xp_circuit=False, add_sx=True
        )

        for idx, circ in enumerate(amp_cal.circuits()):
            self.assertTrue(circ.data[0][0].name == "sx")
            self.assertEqual(circ.count_ops().get("xp", 0), idx)

    def test_x90p(self):
        """Test circuits with an x90p pulse."""

        amp_cal = FineAmplitude(0)
        amp_cal.set_schedule(
            schedule=self.x_90_plus, angle_per_gate=np.pi, add_xp_circuit=False, add_sx=False
        )

        for idx, circ in enumerate(amp_cal.circuits()):
            self.assertTrue(circ.data[0][0].name != "sx")
            self.assertEqual(circ.count_ops().get("x90p", 0), idx)
