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

"""Class to test utility functions for calibrations."""

from test.base import QiskitExperimentsTestCase
import qiskit.pulse as pulse
from qiskit_experiments.calibration_management.calibration_utils import used_in_references


class TestCalibrationUtils(QiskitExperimentsTestCase):
    """Test the function in CalUtils."""

    def test_used_in_calls(self):
        """Test that we can identify schedules by name when calls are present."""

        with pulse.build(name="xp") as xp:
            pulse.play(pulse.Gaussian(160, 0.5, 40), pulse.DriveChannel(1))

        with pulse.build(name="xp2") as xp2:
            pulse.play(pulse.Gaussian(160, 0.5, 40), pulse.DriveChannel(1))

        with pulse.build(name="call_xp") as xp_call:
            pulse.call(xp)

        with pulse.build(name="call_call_xp") as xp_call_call:
            pulse.play(pulse.Drag(160, 0.5, 40, 0.2), pulse.DriveChannel(1))
            pulse.call(xp_call)

        self.assertSetEqual(used_in_references("xp", [xp_call]), {"call_xp"})
        self.assertSetEqual(used_in_references("xp", [xp2]), set())
        self.assertSetEqual(
            used_in_references("xp", [xp_call, xp_call_call]), {"call_xp", "call_call_xp"}
        )

        with pulse.build(name="xp") as xp:
            pulse.play(pulse.Gaussian(160, 0.5, 40), pulse.DriveChannel(2))

        cr_tone_p = pulse.GaussianSquare(640, 0.2, 64, 500)
        rotary_p = pulse.GaussianSquare(640, 0.1, 64, 500)

        cr_tone_m = pulse.GaussianSquare(640, -0.2, 64, 500)
        rotary_m = pulse.GaussianSquare(640, -0.1, 64, 500)

        with pulse.build(name="cr") as cr:
            with pulse.align_sequential():
                with pulse.align_left():
                    pulse.play(rotary_p, pulse.DriveChannel(3))  # Rotary tone
                    pulse.play(cr_tone_p, pulse.ControlChannel(2))  # CR tone.
                pulse.call(xp)
                with pulse.align_left():
                    pulse.play(rotary_m, pulse.DriveChannel(3))
                    pulse.play(cr_tone_m, pulse.ControlChannel(2))
                pulse.call(xp)

        self.assertSetEqual(used_in_references("xp", [cr]), {"cr"})
