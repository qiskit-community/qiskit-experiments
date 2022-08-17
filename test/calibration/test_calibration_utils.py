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

import unittest
from test.base import QiskitExperimentsTestCase

from qiskit.circuit import Parameter
import qiskit.pulse as pulse

from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management import EchoCrossResonance
from qiskit_experiments.calibration_management.calibration_utils import (
    validate_channels,
    used_in_references,
)


class TestUsedInReference(QiskitExperimentsTestCase):
    """Test the function in CalUtils."""

    def setUp(self):
        """Setup the tests."""
        super().setUp()

        with pulse.build(name="xp") as xp:
            pulse.play(pulse.Gaussian(160, 0.5, 40), pulse.DriveChannel(1))

        with pulse.build(name="xp2") as xp2:
            pulse.play(pulse.Gaussian(160, 0.5, 40), pulse.DriveChannel(1))

        with pulse.build(name="ref_xp") as xp_ref:
            pulse.reference(xp.name, "q0")

        self.xp = xp
        self.xp2 = xp2
        self.xp_ref = xp_ref

    def test_used_in_references_simple(self):
        """Test that schedule identification by name with simple references."""
        self.assertSetEqual(used_in_references("xp", [self.xp_ref]), {"ref_xp"})
        self.assertSetEqual(used_in_references("xp", [self.xp2]), set())

    @unittest.skip("This test will fail as it is not supported yet.")
    def test_used_in_references_nested(self):
        """Test that schedule identification by name with nested references."""

        with pulse.build(name="ref_ref_xp") as xp_ref_ref:
            pulse.play(pulse.Drag(160, 0.5, 40, 0.2), pulse.DriveChannel(1))
            pulse.call(self.xp_ref)

        self.assertSetEqual(
            used_in_references("xp", [self.xp_ref, xp_ref_ref]), {"ref_xp", "ref_ref_xp"}
        )

    def test_usind_in_reference_cr(self):
        """Test a CR setting."""
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


class TestValidateChannels(QiskitExperimentsTestCase):
    """Test validate channels."""

    def test_ecr_lib(self):
        """Test channel validation."""

        # Test schedules with references.
        lib = EchoCrossResonance()

        self.assertEqual(validate_channels(lib["ecr"]), set())

        # Has a drive channel and a control channel that should be valid
        self.assertEqual(len(validate_channels(lib["cr45p"])), 2)

    def test_raise_on_multiple_parameters(self):
        """Test that an error is raised on a sum of parameters"""
        p1, p2 = Parameter("p1"), Parameter("p2")

        with pulse.build() as sched:
            pulse.play(pulse.Drag(160, 0.5, 40, 0), pulse.DriveChannel(p1 + p2))

        with self.assertRaises(CalibrationError):
            validate_channels(sched)

    def test_invalid_name(self):
        """Test that an error is raised on an invalid channel name."""
        with pulse.build() as sched:
            pulse.play(pulse.Drag(160, 0.5, 40, 0), pulse.DriveChannel(Parameter("0&1")))

        with self.assertRaises(CalibrationError):
            validate_channels(sched)
