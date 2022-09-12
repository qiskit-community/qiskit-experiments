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
import retworkx as rx

from qiskit.circuit import Parameter
import qiskit.pulse as pulse

from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management import EchoedCrossResonance
from qiskit_experiments.calibration_management.calibration_key_types import ScheduleKey
from qiskit_experiments.calibration_management.calibration_utils import (
    validate_channels,
    used_in_references,
    update_schedule_dependency,
)


class TestUsedInReference(QiskitExperimentsTestCase):
    """Test the function in CalUtils."""

    def setUp(self):
        """Setup the tests."""
        super().setUp()

        with pulse.build(name="xp") as xp1:
            pulse.play(pulse.Gaussian(160, 0.5, 40), pulse.DriveChannel(1))

        with pulse.build(name="xp2") as xp2:
            pulse.play(pulse.Gaussian(160, 0.5, 40), pulse.DriveChannel(1))

        with pulse.build(name="ref_xp") as xp_ref:
            pulse.reference(xp1.name, "q0")

        self.xp1 = xp1
        self.xp2 = xp2
        self.xp_ref = xp_ref

    def test_used_in_references_simple(self):
        """Test that schedule identification by name with simple references."""
        dag = rx.PyDiGraph(check_cycle=True)
        update_schedule_dependency(self.xp1, dag, ScheduleKey(self.xp1.name, tuple()))
        update_schedule_dependency(self.xp2, dag, ScheduleKey(self.xp2.name, tuple()))
        update_schedule_dependency(self.xp_ref, dag, ScheduleKey(self.xp_ref.name, tuple()))

        self.assertSetEqual(used_in_references({ScheduleKey("xp", tuple())}, dag), {"ref_xp"})
        self.assertSetEqual(used_in_references({ScheduleKey("xp2", tuple())}, dag), set())

    def test_used_in_references_nested(self):
        """Test that schedule identification by name with nested references."""

        with pulse.build(name="ref_ref_xp") as xp_ref_ref:
            pulse.play(pulse.Drag(160, 0.5, 40, 0.2), pulse.DriveChannel(1))
            pulse.call(self.xp_ref)

        dag = rx.PyDiGraph(check_cycle=True)
        for sched in [self.xp1, self.xp_ref, xp_ref_ref]:
            update_schedule_dependency(sched, dag, ScheduleKey(sched.name, tuple()))

        expected = {"ref_xp", "ref_ref_xp"}
        self.assertSetEqual(used_in_references({ScheduleKey("xp", tuple())}, dag), expected)

    def test_used_in_references(self):
        """Test a CR setting."""
        cr_tone_p = pulse.GaussianSquare(640, 0.2, 64, 500)
        cr_tone_m = pulse.GaussianSquare(640, -0.2, 64, 500)

        with pulse.build(name="cr") as cr:
            with pulse.align_sequential():
                with pulse.align_left():
                    pulse.play(cr_tone_p, pulse.ControlChannel(2))
                pulse.reference(self.xp1.name, "q0")
                with pulse.align_left():
                    pulse.play(cr_tone_m, pulse.ControlChannel(2))
                pulse.reference(self.xp1.name, "q0")

        dag = rx.PyDiGraph(check_cycle=True)
        update_schedule_dependency(self.xp1, dag, ScheduleKey(self.xp1.name, tuple()))
        update_schedule_dependency(cr, dag, ScheduleKey(cr.name, tuple()))

        self.assertSetEqual(used_in_references({ScheduleKey("xp", tuple())}, dag), {"cr"})


class TestValidateChannels(QiskitExperimentsTestCase):
    """Test validate channels."""

    def test_ecr_lib(self):
        """Test channel validation."""

        # Test schedules with references.
        lib = EchoedCrossResonance()

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
