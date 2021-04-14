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

"""Class to test the calibrations."""

from datetime import datetime
from qiskit.circuit import Parameter
from qiskit.pulse import Drag, DriveChannel, ControlChannel
from qiskit.test.mock import FakeAlmaden
import qiskit.pulse as pulse
from qiskit.test import QiskitTestCase
from qiskit_experiments.calibration.calibrations import Calibrations
from qiskit_experiments.calibration.parameter_value import ParameterValue
from qiskit_experiments.calibration.exceptions import CalibrationError


class TestCalibrationsBasic(QiskitTestCase):
    """Class to test the management of schedules and parameters for calibrations."""

    def setUp(self):
        """Setup a test environment."""
        backend = FakeAlmaden()
        self.cals = Calibrations(backend)

        self.sigma = Parameter("σ")
        self.amp_xp = Parameter("amp")
        self.amp_x90p = Parameter("amp")
        self.amp_y90p = Parameter("amp")
        self.beta = Parameter("β")
        drive = DriveChannel(Parameter("0"))

        # Define and add template schedules.
        with pulse.build(name="xp") as xp:
            pulse.play(Drag(160, self.amp_xp, self.sigma, self.beta), drive)

        with pulse.build(name="xm") as xm:
            pulse.play(Drag(160, -self.amp_xp, self.sigma, self.beta), drive)

        with pulse.build(name="x90p") as x90p:
            pulse.play(Drag(160, self.amp_x90p, self.sigma, self.beta), drive)

        with pulse.build(name="y90p") as y90p:
            pulse.play(Drag(160, self.amp_y90p, self.sigma, self.beta), drive)

        self.cals.add_schedules([xp, x90p, y90p, xm])

        # Add some parameter values.
        now = datetime.now
        self.cals.add_parameter_value(ParameterValue(40, now()), "σ", (3,), "xp")
        self.cals.add_parameter_value(ParameterValue(0.2, now()), "amp", (3,), "xp")
        self.cals.add_parameter_value(ParameterValue(0.1, now()), "amp", (3,), "x90p")
        self.cals.add_parameter_value(ParameterValue(0.08, now()), "amp", (3,), "y90p")
        self.cals.add_parameter_value(ParameterValue(40, now()), "β", (3,), "xp")

    def test_setup(self):
        """Test that the initial setup behaves as expected."""
        self.assertEqual(self.cals.parameters[self.amp_xp], {"xp", "xm"})
        self.assertEqual(self.cals.parameters[self.amp_x90p], {"x90p"})
        self.assertEqual(self.cals.parameters[self.amp_y90p], {"y90p"})
        self.assertEqual(self.cals.parameters[self.beta], {"xp", "xm", "x90p", "y90p"})
        self.assertEqual(self.cals.parameters[self.sigma], {"xp", "xm", "x90p", "y90p"})

        self.assertEqual(self.cals.get_parameter_value("amp", (3,), "xp"), 0.2)
        self.assertEqual(self.cals.get_parameter_value("amp", (3,), "xm"), 0.2)
        self.assertEqual(self.cals.get_parameter_value("amp", (3,), "x90p"), 0.1)
        self.assertEqual(self.cals.get_parameter_value("amp", (3,), "y90p"), 0.08)

    def test_parameter_dependency(self):
        """Check that two schedules that share the same parameter are simultaneously updated."""

        xp = self.cals.get_schedule("xp", (3,))
        self.assertEqual(xp.instructions[0][1].operands[0].amp, 0.2)

        xm = self.cals.get_schedule("xm", (3,))
        self.assertEqual(xm.instructions[0][1].operands[0].amp, -0.2)

        self.cals.add_parameter_value(ParameterValue(0.25, datetime.now()), "amp", (3,), "xp")

        xp = self.cals.get_schedule("xp", (3,))
        self.assertEqual(xp.instructions[0][1].operands[0].amp, 0.25)

        xm = self.cals.get_schedule("xm", (3,))
        self.assertEqual(xm.instructions[0][1].operands[0].amp, -0.25)

    def test_get_value(self):
        """Test the retrieve of parameter values."""

        self.assertEqual(self.cals.get_parameter_value("amp", (3,), "xp"), 0.2)
        self.assertEqual(self.cals.get_parameter_value("amp", (3,), "x90p"), 0.1)

        self.assertEqual(self.cals.get_parameter_value("σ", (3,), "x90p"), 40)
        self.assertEqual(self.cals.get_parameter_value("σ", (3,), "xp"), 40)

        self.cals.add_parameter_value(ParameterValue(50, datetime.now()), "σ", (3,), "xp")
        self.assertEqual(self.cals.get_parameter_value("σ", (3,), "x90p"), 50)
        self.assertEqual(self.cals.get_parameter_value("σ", (3,), "xp"), 50)

    def test_channel_names(self):
        """Check the naming of parametric control channels index1.index2.index3..."""
        drive_0 = DriveChannel(Parameter('0'))
        drive_1 = DriveChannel(Parameter('1'))
        control_bad = ControlChannel(Parameter('u_chan'))
        control_good = ControlChannel(Parameter('1.0'))

        with pulse.build() as sched_good:
            pulse.play(Drag(160, 0.1, 40, 2), drive_0)
            pulse.play(Drag(160, 0.1, 40, 2), drive_1)
            pulse.play(Drag(160, 0.1, 40, 2), control_good)

        with pulse.build() as sched_bad:
            pulse.play(Drag(160, 0.1, 40, 2), drive_0)
            pulse.play(Drag(160, 0.1, 40, 2), drive_1)
            pulse.play(Drag(160, 0.1, 40, 2), control_bad)

        self.cals.add_schedules(sched_good)

        with self.assertRaises(CalibrationError):
            self.cals.add_schedules(sched_bad)
