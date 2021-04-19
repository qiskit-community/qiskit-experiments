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
from qiskit.pulse import Drag, DriveChannel, ControlChannel, Gaussian
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
        self.drive = DriveChannel(Parameter("0"))

        # Define and add template schedules.
        with pulse.build(name="xp") as xp:
            pulse.play(Drag(160, self.amp_xp, self.sigma, self.beta), self.drive)

        with pulse.build(name="xm") as xm:
            pulse.play(Drag(160, -self.amp_xp, self.sigma, self.beta), self.drive)

        with pulse.build(name="x90p") as x90p:
            pulse.play(Drag(160, self.amp_x90p, self.sigma, self.beta), self.drive)

        with pulse.build(name="y90p") as y90p:
            pulse.play(Drag(160, self.amp_y90p, self.sigma, self.beta), self.drive)

        for sched in [xp, x90p, y90p, xm]:
            self.cals.add_schedules(sched)

        # Add some parameter values.
        self.date_time = datetime.strptime("15/09/19 10:21:35", "%d/%m/%y %H:%M:%S")

        self.cals.add_parameter_value(ParameterValue(40, self.date_time), "σ", None, "xp")
        self.cals.add_parameter_value(ParameterValue(0.2, self.date_time), "amp", (3,), "xp")
        self.cals.add_parameter_value(ParameterValue(0.1, self.date_time), "amp", (3,), "x90p")
        self.cals.add_parameter_value(ParameterValue(0.08, self.date_time), "amp", (3,), "y90p")
        self.cals.add_parameter_value(ParameterValue(40, self.date_time), "β", (3,), "xp")

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
        drive_0 = DriveChannel(Parameter("0"))
        drive_1 = DriveChannel(Parameter("1"))
        control_bad = ControlChannel(Parameter("u_chan"))
        control_good = ControlChannel(Parameter("1.0"))

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

    def test_unique_parameter_names(self):
        """Test that we cannot insert schedules in which parameter names are duplicates."""
        with pulse.build() as sched:
            pulse.play(Drag(160, Parameter("a"), Parameter("a"), Parameter("a")), DriveChannel(0))

        with self.assertRaises(CalibrationError):
            self.cals.add_schedules(sched)

    def test_parameter_without_schedule(self):
        """Test that we can manage parameters that are not bound to a schedule."""
        self.cals.register_parameter(Parameter("a"))


class TestCalibrationDefaults(QiskitTestCase):
    """Test that we can override defaults."""

    def setUp(self):
        """Setup a few parameters."""
        backend = FakeAlmaden()
        self.cals = Calibrations(backend)

        self.sigma = Parameter("σ")
        self.amp_xp = Parameter("amp")
        self.amp = Parameter("amp")
        self.beta = Parameter("β")
        self.drive = DriveChannel(Parameter("0"))
        self.date_time = datetime.strptime("15/09/19 10:21:35", "%d/%m/%y %H:%M:%S")

    def test_default_schedules(self):
        """
        In this test we create two xp schedules. A default schedules with a
        Gaussian pulse for all qubits and a Drag schedule for qubit three which
        should override the default schedule. We also test to see that updating
        a common parameter affects both schedules.
        """

        # Template schedule for qubit 3
        with pulse.build(name="xp") as xp_drag:
            pulse.play(Drag(160, self.amp_xp, self.sigma, self.beta), self.drive)

        # Default template schedule for all qubits
        amp = Parameter("amp")  # Same name as self.amp_xp
        with pulse.build(name="xp") as xp:
            pulse.play(Gaussian(160, self.amp, self.sigma), self.drive)

        # Add the schedules
        self.cals.add_schedules(xp)
        self.cals.add_schedules(xp_drag, (3,))

        # Add the minimum number of parameter values
        self.cals.add_parameter_value(ParameterValue(40, self.date_time), "σ", None, "xp")
        self.cals.add_parameter_value(ParameterValue(0.25, self.date_time), "amp", (3,), "xp")
        self.cals.add_parameter_value(ParameterValue(0.15, self.date_time), "amp", (0,), "xp")
        self.cals.add_parameter_value(ParameterValue(10, self.date_time), "β", (3,), "xp")

        xp0 = self.cals.get_schedule("xp", (0,))
        xp3 = self.cals.get_schedule("xp", (3,))

        # Check that xp0 is Play(Gaussian(160, 0.15, 40), 0)
        self.assertTrue(isinstance(xp0.instructions[0][1].pulse, Gaussian))
        self.assertEqual(xp0.instructions[0][1].channel, DriveChannel(0))
        self.assertEqual(xp0.instructions[0][1].pulse.amp, 0.15)
        self.assertEqual(xp0.instructions[0][1].pulse.sigma, 40)
        self.assertEqual(xp0.instructions[0][1].pulse.duration, 160)

        # Check that xp3 is Play(Drag(160, 0.25, 40, 10), 3)
        self.assertTrue(isinstance(xp3.instructions[0][1].pulse, Drag))
        self.assertEqual(xp3.instructions[0][1].channel, DriveChannel(3))
        self.assertEqual(xp3.instructions[0][1].pulse.amp, 0.25)
        self.assertEqual(xp3.instructions[0][1].pulse.sigma, 40)
        self.assertEqual(xp3.instructions[0][1].pulse.duration, 160)
        self.assertEqual(xp3.instructions[0][1].pulse.beta, 10)

        later_date_time = datetime.strptime("16/09/19 10:21:35", "%d/%m/%y %H:%M:%S")
        self.cals.add_parameter_value(ParameterValue(50, later_date_time), "σ", None, "xp")

        xp0 = self.cals.get_schedule("xp", (0,))
        xp3 = self.cals.get_schedule("xp", (3,))

        self.assertEqual(xp0.instructions[0][1].pulse.sigma, 50)
        self.assertEqual(xp3.instructions[0][1].pulse.sigma, 50)

        # Check that we have the expected parameters in the calibrations.
        expected = {self.amp_xp, self.amp, self.sigma, self.beta}
        self.assertEqual(len(set(self.cals.parameters.keys())), len(expected))
