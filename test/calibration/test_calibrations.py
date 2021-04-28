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
from qiskit.pulse import Drag, DriveChannel, ControlChannel, Gaussian, GaussianSquare
import qiskit.pulse as pulse
from qiskit.test import QiskitTestCase
from qiskit_experiments.calibration.calibrations import Calibrations, ParameterKey
from qiskit_experiments.calibration.parameter_value import ParameterValue
from qiskit_experiments.calibration.exceptions import CalibrationError


class TestCalibrationsBasic(QiskitTestCase):
    """Class to test the management of schedules and parameters for calibrations."""

    def setUp(self):
        """Setup a test environment."""
        self.cals = Calibrations()

        self.sigma = Parameter("σ")
        self.amp_xp = Parameter("amp")
        self.amp_x90p = Parameter("amp")
        self.amp_y90p = Parameter("amp")
        self.beta = Parameter("β")
        self.drive = DriveChannel(Parameter("ch0"))

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
            self.cals.add_schedule(sched)

        # Add some parameter values.
        self.date_time = datetime.strptime("15/09/19 10:21:35", "%d/%m/%y %H:%M:%S")

        self.cals.add_parameter_value(ParameterValue(40, self.date_time), "σ", schedule="xp")
        self.cals.add_parameter_value(ParameterValue(0.2, self.date_time), "amp", (3,), "xp")
        self.cals.add_parameter_value(ParameterValue(0.1, self.date_time), "amp", (3,), "x90p")
        self.cals.add_parameter_value(ParameterValue(0.08, self.date_time), "amp", (3,), "y90p")
        self.cals.add_parameter_value(ParameterValue(40, self.date_time), "β", (3,), "xp")

    def test_setup(self):
        """Test that the initial setup behaves as expected."""
        expected = {ParameterKey("xp", "amp", None), ParameterKey("xm", "amp", None)}
        self.assertEqual(self.cals.parameters[(self.amp_xp, hash(self.amp_xp))], expected)

        expected = {ParameterKey("x90p", "amp", None)}
        self.assertEqual(self.cals.parameters[(self.amp_x90p, hash(self.amp_x90p))], expected)

        expected = {ParameterKey("y90p", "amp", None)}
        self.assertEqual(self.cals.parameters[(self.amp_y90p, hash(self.amp_y90p))], expected)

        expected = {
            ParameterKey("xp", "β", None),
            ParameterKey("xm", "β", None),
            ParameterKey("x90p", "β", None),
            ParameterKey("y90p", "β", None),
        }
        self.assertEqual(self.cals.parameters[(self.beta, hash(self.beta))], expected)

        expected = {
            ParameterKey("xp", "σ", None),
            ParameterKey("xm", "σ", None),
            ParameterKey("x90p", "σ", None),
            ParameterKey("y90p", "σ", None),
        }
        self.assertEqual(self.cals.parameters[(self.sigma, hash(self.sigma))], expected)

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
        drive_0 = DriveChannel(Parameter("ch0"))
        drive_1 = DriveChannel(Parameter("ch1"))
        control_bad = ControlChannel(Parameter("u_chan"))
        control_good = ControlChannel(Parameter("ch1.0"))

        with pulse.build() as sched_good:
            pulse.play(Drag(160, 0.1, 40, 2), drive_0)
            pulse.play(Drag(160, 0.1, 40, 2), drive_1)
            pulse.play(Drag(160, 0.1, 40, 2), control_good)

        with pulse.build() as sched_bad:
            pulse.play(Drag(160, 0.1, 40, 2), drive_0)
            pulse.play(Drag(160, 0.1, 40, 2), drive_1)
            pulse.play(Drag(160, 0.1, 40, 2), control_bad)

        self.cals.add_schedule(sched_good)

        with self.assertRaises(CalibrationError):
            self.cals.add_schedule(sched_bad)

    def test_unique_parameter_names(self):
        """Test that we cannot insert schedules in which parameter names are duplicates."""
        with pulse.build() as sched:
            pulse.play(Drag(160, Parameter("a"), Parameter("a"), Parameter("a")), DriveChannel(0))

        with self.assertRaises(CalibrationError):
            self.cals.add_schedule(sched)

    def test_parameter_without_schedule(self):
        """Test that we can manage parameters that are not bound to a schedule."""
        self.cals._register_parameter(Parameter("a"))


class TestCalibrationDefaults(QiskitTestCase):
    """Test that we can override defaults."""

    def setUp(self):
        """Setup a few parameters."""
        self.cals = Calibrations()

        self.sigma = Parameter("σ")
        self.amp_xp = Parameter("amp")
        self.amp = Parameter("amp")
        self.beta = Parameter("β")
        self.drive = DriveChannel(Parameter("ch0"))
        self.date_time = datetime.strptime("15/09/19 10:21:35", "%d/%m/%y %H:%M:%S")

        # Template schedule for qubit 3
        with pulse.build(name="xp") as xp_drag:
            pulse.play(Drag(160, self.amp_xp, self.sigma, self.beta), self.drive)

        # Default template schedule for all qubits
        with pulse.build(name="xp") as xp:
            pulse.play(Gaussian(160, self.amp, self.sigma), self.drive)

        # Add the schedules
        self.cals.add_schedule(xp)
        self.cals.add_schedule(xp_drag, (3,))

    def test_parameter_value_adding_and_filtering(self):
        """Test that adding parameter values behaves in the expected way."""

        # Ensure that no parameter values are present when none have been added.
        params = self.cals.parameters_table()
        self.assertEqual(params, [])

        # Add a default parameter common to all qubits.
        self.cals.add_parameter_value(ParameterValue(40, self.date_time), "σ", None, "xp")
        self.assertEqual(len(self.cals.parameters_table()), 1)

        # Check that we can get a default parameter in the parameter table
        self.assertEqual(len(self.cals.parameters_table(parameters=["σ"])), 1)
        self.assertEqual(len(self.cals.parameters_table(parameters=["σ"], schedules=["xp"])), 1)
        self.assertEqual(len(self.cals.parameters_table(parameters=["σ"], schedules=["xm"])), 0)

        # Test behaviour of qubit-specific parameter and without ParameterValue.
        self.cals.add_parameter_value(0.25, "amp", (3,), "xp")
        self.cals.add_parameter_value(0.15, "amp", (0,), "xp")

        # Check the value for qubit 0
        params = self.cals.parameters_table(parameters=["amp"], qubit_list=[(0,)])
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0]["value"], 0.15)
        self.assertEqual(params[0]["qubits"], (0,))

        # Check the value for qubit 3
        params = self.cals.parameters_table(parameters=["amp"], qubit_list=[(3,)])
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0]["value"], 0.25)
        self.assertEqual(params[0]["qubits"], (3,))

    def _add_parameters(self):
        """Helper function."""

        # Add the minimum number of parameter values. Sigma is shared across both schedules.
        self.cals.add_parameter_value(ParameterValue(40, self.date_time), "σ", None, "xp")
        self.cals.add_parameter_value(ParameterValue(0.25, self.date_time), "amp", (3,), "xp")
        self.cals.add_parameter_value(ParameterValue(0.15, self.date_time), "amp", (0,), "xp")
        self.cals.add_parameter_value(ParameterValue(10, self.date_time), "β", (3,), "xp")

    def test_default_schedules(self):
        """
        In this test we create two xp schedules. A default schedules with a
        Gaussian pulse for all qubits and a Drag schedule for qubit three which
        should override the default schedule. We also test to see that updating
        a common parameter affects both schedules.
        """
        self._add_parameters()

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

        # Check that updating sigma updates both schedules.
        later_date_time = datetime.strptime("16/09/19 10:21:35", "%d/%m/%y %H:%M:%S")
        self.cals.add_parameter_value(ParameterValue(50, later_date_time), "σ", schedule="xp")

        xp0 = self.cals.get_schedule("xp", (0,))
        xp3 = self.cals.get_schedule("xp", (3,))

        self.assertEqual(xp0.instructions[0][1].pulse.sigma, 50)
        self.assertEqual(xp3.instructions[0][1].pulse.sigma, 50)

        # Check that we have the expected parameters in the calibrations.
        expected = {self.amp_xp, self.amp, self.sigma, self.beta}
        self.assertEqual(len(set(self.cals.parameters.keys())), len(expected))

    def test_parameter_filtering(self):
        """Test that we can properly filter parameter values."""

        self._add_parameters()

        # Check that these values are split between the qubits.
        amp_values = self.cals.parameters_table(parameters=["amp"], qubit_list=[(0,)])
        self.assertEqual(len(amp_values), 1)

        # Check that we have one value for sigma.
        sigma_values = self.cals.parameters_table(parameters=["σ"])
        self.assertEqual(len(sigma_values), 1)

        # Check that we have two values for amp.
        amp_values = self.cals.parameters_table(parameters=["amp"])
        self.assertEqual(len(amp_values), 2)

        amp_values = self.cals.parameters_table(parameters=["amp"], qubit_list=[(3,)])
        self.assertEqual(len(amp_values), 1)

        # Check to see if we get back the two qubits when explicitly specifying them.
        amp_values = self.cals.parameters_table(parameters=["amp"], qubit_list=[(3,), (0,)])
        self.assertEqual(len(amp_values), 2)


class TestInstructions(QiskitTestCase):
    """Class to test that instructions like Shift and Set Phase/Frequency are properly managed."""

    def setUp(self):
        """Create the setting to test."""
        self.phase = Parameter("φ")
        self.freq = Parameter("ν")
        self.d0_ = DriveChannel(Parameter("ch0"))

        with pulse.build(name="xp") as xp:
            pulse.play(Gaussian(160, 0.5, 40), self.d0_)

        with pulse.build(name="xp12") as xp12:
            pulse.shift_phase(self.phase, self.d0_)
            pulse.set_frequency(self.freq, self.d0_)
            pulse.play(Gaussian(160, 0.5, 40), self.d0_)

        # To make things more interesting we will use a call.
        with pulse.build(name="xp02") as xp02:
            pulse.call(xp)
            pulse.call(xp12)

        self.cals = Calibrations()
        self.cals.add_schedule(xp02)

        self.date_time = datetime.strptime("15/09/19 10:21:35", "%d/%m/%y %H:%M:%S")

        self.cals.add_parameter_value(ParameterValue(1.57, self.date_time), "φ", (3,), "xp12")
        self.cals.add_parameter_value(ParameterValue(200, self.date_time), "ν", (3,), "xp12")

    def test_call_registration(self):
        """Check that by registering the call we registered three schedules."""

        self.assertEqual(len(self.cals.schedules()), 3)

    def test_instructions(self):
        """Check that we get a properly assigned schedule."""

        sched = self.cals.get_schedule("xp02", (3, ))

        self.assertTrue(isinstance(sched.instructions[0][1], pulse.Play))
        self.assertEqual(sched.instructions[1][1].phase, 1.57)
        self.assertEqual(sched.instructions[2][1].frequency, 200)

class TestControlChannels(QiskitTestCase):
    """Test more complex schedules such as an echoed cross-resonance."""

    def setUp(self):
        """Create the setup we will deal with."""
        controls = {
            (3, 2): [ControlChannel(10), ControlChannel(123)],
            (2, 3): [ControlChannel(15), ControlChannel(23)],
        }
        self.cals = Calibrations(control_config=controls)

        self.amp_cr = Parameter("amp_cr")
        self.amp_rot = Parameter("amp_rot")
        self.amp = Parameter("amp")
        self.d0_ = DriveChannel(Parameter("ch0"))
        self.d1_ = DriveChannel(Parameter("ch1"))
        self.c1_ = ControlChannel(Parameter("ch0.1"))
        self.sigma = Parameter("σ")
        self.width = Parameter("w")
        self.date_time = datetime.strptime("15/09/19 10:21:35", "%d/%m/%y %H:%M:%S")

        cr_tone = GaussianSquare(640, self.amp_cr, self.sigma, self.width)
        rotary = GaussianSquare(640, self.amp_rot, self.sigma, self.width)

        with pulse.build(name="xp") as xp:
            pulse.play(Gaussian(160, self.amp, self.sigma), self.d0_)

        with pulse.build(name="cr") as cr:
            with pulse.align_sequential():
                with pulse.align_left():
                    pulse.play(rotary, self.d1_)  # Rotary tone
                    pulse.play(cr_tone, self.c1_)  # CR tone.
                with pulse.align_sequential():
                    pulse.call(xp)
                with pulse.align_left():
                    pulse.play(rotary, self.d1_)
                    pulse.play(cr_tone, self.c1_)
                with pulse.align_sequential():
                    pulse.call(xp)

        self.cals.add_schedule(xp)
        self.cals.add_schedule(cr)

        self.cals.add_parameter_value(ParameterValue(40, self.date_time), "σ", None, "xp")
        self.cals.add_parameter_value(ParameterValue(0.1, self.date_time), "amp", (3,), "xp")
        self.cals.add_parameter_value(ParameterValue(0.3, self.date_time), "amp_cr", (3, 2), "cr")
        self.cals.add_parameter_value(ParameterValue(0.2, self.date_time), "amp_rot", (3, 2), "cr")
        self.cals.add_parameter_value(ParameterValue(20, self.date_time), "w", (3, 2), "cr")

        # Reverse gate parameters
        self.cals.add_parameter_value(ParameterValue(0.15, self.date_time), "amp", (2,), "xp")
        self.cals.add_parameter_value(ParameterValue(0.5, self.date_time), "amp_cr", (2, 3), "cr")
        self.cals.add_parameter_value(ParameterValue(0.4, self.date_time), "amp_rot", (2, 3), "cr")
        self.cals.add_parameter_value(ParameterValue(30, self.date_time), "w", (2, 3), "cr")

    def test_get_schedule(self):
        """Check that we can get a CR schedule with a built in Call."""

        with pulse.build(name="cr") as cr_32:
            with pulse.align_sequential():
                with pulse.align_left():
                    pulse.play(GaussianSquare(640, 0.2, 40, 20), DriveChannel(2))  # Rotary tone
                    pulse.play(GaussianSquare(640, 0.3, 40, 20), ControlChannel(10))  # CR tone.
                with pulse.align_sequential():
                    pulse.play(Gaussian(160, 0.1, 40), DriveChannel(3))
                with pulse.align_left():
                    pulse.play(GaussianSquare(640, 0.2, 40, 20), DriveChannel(2))  # Rotary tone
                    pulse.play(GaussianSquare(640, 0.3, 40, 20), ControlChannel(10))  # CR tone.
                with pulse.align_sequential():
                    pulse.play(Gaussian(160, 0.1, 40), DriveChannel(3))

        self.assertTrue(self.cals.get_schedule("cr", (3, 2)) == cr_32)

        with pulse.build(name="cr") as cr_23:
            with pulse.align_sequential():
                with pulse.align_left():
                    pulse.play(GaussianSquare(640, 0.4, 40, 30), DriveChannel(3))  # Rotary tone
                    pulse.play(GaussianSquare(640, 0.5, 40, 30), ControlChannel(15))  # CR tone.
                with pulse.align_sequential():
                    pulse.play(Gaussian(160, 0.15, 40), DriveChannel(2))
                with pulse.align_left():
                    pulse.play(GaussianSquare(640, 0.4, 40, 30), DriveChannel(3))  # Rotary tone
                    pulse.play(GaussianSquare(640, 0.5, 40, 30), ControlChannel(15))  # CR tone.
                with pulse.align_sequential():
                    pulse.play(Gaussian(160, 0.15, 40), DriveChannel(2))

        self.assertTrue(self.cals.get_schedule("cr", (2, 3)) == cr_23)
