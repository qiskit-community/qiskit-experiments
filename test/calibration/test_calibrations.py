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

import os
from collections import defaultdict
from datetime import datetime
from qiskit.circuit import Parameter
from qiskit.pulse import (
    Drag,
    DriveChannel,
    ControlChannel,
    AcquireChannel,
    Gaussian,
    GaussianSquare,
    MeasureChannel,
    RegisterSlot,
    Play,
)
from qiskit.pulse.transforms import inline_subroutines, block_to_schedule
import qiskit.pulse as pulse
from qiskit.test import QiskitTestCase
from qiskit_experiments.calibration.calibrations import Calibrations, ParameterKey
from qiskit_experiments.calibration.parameter_value import ParameterValue
from qiskit_experiments.calibration.exceptions import CalibrationError


class TestCalibrationsBasic(QiskitTestCase):
    """Class to test the management of schedules and parameters for calibrations."""

    def setUp(self):
        """Create the setting to test."""
        super().setUp()

        self.cals = Calibrations()

        self.sigma = Parameter("σ")
        self.amp_xp = Parameter("amp")
        self.amp_x90p = Parameter("amp")
        self.amp_y90p = Parameter("amp")
        self.beta = Parameter("β")
        self.chan = Parameter("ch0")
        self.drive = DriveChannel(self.chan)
        self.duration = Parameter("dur")

        # Define and add template schedules.
        with pulse.build(name="xp") as xp:
            pulse.play(Drag(self.duration, self.amp_xp, self.sigma, self.beta), self.drive)

        with pulse.build(name="xm") as xm:
            pulse.play(Drag(self.duration, -self.amp_xp, self.sigma, self.beta), self.drive)

        with pulse.build(name="x90p") as x90p:
            pulse.play(Drag(self.duration, self.amp_x90p, self.sigma, self.beta), self.drive)

        with pulse.build(name="y90p") as y90p:
            pulse.play(Drag(self.duration, self.amp_y90p, self.sigma, self.beta), self.drive)

        for sched in [xp, x90p, y90p, xm]:
            self.cals.add_schedule(sched)

        self.xm_pulse = xm

        # Add some parameter values.
        self.date_time = datetime.strptime("15/09/19 10:21:35", "%d/%m/%y %H:%M:%S")

        self.cals.add_parameter_value(ParameterValue(40, self.date_time), "σ", schedule="xp")
        self.cals.add_parameter_value(ParameterValue(160, self.date_time), "dur", schedule="xp")
        self.cals.add_parameter_value(ParameterValue(0.2, self.date_time), "amp", 3, "xp")
        self.cals.add_parameter_value(ParameterValue(0.1, self.date_time), "amp", (3,), "x90p")
        self.cals.add_parameter_value(ParameterValue(0.08, self.date_time), "amp", (3,), "y90p")
        self.cals.add_parameter_value(ParameterValue(40, self.date_time), "β", (3,), "xp")

    def test_setup(self):
        """Test that the initial setup behaves as expected."""
        expected = {ParameterKey("amp", (), "xp"), ParameterKey("amp", (), "xm")}
        self.assertEqual(self.cals.parameters[self.amp_xp], expected)

        expected = {ParameterKey("amp", (), "x90p")}
        self.assertEqual(self.cals.parameters[self.amp_x90p], expected)

        expected = {ParameterKey("amp", (), "y90p")}
        self.assertEqual(self.cals.parameters[self.amp_y90p], expected)

        expected = {
            ParameterKey("β", (), "xp"),
            ParameterKey("β", (), "xm"),
            ParameterKey("β", (), "x90p"),
            ParameterKey("β", (), "y90p"),
        }
        self.assertEqual(self.cals.parameters[self.beta], expected)

        expected = {
            ParameterKey("σ", (), "xp"),
            ParameterKey("σ", (), "xm"),
            ParameterKey("σ", (), "x90p"),
            ParameterKey("σ", (), "y90p"),
        }
        self.assertEqual(self.cals.parameters[self.sigma], expected)

        self.assertEqual(self.cals.get_parameter_value("amp", (3,), "xp"), 0.2)
        self.assertEqual(self.cals.get_parameter_value("amp", (3,), "xm"), 0.2)
        self.assertEqual(self.cals.get_parameter_value("amp", 3, "x90p"), 0.1)
        self.assertEqual(self.cals.get_parameter_value("amp", 3, "y90p"), 0.08)

    def test_preserve_template(self):
        """Test that the template schedule is still fully parametric after we get a schedule."""

        # First get a schedule
        xp = self.cals.get_schedule("xp", (3,))
        self.assertEqual(xp.instructions[0][1].operands[0].amp, 0.2)

        # Find the template schedule for xp and test it.
        schedule = pulse.Schedule()
        for sched_dict in self.cals.schedules():
            if sched_dict["schedule"].name == "xp":
                schedule = sched_dict["schedule"]

        for param in {self.amp_xp, self.sigma, self.beta, self.duration, self.chan}:
            self.assertTrue(param in schedule.parameters)

        self.assertEqual(len(schedule.parameters), 5)
        self.assertEqual(len(schedule.blocks), 1)

    def test_remove_schedule(self):
        """Test that we can easily remove a schedule."""

        self.assertEqual(len(self.cals.schedules()), 4)

        self.cals.remove_schedule(self.xm_pulse)

        # Removing xm should remove the schedule but not the parameters as they are shared.
        self.assertEqual(len(self.cals.schedules()), 3)
        for param in [self.sigma, self.amp_xp, self.amp_x90p, self.amp_y90p, self.beta]:
            self.assertTrue(param in self.cals.parameters)

        # Add a schedule with a different parameter and then remove it
        with pulse.build(name="error") as sched:
            pulse.play(Gaussian(160, Parameter("xyz"), 40), DriveChannel(Parameter("ch0")))

        self.cals.add_schedule(sched)

        self.assertEqual(len(self.cals.schedules()), 4)
        self.assertEqual(len(self.cals.parameters), 7)

        self.cals.remove_schedule(sched)

        self.assertEqual(len(self.cals.schedules()), 3)
        self.assertEqual(len(self.cals.parameters), 6)
        for param in [self.sigma, self.amp_xp, self.amp_x90p, self.amp_y90p, self.beta]:
            self.assertTrue(param in self.cals.parameters)

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

        with pulse.build(name="good_sched") as sched_good:
            pulse.play(Drag(160, 0.1, 40, 2), drive_0)
            pulse.play(Drag(160, 0.1, 40, 2), drive_1)
            pulse.play(Drag(160, 0.1, 40, 2), control_good)

        with pulse.build(name="bad_sched") as sched_bad:
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
        self.cals._register_parameter(Parameter("a"), ())

    def test_free_parameters(self):
        """Test that we can get a schedule with a free parameter."""
        xp = self.cals.get_schedule("xp", 3, assign_params={"amp": self.amp_xp})
        self.assertEqual(xp.parameters, {self.amp_xp})

        xp = self.cals.get_schedule("xp", 3, assign_params={"amp": self.amp_xp, "σ": self.sigma})
        self.assertEqual(xp.parameters, {self.amp_xp, self.sigma})

    def test_qubit_input(self):
        """Test the qubit input."""

        xp = self.cals.get_schedule("xp", 3)
        self.assertEqual(xp.instructions[0][1].operands[0].amp, 0.2)

        val = self.cals.get_parameter_value("amp", 3, "xp")
        self.assertEqual(val, 0.2)

        val = self.cals.get_parameter_value("amp", (3,), "xp")
        self.assertEqual(val, 0.2)

        with self.assertRaises(CalibrationError):
            self.cals.get_parameter_value("amp", ("3",), "xp")

        val = self.cals.get_parameter_value("amp", "3", "xp")
        self.assertEqual(val, 0.2)

        with self.assertRaises(CalibrationError):
            self.cals.get_parameter_value("amp", "(1, a)", "xp")

        with self.assertRaises(CalibrationError):
            self.cals.get_parameter_value("amp", [3], "xp")


class TestOverrideDefaults(QiskitTestCase):
    """
    Test that we can override defaults. For example, this means that all qubits may have a
    Gaussian as xp pulse but a specific qubit may have a Drag pulse which overrides the
    default Gaussian.
    """

    def setUp(self):
        """Create the setting to test."""
        super().setUp()

        self.cals = Calibrations()

        self.sigma = Parameter("σ")
        self.amp_xp = Parameter("amp")
        self.amp = Parameter("amp")
        self.beta = Parameter("β")
        self.drive = DriveChannel(Parameter("ch0"))
        self.date_time = datetime.strptime("15/09/19 10:21:35", "%d/%m/%y %H:%M:%S")
        self.duration = Parameter("dur")

        # Template schedule for qubit 3
        with pulse.build(name="xp") as xp_drag:
            pulse.play(Drag(self.duration, self.amp_xp, self.sigma, self.beta), self.drive)

        # Default template schedule for all qubits
        with pulse.build(name="xp") as xp:
            pulse.play(Gaussian(self.duration, self.amp, self.sigma), self.drive)

        # Add the schedules
        self.cals.add_schedule(xp)
        self.cals.add_schedule(xp_drag, (3,))

    def test_parameter_value_adding_and_filtering(self):
        """Test that adding parameter values behaves in the expected way."""

        # Ensure that no parameter values are present when none have been added.
        params = self.cals.parameters_table()
        self.assertEqual(params, [])

        # Add a default parameter common to all qubits.
        self.cals.add_parameter_value(ParameterValue(40, self.date_time), "σ", schedule="xp")
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
        self.cals.add_parameter_value(ParameterValue(40, self.date_time), "σ", schedule="xp")
        self.cals.add_parameter_value(ParameterValue(0.25, self.date_time), "amp", (3,), "xp")
        self.cals.add_parameter_value(ParameterValue(0.15, self.date_time), "amp", (0,), "xp")
        self.cals.add_parameter_value(ParameterValue(10, self.date_time), "β", (3,), "xp")
        self.cals.add_parameter_value(160, "dur", schedule="xp")

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
        expected = {self.amp_xp, self.amp, self.sigma, self.beta, self.duration}
        self.assertEqual(len(set(self.cals.parameters.keys())), len(expected))

    def test_replace_schedule(self):
        """Test that schedule replacement works as expected."""

        self.cals.add_parameter_value(ParameterValue(0.25, self.date_time), "amp", (3,), "xp")
        self.cals.add_parameter_value(ParameterValue(40, self.date_time), "σ", schedule="xp")
        self.cals.add_parameter_value(ParameterValue(10, self.date_time), "β", (3,), "xp")

        # Let's replace the schedule for qubit 3 with a double Drag pulse.
        with pulse.build(name="xp") as sched:
            pulse.play(Drag(160, self.amp_xp / 2, self.sigma, self.beta), self.drive)
            pulse.play(Drag(160, self.amp_xp / 2, self.sigma, self.beta), self.drive)

        expected = self.cals.parameters

        # Adding this new schedule should not change the parameter mapping
        self.cals.add_schedule(sched, (3,))

        self.assertEqual(self.cals.parameters, expected)

        # For completeness we check that schedule that comes out.
        sched_cal = self.cals.get_schedule("xp", (3,))

        self.assertTrue(isinstance(sched_cal.instructions[0][1].pulse, Drag))
        self.assertTrue(isinstance(sched_cal.instructions[1][1].pulse, Drag))
        self.assertEqual(sched_cal.instructions[0][1].pulse.amp, 0.125)
        self.assertEqual(sched_cal.instructions[1][1].pulse.amp, 0.125)

        # Let's replace the schedule for qubit 3 with a Gaussian pulse.
        # This should change the parameter mapping
        with pulse.build(name="xp") as sched2:
            pulse.play(Gaussian(160, self.amp_xp / 2, self.sigma), self.drive)

        # Check that beta is in the mapping
        self.assertEqual(
            self.cals.parameters[self.beta],
            {ParameterKey("β", (3,), "xp")},
        )

        self.cals.add_schedule(sched2, (3,))

        # Check that beta no longer maps to a schedule
        self.assertEqual(self.cals.parameters[self.beta], set())

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


class TestMeasurements(QiskitTestCase):
    """Test that schedules on measure channels are handled properly."""

    def setUp(self):
        """Create the setting to test."""
        super().setUp()

        self.amp = Parameter("amp")
        self.amp_xp = Parameter("amp")
        self.sigma = Parameter("σ")
        self.sigma_xp = Parameter("σ")
        self.width = Parameter("w")
        self.duration = Parameter("dur")
        self.duration_xp = Parameter("dur")
        ch0 = Parameter("ch0")
        ch1 = Parameter("ch1")
        self.m0_ = MeasureChannel(ch0)
        self.d0_ = DriveChannel(ch0)
        self.delay = Parameter("delay")

        with pulse.build(name="meas") as meas:
            pulse.play(GaussianSquare(self.duration, self.amp, self.sigma, self.width), self.m0_)

        with pulse.build(name="meas_acquire") as meas_acq:
            pulse.play(GaussianSquare(self.duration, self.amp, self.sigma, self.width), self.m0_)
            pulse.delay(self.delay, pulse.AcquireChannel(ch0))
            pulse.acquire(self.duration, pulse.AcquireChannel(ch0), pulse.RegisterSlot(ch0))

        with pulse.build(name="xp") as xp:
            pulse.play(Gaussian(self.duration_xp, self.amp_xp, self.sigma_xp), self.d0_)

        with pulse.build(name="xp_meas") as xp_meas:
            pulse.call(xp)
            pulse.call(meas)

        with pulse.build(name="xt_meas") as xt_meas:
            with pulse.align_sequential():
                pulse.call(xp)
                pulse.call(meas)
            with pulse.align_sequential():
                pulse.call(xp, value_dict={ch0: ch1})
                pulse.call(meas, value_dict={ch0: ch1})

        self.cals = Calibrations()
        self.cals.add_schedule(meas)
        self.cals.add_schedule(xp)
        self.cals.add_schedule(xp_meas)
        self.cals.add_schedule(xt_meas)
        self.cals.add_schedule(meas_acq)

        # self.cals.add_parameter_value(8000, self.duration, schedule="meas")
        self.cals.add_parameter_value(0.5, self.amp, (0,), "meas")
        self.cals.add_parameter_value(0.56, self.amp, (123,), "meas")
        self.cals.add_parameter_value(0.3, self.amp, (2,), "meas")
        self.cals.add_parameter_value(160, self.sigma, schedule="meas")
        self.cals.add_parameter_value(7000, self.width, schedule="meas")
        self.cals.add_parameter_value(8000, self.duration, schedule="meas")
        self.cals.add_parameter_value(100, self.delay, schedule="meas_acquire")

        self.cals.add_parameter_value(0.9, self.amp_xp, (0,), "xp")
        self.cals.add_parameter_value(0.7, self.amp_xp, (2,), "xp")
        self.cals.add_parameter_value(40, self.sigma_xp, schedule="xp")
        self.cals.add_parameter_value(160, self.duration_xp, schedule="xp")

    def test_meas_schedule(self):
        """Test that we get a properly assigned measure schedule without drive channels."""
        sched = self.cals.get_schedule("meas", (0,))
        meas = Play(GaussianSquare(8000, 0.5, 160, 7000), MeasureChannel(0))
        self.assertTrue(sched.instructions[0][1], meas)

        sched = self.cals.get_schedule("meas", (2,))
        meas = Play(GaussianSquare(8000, 0.3, 160, 7000), MeasureChannel(0))
        self.assertTrue(sched.instructions[0][1], meas)

    def test_call_meas(self):
        """Test that we can call a measurement pulse."""
        sched = self.cals.get_schedule("xp_meas", (0,))
        xp = Play(Gaussian(160, 0.9, 40), DriveChannel(0))
        meas = Play(GaussianSquare(8000, 0.5, 160, 7000), MeasureChannel(0))

        self.assertTrue(sched.instructions[0][1], xp)
        self.assertTrue(sched.instructions[1][1], meas)

    def test_xt_meas(self):
        """Test that creating multi-qubit schedules out of calls works."""

        sched = self.cals.get_schedule("xt_meas", (0, 2))

        xp0 = Play(Gaussian(160, 0.9, 40), DriveChannel(0))
        xp2 = Play(Gaussian(160, 0.7, 40), DriveChannel(2))

        meas0 = Play(GaussianSquare(8000, 0.5, 160, 7000), MeasureChannel(0))
        meas2 = Play(GaussianSquare(8000, 0.3, 160, 7000), MeasureChannel(2))

        self.assertEqual(sched.instructions[0][1], xp0)
        self.assertEqual(sched.instructions[1][1], xp2)
        self.assertEqual(sched.instructions[2][1], meas0)
        self.assertEqual(sched.instructions[3][1], meas2)

    def test_free_parameters(self):
        """Test that we can get a schedule with free parameters."""

        # Test coupling breaking
        my_amp = Parameter("my_amp")
        schedule = self.cals.get_schedule(
            "xt_meas",
            (0, 2),
            assign_params={("amp", (0,), "xp"): my_amp},
        )

        schedule = block_to_schedule(schedule)

        with pulse.build(name="xt_meas") as expected:
            with pulse.align_sequential():
                pulse.play(Gaussian(160, my_amp, 40), DriveChannel(0))
                pulse.play(GaussianSquare(8000, 0.5, 160, 7000), MeasureChannel(0))
            with pulse.align_sequential():
                pulse.play(Gaussian(160, 0.7, 40), DriveChannel(2))
                pulse.play(GaussianSquare(8000, 0.3, 160, 7000), MeasureChannel(2))

        expected = block_to_schedule(expected)

        self.assertEqual(schedule.parameters, {my_amp})
        self.assertEqual(schedule, expected)

    def test_free_parameters_check(self):
        """
        Test that get_schedule raises an error if the number of parameters does not match.
        This test ensures that we forbid ambiguity in free parameters in schedules with
        calls that share parameters.
        """

        amp1 = Parameter("amp1")
        amp2 = Parameter("amp2")
        assign_dict = {("amp", (0,), "xp"): amp1, ("amp", (2,), "xp"): amp2}

        sched = self.cals.get_schedule("xt_meas", (0, 2), assign_params=assign_dict)

        self.assertEqual(sched.parameters, {amp1, amp2})

        sched = block_to_schedule(sched)

        self.assertEqual(sched.instructions[0][1].parameters, {amp1})
        self.assertEqual(sched.instructions[1][1].parameters, {amp2})

    def test_measure_and_acquire(self):
        """Test that we can get a measurement schedule with an acquire instruction."""

        sched = self.cals.get_schedule("meas_acquire", (123,))

        with pulse.build(name="meas_acquire") as expected:
            pulse.play(GaussianSquare(8000, 0.56, 160, 7000), MeasureChannel(123))
            pulse.delay(100, AcquireChannel(123))
            pulse.acquire(8000, AcquireChannel(123), RegisterSlot(123))

        self.assertEqual(sched, expected)


class TestInstructions(QiskitTestCase):
    """Class to test that instructions like Shift and Set Phase/Frequency are properly managed."""

    def setUp(self):
        """Create the setting to test."""
        super().setUp()

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
        self.cals.add_schedule(xp)
        self.cals.add_schedule(xp12)
        self.cals.add_schedule(xp02)

        self.date_time = datetime.strptime("15/09/19 10:21:35", "%d/%m/%y %H:%M:%S")

        self.cals.add_parameter_value(ParameterValue(1.57, self.date_time), "φ", (3,), "xp12")
        self.cals.add_parameter_value(ParameterValue(200, self.date_time), "ν", (3,), "xp12")

    def test_call_registration(self):
        """Check that by registering the call we registered three schedules."""

        self.assertEqual(len(self.cals.schedules()), 3)

    def test_instructions(self):
        """Check that we get a properly assigned schedule."""

        sched = self.cals.get_schedule("xp02", (3,))

        self.assertEqual(sched.parameters, set())

        sched = inline_subroutines(sched)  # inline makes the check more transparent.

        self.assertTrue(isinstance(sched.instructions[0][1], pulse.Play))
        self.assertEqual(sched.instructions[1][1].phase, 1.57)
        self.assertEqual(sched.instructions[2][1].frequency, 200)


class TestRegistering(QiskitTestCase):
    """Class to test registering of subroutines with calls."""

    def setUp(self):
        """Create the setting to test."""
        super().setUp()

        self.cals = Calibrations()
        self.d0_ = DriveChannel(Parameter("ch0"))

    def test_call_registering(self):
        """Test registering of schedules with call."""
        with pulse.build(name="xp") as xp:
            pulse.play(Gaussian(160, 0.5, 40), self.d0_)

        with pulse.build(name="call_xp") as call_xp:
            pulse.call(xp)

        with self.assertRaises(CalibrationError):
            self.cals.add_schedule(call_xp)

        self.cals.add_schedule(xp)
        self.cals.add_schedule(call_xp)

        self.assertTrue(isinstance(self.cals.get_schedule("call_xp", 2), pulse.ScheduleBlock))

    def test_get_template(self):
        """Test that we can get a registered template and use it."""
        amp = Parameter("amp")

        with pulse.build(name="xp") as xp:
            pulse.play(Gaussian(160, amp, 40), self.d0_)

        self.cals.add_schedule(xp)

        registered_xp = self.cals.get_template("xp")

        self.assertEqual(registered_xp, xp)

        with pulse.build(name="dxp") as dxp:
            pulse.call(registered_xp)
            pulse.play(Gaussian(160, amp, 40), self.d0_)

        self.cals.add_schedule(dxp)
        self.cals.add_parameter_value(0.5, "amp", 3, "xp")

        sched = block_to_schedule(self.cals.get_schedule("dxp", 3))

        self.assertEqual(sched.instructions[0][1], Play(Gaussian(160, 0.5, 40), DriveChannel(3)))
        self.assertEqual(sched.instructions[1][1], Play(Gaussian(160, 0.5, 40), DriveChannel(3)))

        with self.assertRaises(CalibrationError):
            self.cals.get_template("not registered")

        self.cals.get_template("xp", (3,))

    def test_register_schedule(self):
        """Test that we cannot register a schedule in a call."""

        xp = pulse.Schedule(name="xp")
        xp.insert(0, pulse.Play(pulse.Gaussian(160, 0.5, 40), pulse.DriveChannel(0)), inplace=True)

        with pulse.build(name="call_xp") as call_xp:
            pulse.call(xp)

        try:
            self.cals.add_schedule(call_xp)
        except CalibrationError as error:
            self.assertEqual(
                error.message, "Calling a Schedule is forbidden, call ScheduleBlock instead."
            )


class CrossResonanceTest(QiskitTestCase):
    """Setup class for an echoed cross-resonance calibration."""

    def setUp(self):
        """Create the setting to test."""
        super().setUp()

        controls = {
            (3, 2): [ControlChannel(10), ControlChannel(123)],
            (2, 3): [ControlChannel(15), ControlChannel(23)],
        }
        self.cals = Calibrations(control_config=controls)

        self.amp_cr = Parameter("amp")
        self.amp_rot = Parameter("amp_rot")
        self.amp = Parameter("amp")
        self.amp_tcp = Parameter("amp")
        self.d0_ = DriveChannel(Parameter("ch0"))
        self.d1_ = DriveChannel(Parameter("ch1"))
        self.c1_ = ControlChannel(Parameter("ch0.1"))
        self.sigma = Parameter("σ")
        self.width = Parameter("w")
        self.date_time = datetime.strptime("15/09/19 10:21:35", "%d/%m/%y %H:%M:%S")

        cr_tone_p = GaussianSquare(640, self.amp_cr, self.sigma, self.width)
        rotary_p = GaussianSquare(640, self.amp_rot, self.sigma, self.width)

        cr_tone_m = GaussianSquare(640, -self.amp_cr, self.sigma, self.width)
        rotary_m = GaussianSquare(640, -self.amp_rot, self.sigma, self.width)

        with pulse.build(name="xp") as xp:
            pulse.play(Gaussian(160, self.amp, self.sigma), self.d0_)

        with pulse.build(name="cr") as cr:
            with pulse.align_sequential():
                with pulse.align_left():
                    pulse.play(rotary_p, self.d1_)  # Rotary tone
                    pulse.play(cr_tone_p, self.c1_)  # CR tone.
                pulse.call(xp)
                with pulse.align_left():
                    pulse.play(rotary_m, self.d1_)
                    pulse.play(cr_tone_m, self.c1_)
                pulse.call(xp)

        # Mimic a tunable coupler pulse that is just a pulse on a control channel.
        with pulse.build(name="tcp") as tcp:
            pulse.play(GaussianSquare(640, self.amp_tcp, self.sigma, self.width), self.c1_)

        self.cals.add_schedule(xp)
        self.cals.add_schedule(cr)
        self.cals.add_schedule(tcp)

        self.cals.add_parameter_value(ParameterValue(40, self.date_time), "σ", schedule="xp")
        self.cals.add_parameter_value(
            ParameterValue(0.1 + 0.01j, self.date_time), "amp", (3,), "xp"
        )
        self.cals.add_parameter_value(ParameterValue(0.3, self.date_time), "amp", (3, 2), "cr")
        self.cals.add_parameter_value(ParameterValue(0.2, self.date_time), "amp_rot", (3, 2), "cr")
        self.cals.add_parameter_value(ParameterValue(0.8, self.date_time), "amp", (3, 2), "tcp")
        self.cals.add_parameter_value(ParameterValue(20, self.date_time), "w", (3, 2), "cr")

        # Reverse gate parameters
        self.cals.add_parameter_value(ParameterValue(0.15, self.date_time), "amp", (2,), "xp")
        self.cals.add_parameter_value(ParameterValue(0.5, self.date_time), "amp", (2, 3), "cr")
        self.cals.add_parameter_value(ParameterValue(0.4, self.date_time), "amp_rot", (2, 3), "cr")
        self.cals.add_parameter_value(ParameterValue(30, self.date_time), "w", (2, 3), "cr")


class TestControlChannels(CrossResonanceTest):
    """
    Test the echoed cross-resonance schedule which is more complex than single-qubit
    schedules. The example also shows that a schedule with call instructions can
    support parameters with the same names.
    """

    def test_get_schedule(self):
        """Check that we can get a CR schedule with a built in Call."""

        with pulse.build(name="cr") as cr_32:
            with pulse.align_sequential():
                with pulse.align_left():
                    pulse.play(GaussianSquare(640, 0.2, 40, 20), DriveChannel(2))  # Rotary tone
                    pulse.play(GaussianSquare(640, 0.3, 40, 20), ControlChannel(10))  # CR tone.
                pulse.play(Gaussian(160, 0.1 + 0.01j, 40), DriveChannel(3))
                with pulse.align_left():
                    pulse.play(GaussianSquare(640, -0.2, 40, 20), DriveChannel(2))  # Rotary tone
                    pulse.play(GaussianSquare(640, -0.3, 40, 20), ControlChannel(10))  # CR tone.
                pulse.play(Gaussian(160, 0.1 + 0.01j, 40), DriveChannel(3))

        # We inline to make the schedules comparable with the construction directly above.
        schedule = self.cals.get_schedule("cr", (3, 2))
        inline_schedule = inline_subroutines(schedule)
        for idx, inst in enumerate(inline_schedule.instructions):
            self.assertTrue(inst == cr_32.instructions[idx])

        self.assertEqual(schedule.parameters, set())

        # Do the CR in the other direction
        with pulse.build(name="cr") as cr_23:
            with pulse.align_sequential():
                with pulse.align_left():
                    pulse.play(GaussianSquare(640, 0.4, 40, 30), DriveChannel(3))  # Rotary tone
                    pulse.play(GaussianSquare(640, 0.5, 40, 30), ControlChannel(15))  # CR tone.
                pulse.play(Gaussian(160, 0.15, 40), DriveChannel(2))
                with pulse.align_left():
                    pulse.play(GaussianSquare(640, -0.4, 40, 30), DriveChannel(3))  # Rotary tone
                    pulse.play(GaussianSquare(640, -0.5, 40, 30), ControlChannel(15))  # CR tone.
                pulse.play(Gaussian(160, 0.15, 40), DriveChannel(2))

        schedule = self.cals.get_schedule("cr", (2, 3))
        inline_schedule = inline_subroutines(schedule)
        for idx, inst in enumerate(inline_schedule.instructions):
            self.assertTrue(inst == cr_23.instructions[idx])

        self.assertEqual(schedule.parameters, set())

    def test_free_parameters(self):
        """Test that we can get a schedule with free parameters."""

        schedule = self.cals.get_schedule("cr", (3, 2), assign_params={"amp": self.amp_cr})

        self.assertEqual(schedule.parameters, {self.amp_cr})

    def test_single_control_channel(self):
        """Test that getting a correct pulse on a control channel only works."""

        with pulse.build(name="tcp") as expected:
            pulse.play(GaussianSquare(640, 0.8, 40, 20), ControlChannel(10))

        self.assertEqual(self.cals.get_schedule("tcp", (3, 2)), expected)


class TestAssignment(QiskitTestCase):
    """Test simple assignment"""

    def setUp(self):
        """Create the setting to test."""
        super().setUp()

        controls = {(3, 2): [ControlChannel(10)]}

        self.cals = Calibrations(control_config=controls)

        self.amp_xp = Parameter("amp")
        self.ch0 = Parameter("ch0")
        self.d0_ = DriveChannel(self.ch0)
        self.ch1 = Parameter("ch1")
        self.d1_ = DriveChannel(self.ch1)
        self.sigma = Parameter("σ")
        self.width = Parameter("w")
        self.dur = Parameter("duration")

        with pulse.build(name="xp") as xp:
            pulse.play(Gaussian(160, self.amp_xp, self.sigma), self.d0_)

        with pulse.build(name="xpxp") as xpxp:
            with pulse.align_left():
                pulse.call(xp)
                pulse.call(xp, value_dict={self.ch0: self.ch1})

        self.xp_ = xp
        self.cals.add_schedule(xp)
        self.cals.add_schedule(xpxp)

        self.cals.add_parameter_value(0.2, "amp", (2,), "xp")
        self.cals.add_parameter_value(0.3, "amp", (3,), "xp")
        self.cals.add_parameter_value(40, "σ", (), "xp")

    def test_short_key(self):
        """Test simple value assignment"""
        sched = self.cals.get_schedule("xp", (2,), assign_params={"amp": 0.1})

        with pulse.build(name="xp") as expected:
            pulse.play(Gaussian(160, 0.1, 40), DriveChannel(2))

        self.assertEqual(sched, expected)

    def test_assign_to_parameter(self):
        """Test assigning to a Parameter instance"""
        my_amp = Parameter("my_amp")
        sched = self.cals.get_schedule("xp", (2,), assign_params={"amp": my_amp})

        with pulse.build(name="xp") as expected:
            pulse.play(Gaussian(160, my_amp, 40), DriveChannel(2))

        self.assertEqual(sched, expected)

    def test_assign_to_parameter_in_call(self):
        """Test assigning to a Parameter instance in a call"""
        with pulse.build(name="call_xp") as call_xp:
            pulse.call(self.xp_)
        self.cals.add_schedule(call_xp)

        my_amp = Parameter("my_amp")
        sched = self.cals.get_schedule("call_xp", (2,), assign_params={("amp", (2,), "xp"): my_amp})
        sched = block_to_schedule(sched)

        with pulse.build(name="xp") as expected:
            pulse.play(Gaussian(160, my_amp, 40), DriveChannel(2))
        expected = block_to_schedule(expected)

        self.assertEqual(sched, expected)

    def test_assign_to_parameter_in_call_and_to_value_in_caller(self):
        """Test assigning to a Parameter instances in a call and caller"""
        with pulse.build(name="call_xp_xp") as call_xp_xp:
            pulse.call(self.xp_)
            pulse.play(Gaussian(160, self.amp_xp, self.sigma), self.d0_)
        self.cals.add_schedule(call_xp_xp)

        my_amp = Parameter("amp")
        sched = self.cals.get_schedule(
            "call_xp_xp",
            (2,),
            assign_params={
                ("amp", (2,), "xp"): my_amp,
                ("amp", (2,), "call_xp_xp"): 0.2,
            },
        )
        sched = block_to_schedule(sched)

        with pulse.build(name="xp") as expected:
            pulse.play(Gaussian(160, my_amp, 40), DriveChannel(2))
            pulse.play(Gaussian(160, 0.2, 40), DriveChannel(2))
        expected = block_to_schedule(expected)

        self.assertEqual(sched, expected)

    def test_assign_to_same_parameter_in_call_and_caller(self):
        """
        Test assigning to a Parameter in a call and reassigning in caller raises

        Check that it is not allowed to leave a parameter in a subschedule free
        by assigning it to a Parameter that is also used in the calling
        schedule as that will re-bind the Parameter in the subschedule as well.
        """
        with pulse.build(name="call_xp_xp") as call_xp_xp:
            pulse.call(self.xp_)
            pulse.play(Gaussian(160, self.amp_xp, self.sigma), self.d0_)
        self.cals.add_schedule(call_xp_xp)

        my_amp = Parameter("amp")
        with self.assertRaises(CalibrationError):
            self.cals.get_schedule(
                "call_xp_xp",
                (2,),
                assign_params={
                    ("amp", (2,), "xp"): self.amp_xp,
                    ("amp", (2,), "call_xp_xp"): my_amp,
                },
            )

    def test_full_key(self):
        """Test value assignment with full key"""
        sched = self.cals.get_schedule("xp", (2,), assign_params={("amp", (2,), "xp"): 0.1})

        with pulse.build(name="xp") as expected:
            pulse.play(Gaussian(160, 0.1, 40), DriveChannel(2))

        self.assertEqual(sched, expected)

    def test_default_qubit(self):
        """Test value assignment with default qubit"""
        sched = self.cals.get_schedule("xp", (2,), assign_params={("amp", (), "xp"): 0.1})

        with pulse.build(name="xp") as expected:
            pulse.play(Gaussian(160, 0.1, 40), DriveChannel(2))

        self.assertEqual(sched, expected)

    def test_default_across_qubits(self):
        """Test assigning to multiple schedules through default parameter"""
        sched = self.cals.get_schedule("xpxp", (2, 3), assign_params={("amp", (), "xp"): 0.4})
        sched = block_to_schedule(sched)

        with pulse.build(name="xpxp") as expected:
            with pulse.align_left():
                pulse.play(Gaussian(160, 0.4, 40), DriveChannel(2))
                pulse.play(Gaussian(160, 0.4, 40), DriveChannel(3))

        expected = block_to_schedule(expected)

        self.assertEqual(sched, expected)


class TestReplaceScheduleAndCall(QiskitTestCase):
    """A test to ensure that inconsistencies are picked up when a schedule is reassigned."""

    def setUp(self):
        """Create the setting to test."""
        super().setUp()

        self.cals = Calibrations()

        self.amp = Parameter("amp")
        self.dur = Parameter("duration")
        self.sigma = Parameter("σ")
        self.beta = Parameter("β")
        self.ch0 = Parameter("ch0")

        with pulse.build(name="xp") as xp:
            pulse.play(Gaussian(self.dur, self.amp, self.sigma), DriveChannel(self.ch0))

        with pulse.build(name="call_xp") as call_xp:
            pulse.call(xp)

        self.cals.add_schedule(xp)
        self.cals.add_schedule(call_xp)

        self.cals.add_parameter_value(0.2, "amp", (4,), "xp")
        self.cals.add_parameter_value(160, "duration", (4,), "xp")
        self.cals.add_parameter_value(40, "σ", (), "xp")

    def test_call_replaced(self):
        """Test that we get an error when there is an inconsistency in subroutines."""

        sched = self.cals.get_schedule("call_xp", (4,))
        sched = block_to_schedule(sched)

        with pulse.build(name="xp") as expected:
            pulse.play(Gaussian(160, 0.2, 40), DriveChannel(4))

        expected = block_to_schedule(expected)

        self.assertEqual(sched, expected)

        # Now update the xp pulse without updating the call_xp schedule and ensure that
        # an error is raised.
        with pulse.build(name="xp") as drag:
            pulse.play(Drag(self.dur, self.amp, self.sigma, self.beta), DriveChannel(self.ch0))

        self.cals.add_schedule(drag)
        self.cals.add_parameter_value(10.0, "β", (4,), "xp")

        with self.assertRaises(CalibrationError):
            self.cals.get_schedule("call_xp", (4,))


class TestCoupledAssigning(QiskitTestCase):
    """Test that assigning parameters works when they are coupled in calls."""

    def setUp(self):
        """Create the setting to test."""
        super().setUp()

        controls = {(3, 2): [ControlChannel(10)]}

        self.cals = Calibrations(control_config=controls)

        self.amp_cr = Parameter("amp")
        self.amp_xp = Parameter("amp")
        self.ch0 = Parameter("ch0")
        self.d0_ = DriveChannel(self.ch0)
        self.ch1 = Parameter("ch1")
        self.d1_ = DriveChannel(self.ch1)
        self.c1_ = ControlChannel(Parameter("ch0.1"))
        self.sigma = Parameter("σ")
        self.width = Parameter("w")
        self.dur = Parameter("duration")

        with pulse.build(name="cr_p") as cr_p:
            pulse.play(GaussianSquare(self.dur, self.amp_cr, self.sigma, self.width), self.c1_)

        with pulse.build(name="cr_m") as cr_m:
            pulse.play(GaussianSquare(self.dur, -self.amp_cr, self.sigma, self.width), self.c1_)

        with pulse.build(name="xp") as xp:
            pulse.play(Gaussian(160, self.amp_xp, self.sigma), self.d0_)

        with pulse.build(name="ecr") as ecr:
            with pulse.align_sequential():
                pulse.call(cr_p)
                pulse.call(xp)
                pulse.call(cr_m)

        with pulse.build(name="cr_echo_both") as cr_echo_both:
            with pulse.align_sequential():
                pulse.call(cr_p)
                with pulse.align_left():
                    pulse.call(xp)
                    pulse.call(xp, value_dict={self.ch0: self.ch1})
                pulse.call(cr_m)

        self.cals.add_schedule(cr_p)
        self.cals.add_schedule(cr_m)
        self.cals.add_schedule(xp)
        self.cals.add_schedule(ecr)
        self.cals.add_schedule(cr_echo_both)

        self.cals.add_parameter_value(0.3, "amp", (3, 2), "cr_p")
        self.cals.add_parameter_value(0.2, "amp", (3,), "xp")
        self.cals.add_parameter_value(0.4, "amp", (2,), "xp")
        self.cals.add_parameter_value(40, "σ", (), "xp")
        self.cals.add_parameter_value(640, "w", (3, 2), "cr_p")
        self.cals.add_parameter_value(800, "duration", (3, 2), "cr_p")

    def test_assign_coupled_explicitly(self):
        """Test that we get the proper schedules when they are coupled."""

        # Test that we can preserve the coupling
        my_amp = Parameter("my_amp")
        assign_params = {("amp", (3, 2), "cr_p"): my_amp, ("amp", (3, 2), "cr_m"): my_amp}
        sched = self.cals.get_schedule("ecr", (3, 2), assign_params=assign_params)
        sched = block_to_schedule(sched)

        with pulse.build(name="ecr") as expected:
            with pulse.align_sequential():
                pulse.play(GaussianSquare(800, my_amp, 40, 640), ControlChannel(10))
                pulse.play(Gaussian(160, 0.2, 40), DriveChannel(3))
                pulse.play(GaussianSquare(800, -my_amp, 40, 640), ControlChannel(10))

        expected = block_to_schedule(expected)

        self.assertEqual(sched, expected)

    def test_assign_coupled_implicitly_float(self):
        """Test that we get the proper schedules when they are coupled."""
        assign_params = {("amp", (3, 2), "cr_m"): 0.8}
        sched = self.cals.get_schedule("ecr", (3, 2), assign_params=assign_params)
        sched = block_to_schedule(sched)

        with pulse.build(name="ecr") as expected:
            with pulse.align_sequential():
                pulse.play(GaussianSquare(800, 0.8, 40, 640), ControlChannel(10))
                pulse.play(Gaussian(160, 0.2, 40), DriveChannel(3))
                pulse.play(GaussianSquare(800, -0.8, 40, 640), ControlChannel(10))

        expected = block_to_schedule(expected)

        self.assertEqual(sched, expected)

    def test_assign_coupled_implicitly(self):
        """Test that we get the proper schedules when they are coupled."""
        my_amp = Parameter("my_amp")
        assign_params = {("amp", (3, 2), "cr_p"): my_amp}
        sched = self.cals.get_schedule("ecr", (3, 2), assign_params=assign_params)
        sched = block_to_schedule(sched)

        with pulse.build(name="ecr") as expected:
            with pulse.align_sequential():
                pulse.play(GaussianSquare(800, my_amp, 40, 640), ControlChannel(10))
                pulse.play(Gaussian(160, 0.2, 40), DriveChannel(3))
                pulse.play(GaussianSquare(800, -my_amp, 40, 640), ControlChannel(10))

        expected = block_to_schedule(expected)

        self.assertEqual(sched, expected)

    def test_break_coupled(self):
        """Test that we get the proper schedules when they are coupled."""
        my_amp = Parameter("my_amp")
        my_amp2 = Parameter("my_amp2")
        assign_params = {("amp", (3, 2), "cr_p"): my_amp, ("amp", (3, 2), "cr_m"): my_amp2}
        sched = self.cals.get_schedule("ecr", (3, 2), assign_params=assign_params)
        sched = block_to_schedule(sched)

        with pulse.build(name="ecr") as expected:
            with pulse.align_sequential():
                pulse.play(GaussianSquare(800, my_amp, 40, 640), ControlChannel(10))
                pulse.play(Gaussian(160, 0.2, 40), DriveChannel(3))
                pulse.play(GaussianSquare(800, -my_amp2, 40, 640), ControlChannel(10))

        expected = block_to_schedule(expected)

        self.assertEqual(sched, expected)

    def test_assign_coupled_explicitly_two_channel(self):
        """Test that we get the proper schedules when they are coupled."""

        # Test that we can preserve the coupling
        my_amp = Parameter("my_amp")
        my_amp2 = Parameter("my_amp2")
        assign_params = {("amp", (3,), "xp"): my_amp, ("amp", (2,), "xp"): my_amp2}
        sched = self.cals.get_schedule("cr_echo_both", (3, 2), assign_params=assign_params)
        sched = block_to_schedule(sched)

        with pulse.build(name="cr_echo_both") as expected:
            with pulse.align_sequential():
                pulse.play(GaussianSquare(800, 0.3, 40, 640), ControlChannel(10))
                with pulse.align_left():
                    pulse.play(Gaussian(160, my_amp, 40), DriveChannel(3))
                    pulse.play(Gaussian(160, my_amp2, 40), DriveChannel(2))
                pulse.play(GaussianSquare(800, -0.3, 40, 640), ControlChannel(10))

        expected = block_to_schedule(expected)

        self.assertEqual(sched, expected)


class TestFiltering(QiskitTestCase):
    """Test that the filtering works as expected."""

    def setUp(self):
        """Setup a calibration."""
        super().setUp()

        self.cals = Calibrations()

        self.sigma = Parameter("σ")
        self.amp = Parameter("amp")
        self.drive = DriveChannel(Parameter("ch0"))

        # Define and add template schedules.
        with pulse.build(name="xp") as xp:
            pulse.play(Gaussian(160, self.amp, self.sigma), self.drive)

        self.cals.add_schedule(xp)

        self.date_time1 = datetime.strptime("15/09/19 10:21:35", "%d/%m/%y %H:%M:%S")
        self.date_time2 = datetime.strptime("15/09/19 11:21:35", "%d/%m/%y %H:%M:%S")

        self.cals.add_parameter_value(ParameterValue(40, self.date_time1), "σ", schedule="xp")
        self.cals.add_parameter_value(
            ParameterValue(45, self.date_time2, False), "σ", schedule="xp"
        )
        self.cals.add_parameter_value(ParameterValue(0.1, self.date_time1), "amp", (0,), "xp")
        self.cals.add_parameter_value(ParameterValue(0.2, self.date_time2), "amp", (0,), "xp")
        self.cals.add_parameter_value(
            ParameterValue(0.4, self.date_time2, group="super_cal"), "amp", (0,), "xp"
        )

    def test_get_parameter_value(self):
        """Test that getting parameter values funcions properly."""

        amp = self.cals.get_parameter_value(self.amp, (0,), "xp")
        self.assertEqual(amp, 0.2)

        amp = self.cals.get_parameter_value(self.amp, (0,), "xp", group="super_cal")
        self.assertEqual(amp, 0.4)

        cutoff_date = datetime.strptime("15/09/19 11:21:34", "%d/%m/%y %H:%M:%S")
        amp = self.cals.get_parameter_value(self.amp, (0,), "xp", cutoff_date=cutoff_date)
        self.assertEqual(amp, 0.1)

        sigma = self.cals.get_parameter_value(self.sigma, (0,), "xp")
        self.assertEqual(sigma, 40)

        sigma = self.cals.get_parameter_value(self.sigma, (0,), "xp", valid_only=False)
        self.assertEqual(sigma, 45)


class TestSavingAndLoading(CrossResonanceTest):
    """Test that calibrations can be saved and loaded to and from files."""

    def test_save_load_parameter_values(self):
        """Test that we can save and load parameter values."""

        self.cals.save("csv", overwrite=True)
        self.assertEqual(self.cals.get_parameter_value("amp", (3,), "xp"), 0.1 + 0.01j)

        self.cals._params = defaultdict(list)

        with self.assertRaises(CalibrationError):
            self.cals.get_parameter_value("amp", (3,), "xp")

        # Load the parameters, check value and type.
        self.cals.load_parameter_values("parameter_values.csv")

        val = self.cals.get_parameter_value("amp", (3,), "xp")
        self.assertEqual(val, 0.1 + 0.01j)
        self.assertTrue(isinstance(val, complex))

        val = self.cals.get_parameter_value("σ", (3,), "xp")
        self.assertEqual(val, 40)
        self.assertTrue(isinstance(val, int))

        val = self.cals.get_parameter_value("amp", (3, 2), "cr")
        self.assertEqual(val, 0.3)
        self.assertTrue(isinstance(val, float))

        # Check that we cannot rewrite files as they already exist.
        with self.assertRaises(CalibrationError):
            self.cals.save("csv")

        self.cals.save("csv", overwrite=True)

        # Clean-up
        os.remove("parameter_values.csv")
        os.remove("parameter_config.csv")
        os.remove("schedules.csv")
