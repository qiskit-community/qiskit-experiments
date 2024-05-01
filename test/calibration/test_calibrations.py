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

from test.base import QiskitExperimentsTestCase
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone, timedelta

from ddt import data, ddt, unpack

from qiskit.circuit import Parameter, Gate
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
from qiskit import QuantumCircuit, pulse, transpile
from qiskit.circuit.library import CXGate, XGate
from qiskit.pulse.transforms import inline_subroutines, block_to_schedule
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target
from qiskit_ibm_runtime.fake_provider import FakeArmonkV2, FakeBelemV2

from qiskit_experiments.framework import BackendData
from qiskit_experiments.calibration_management.calibrations import Calibrations, ParameterKey
from qiskit_experiments.calibration_management.parameter_value import ParameterValue
from qiskit_experiments.calibration_management.basis_gate_library import (
    FixedFrequencyTransmon,
)
from qiskit_experiments.exceptions import CalibrationError


class MinimalBackend(BackendV2):
    """Class for testing a backend with minimal data"""

    def __init__(self, num_qubits=1):
        super().__init__()
        self._target = Target(num_qubits=num_qubits)

    @property
    def max_circuits(self):
        """Maximum circuits to run at once"""
        return 100

    @classmethod
    def _default_options(cls):
        return Options()

    @property
    def target(self) -> Target:
        """Target instance for the backend"""
        return self._target

    def run(self, run_input, **options):
        """Empty method to satisfy abstract base class"""
        pass


@ddt
class TestCalibrationsBasic(QiskitExperimentsTestCase):
    """Class to test the management of schedules and parameters for calibrations."""

    def setUp(self):
        """Create the setting to test."""
        super().setUp()

        self.cals = Calibrations(coupling_map=[])

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
            self.cals.add_schedule(sched, num_qubits=1)

        self.xm_pulse = xm

        # Add some parameter values.
        self.date_time = datetime.strptime("15/09/19 10:21:35", "%d/%m/%y %H:%M:%S")

        self.cals.add_parameter_value(ParameterValue(40, self.date_time), "σ", schedule="xp")
        self.cals.add_parameter_value(ParameterValue(160, self.date_time), "dur", schedule="xp")
        self.cals.add_parameter_value(ParameterValue(0.2, self.date_time), "amp", 3, "xp")
        self.cals.add_parameter_value(ParameterValue(0.1, self.date_time), "amp", (3,), "x90p")
        self.cals.add_parameter_value(ParameterValue(0.08, self.date_time), "amp", (3,), "y90p")
        self.cals.add_parameter_value(ParameterValue(40, self.date_time), "β", (3,), "xp")

    def test_calibration_save_json(self):
        """Test that the calibration under test can be serialized through JSON."""
        filename = self.__class__.__name__

        try:
            self.cals.save(file_type="json", file_prefix=filename)
            loaded = self.cals.load(file_path=f"{filename}.json")
            self.assertEqual(self.cals, loaded)
        finally:
            if os.path.exists(f"{filename}.json"):
                os.remove(f"{filename}.json")

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

    def test_improper_setup(self):
        """Check that an error is raised when coupling map and control channel map do not match."""
        controls = {
            (3, 2): [ControlChannel(10), ControlChannel(123)],
            (2, 3): [ControlChannel(15), ControlChannel(23)],
        }
        coupling_map = [[0, 1], [1, 0]]

        with self.assertRaises(CalibrationError):
            Calibrations(coupling_map=coupling_map, control_channel_map=controls)

        with self.assertRaises(CalibrationError):
            Calibrations(coupling_map=[], control_channel_map=controls)

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

        for param in (self.amp_xp, self.sigma, self.beta, self.duration, self.chan):
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

        self.cals.add_schedule(sched, num_qubits=1)

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

        self.cals.add_schedule(sched_good, num_qubits=2)

        with self.assertRaises(CalibrationError):
            self.cals.add_schedule(sched_bad, num_qubits=2)

    def test_unique_parameter_names(self):
        """Test that we cannot insert schedules in which parameter names are duplicates."""
        with pulse.build() as sched:
            pulse.play(Drag(160, Parameter("a"), Parameter("a"), Parameter("a")), DriveChannel(0))

        with self.assertRaises(CalibrationError):
            self.cals.add_schedule(sched, num_qubits=1)

    def test_parameter_without_schedule(self):
        """Test that we can manage parameters that are not bound to a schedule."""
        self.cals._register_parameter(Parameter("a"), ())

    def test_free_parameters(self):
        """Test that we can get a schedule with a free parameter."""
        xp = self.cals.get_schedule("xp", 3, assign_params={"amp": self.amp_xp})
        self.assertEqual(set(xp.parameters), {self.amp_xp})

        xp = self.cals.get_schedule("xp", 3, assign_params={"amp": self.amp_xp, "σ": self.sigma})
        self.assertEqual(set(xp.parameters), {self.amp_xp, self.sigma})

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

    def test_from_backend(self):
        """Test that when generating calibrations from backend
        the data is passed correctly"""
        backend = FakeBelemV2()
        cals = Calibrations.from_backend(backend, libraries=[FixedFrequencyTransmon()])
        with self.assertWarns(DeprecationWarning):
            config_args = cals.config()["kwargs"]
        control_channel_map_size = len(config_args["control_channel_map"].chan_map)
        coupling_map_size = len(config_args["coupling_map"])
        self.assertEqual(control_channel_map_size, 8)
        self.assertEqual(coupling_map_size, 8)
        self.assertEqual(cals.get_parameter_value("drive_freq", 0), 5090167234.445013)

    @data(
        (0, None, False),  # Edge case. Perhaps does not need to be supported
        (1, None, False),  # Produces backend.target.qubit_properties is None
        (2, None, False),
        (1, "x", False),  # Produces backend.coupling_map is None
        (1, "x", True),
        (2, "x", True),
        (2, "cx", True),  # backend.control_channel raises NotImplementedError
    )
    @unpack
    def test_from_minimal_backend(self, num_qubits, gate_name, pass_properties):
        """Test that from_backend works for a backend with minimal data"""
        # We do not use Gate or dict test arguments directly because they do
        # not translate to printable test case names, so we translate here.
        properties = None
        if gate_name == "x":
            gate = XGate()
            if pass_properties:
                properties = {(i,): None for i in range(num_qubits)}
        elif gate_name == "cx":
            gate = CXGate()
            if pass_properties:
                properties = {(0, 1): None}
        else:
            gate = None

        backend = MinimalBackend(num_qubits=num_qubits)
        if gate is not None:
            backend.target.add_instruction(gate, properties=properties)
        Calibrations.from_backend(backend)

    def test_equality(self):
        """Test the equal method on calibrations."""
        backend = FakeBelemV2()
        library = FixedFrequencyTransmon(basis_gates=["sx", "x"])

        cals1 = Calibrations.from_backend(
            backend, libraries=[library], add_parameter_defaults=False
        )
        cals2 = Calibrations.from_backend(
            backend, libraries=[library], add_parameter_defaults=False
        )
        self.assertTrue(cals1 == cals2)

        date_time = datetime.now(timezone.utc).astimezone()
        param_val = ParameterValue(0.12345, date_time=date_time)
        cals1.add_parameter_value(param_val, "amp", 3, "x")

        # The two objects are different due to missing parameter value
        self.assertFalse(cals1 == cals2)

        # The two objects are different due to time stamps
        param_val2 = ParameterValue(0.12345, date_time=date_time - timedelta(seconds=1))
        cals2.add_parameter_value(param_val2, "amp", 3, "x")
        self.assertFalse(cals1 == cals2)

        # The two objects are different due to missing parameter value
        cals3 = Calibrations.from_backend(
            backend, libraries=[library], add_parameter_defaults=False
        )
        self.assertFalse(cals1 == cals3)

        # The two objects are identical due to time stamps
        cals2.add_parameter_value(param_val, "amp", 3, "x")
        self.assertFalse(cals1 == cals3)

        # The schedules contained in the cals are different.
        library2 = FixedFrequencyTransmon(basis_gates=["sx", "x", "y"])
        cals1 = Calibrations.from_backend(backend, libraries=[library])
        cals2 = Calibrations.from_backend(backend, libraries=[library2])
        self.assertFalse(cals1 == cals2)

        # Ensure that the equality is not sensitive to parameter adding order.
        cals1 = Calibrations.from_backend(
            backend, libraries=[library], add_parameter_defaults=False
        )
        cals2 = Calibrations.from_backend(
            backend, libraries=[library], add_parameter_defaults=False
        )
        param_val1 = ParameterValue(0.54321, date_time=date_time)
        param_val2 = ParameterValue(0.12345, date_time=date_time - timedelta(seconds=1))

        cals1.add_parameter_value(param_val2, "amp", 3, "x")
        cals1.add_parameter_value(param_val1, "amp", 3, "x")

        cals2.add_parameter_value(param_val1, "amp", 3, "x")
        cals2.add_parameter_value(param_val2, "amp", 3, "x")

        self.assertTrue(cals1 == cals2)


class TestOverrideDefaults(QiskitExperimentsTestCase):
    """
    Test that we can override defaults. For example, this means that all qubits may have a
    Gaussian as xp pulse but a specific qubit may have a Drag pulse which overrides the
    default Gaussian.
    """

    def setUp(self):
        """Create the setting to test."""
        super().setUp()

        self.cals = Calibrations(coupling_map=[])

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
        self.cals.add_schedule(xp, num_qubits=1)
        self.cals.add_schedule(xp_drag, (3,))

    def test_calibration_save_json(self):
        """Test that the calibration under test can be serialized through JSON."""
        filename = self.__class__.__name__

        try:
            self.cals.save(file_type="json", file_prefix=filename)
            loaded = self.cals.load(file_path=f"{filename}.json")
            self.assertEqual(self.cals, loaded)
        finally:
            if os.path.exists(f"{filename}.json"):
                os.remove(f"{filename}.json")

    def test_parameter_value_adding_and_filtering(self):
        """Test that adding parameter values behaves in the expected way."""

        # Ensure that no parameter values are present when none have been added.
        params = self.cals.parameters_table()["data"]
        self.assertEqual(params, [])

        # Add a default parameter common to all qubits.
        self.cals.add_parameter_value(ParameterValue(40, self.date_time), "σ", schedule="xp")
        self.assertEqual(len(self.cals.parameters_table()["data"]), 1)

        # Check that we can get a default parameter in the parameter table
        self.assertEqual(len(self.cals.parameters_table(parameters=["σ"])["data"]), 1)
        self.assertEqual(
            len(self.cals.parameters_table(parameters=["σ"], schedules=["xp"])["data"]), 1
        )
        self.assertEqual(
            len(self.cals.parameters_table(parameters=["σ"], schedules=["xm"])["data"]), 0
        )

        # Test behaviour of qubit-specific parameter and without ParameterValue.
        self.cals.add_parameter_value(0.25, "amp", (3,), "xp")
        self.cals.add_parameter_value(0.15, "amp", (0,), "xp")

        # Check the value for qubit 0
        params = self.cals.parameters_table(parameters=["amp"], qubit_list=[(0,)])["data"]
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0]["value"], 0.15)
        self.assertEqual(params[0]["qubits"], (0,))

        # Check the value for qubit 3
        params = self.cals.parameters_table(parameters=["amp"], qubit_list=[(3,)])["data"]
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0]["value"], 0.25)
        self.assertEqual(params[0]["qubits"], (3,))

    def test_complex_parameter_value_deprecation_warning(self):
        """Test that complex parameter values raise PendingDeprecationWarning"""
        with self.assertWarns(PendingDeprecationWarning):
            ParameterValue(40j, self.date_time)
        with self.assertWarns(PendingDeprecationWarning):
            self.cals.add_parameter_value(40j, "amp", schedule="xp")

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
        self.assertTrue(xp0.instructions[0][1].pulse.pulse_type == "Gaussian")
        self.assertEqual(xp0.instructions[0][1].channel, DriveChannel(0))
        self.assertEqual(xp0.instructions[0][1].pulse.amp, 0.15)
        self.assertEqual(xp0.instructions[0][1].pulse.sigma, 40)
        self.assertEqual(xp0.instructions[0][1].pulse.duration, 160)

        # Check that xp3 is Play(Drag(160, 0.25, 40, 10), 3)
        self.assertTrue(xp3.instructions[0][1].pulse.pulse_type == "Drag")
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
        expected = {
            self.amp_xp,
            self.amp,
            self.sigma,
            self.beta,
            self.duration,
        }
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

        self.assertTrue(sched_cal.instructions[0][1].pulse.pulse_type == "Drag")
        self.assertTrue(sched_cal.instructions[1][1].pulse.pulse_type == "Drag")
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
        amp_values = self.cals.parameters_table(parameters=["amp"], qubit_list=[(0,)])["data"]
        self.assertEqual(len(amp_values), 1)

        # Check that we have one value for sigma.
        sigma_values = self.cals.parameters_table(parameters=["σ"])["data"]
        self.assertEqual(len(sigma_values), 1)

        # Check that we have two values for amp.
        amp_values = self.cals.parameters_table(parameters=["amp"])["data"]
        self.assertEqual(len(amp_values), 2)

        amp_values = self.cals.parameters_table(parameters=["amp"], qubit_list=[(3,)])["data"]
        self.assertEqual(len(amp_values), 1)

        # Check to see if we get back the two qubits when explicitly specifying them.
        amp_values = self.cals.parameters_table(parameters=["amp"], qubit_list=[(3,), (0,)])["data"]
        self.assertEqual(len(amp_values), 2)


class TestConcurrentParameters(QiskitExperimentsTestCase):
    """Test a particular edge case with the time in the parameter values."""

    def test_concurrent_values(self):
        """
        Ensure that if the max time has multiple entries we take the most recent appended one.
        """

        cals = Calibrations(coupling_map=[])

        amp = Parameter("amp")
        ch0 = Parameter("ch0")
        with pulse.build(name="xp") as xp:
            pulse.play(Gaussian(160, amp, 40), DriveChannel(ch0))

        cals.add_schedule(xp, num_qubits=1)

        date_time = datetime.strptime("15/09/19 10:21:35", "%d/%m/%y %H:%M:%S")

        cals.add_parameter_value(ParameterValue(0.25, date_time), "amp", (3,), "xp")
        cals.add_parameter_value(ParameterValue(0.35, date_time), "amp", (3,), "xp")
        cals.add_parameter_value(ParameterValue(0.45, date_time), "amp", (3,), "xp")

        self.assertEqual(cals.get_parameter_value("amp", 3, "xp"), 0.45)


class TestMeasurements(QiskitExperimentsTestCase):
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
            pulse.reference(xp.name, "q0")
            pulse.reference(meas.name, "q0")

        with pulse.build(name="xt_meas") as xt_meas:
            with pulse.align_sequential():
                pulse.reference(xp.name, "q0")
                pulse.reference(meas.name, "q0")
            with pulse.align_sequential():
                pulse.reference(xp.name, "q1")
                pulse.reference(meas.name, "q1")

        self.cals = Calibrations(coupling_map=[])
        self.cals.add_schedule(meas, num_qubits=1)
        self.cals.add_schedule(xp, num_qubits=1)
        self.cals.add_schedule(xp_meas, num_qubits=1)
        self.cals.add_schedule(xt_meas, num_qubits=2)
        self.cals.add_schedule(meas_acq, num_qubits=1)

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

    def test_calibration_save_json(self):
        """Test that the calibration under test can be serialized through JSON."""
        filename = self.__class__.__name__

        try:
            self.cals.save(file_type="json", file_prefix=filename)
            loaded = self.cals.load(file_path=f"{filename}.json")
            self.assertEqual(self.cals, loaded)
        finally:
            if os.path.exists(f"{filename}.json"):
                os.remove(f"{filename}.json")

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

        self.assertEqual(set(sched.parameters), {amp1, amp2})

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


class TestInstructions(QiskitExperimentsTestCase):
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
            pulse.reference(xp.name, "q0")
            pulse.reference(xp12.name, "q0")

        self.cals = Calibrations(coupling_map=[])
        self.cals.add_schedule(xp, num_qubits=1)
        self.cals.add_schedule(xp12, num_qubits=1)
        self.cals.add_schedule(xp02, num_qubits=1)

        self.date_time = datetime.strptime("15/09/19 10:21:35", "%d/%m/%y %H:%M:%S")

        self.cals.add_parameter_value(ParameterValue(1.57, self.date_time), "φ", (3,), "xp12")
        self.cals.add_parameter_value(ParameterValue(200, self.date_time), "ν", (3,), "xp12")

    def test_calibration_save_json(self):
        """Test that the calibration under test can be serialized through JSON."""
        filename = self.__class__.__name__

        try:
            self.cals.save(file_type="json", file_prefix=filename)
            loaded = self.cals.load(file_path=f"{filename}.json")
            self.assertEqual(self.cals, loaded)
        finally:
            if os.path.exists(f"{filename}.json"):
                os.remove(f"{filename}.json")

    def test_call_registration(self):
        """Check that by registering the call we registered three schedules."""

        self.assertEqual(len(self.cals.schedules()), 3)

    def test_instructions(self):
        """Check that we get a properly assigned schedule."""

        sched = self.cals.get_schedule("xp02", (3,))

        self.assertEqual(set(sched.parameters), set())

        sched = inline_subroutines(sched)  # inline makes the check more transparent.

        self.assertTrue(isinstance(sched.instructions[0][1], pulse.Play))
        self.assertEqual(sched.instructions[1][1].phase, 1.57)
        self.assertEqual(sched.instructions[2][1].frequency, 200)


class TestRegistering(QiskitExperimentsTestCase):
    """Class to test registering of subroutines with calls."""

    def setUp(self):
        """Create the setting to test."""
        super().setUp()

        self.cals = Calibrations(coupling_map=[])
        self.d0_ = DriveChannel(Parameter("ch0"))

    def test_call_registering(self):
        """Test registering of schedules with call."""
        with pulse.build(name="xp") as xp:
            pulse.play(Gaussian(160, 0.5, 40), self.d0_)

        with pulse.build(name="call_xp") as call_xp:
            pulse.reference(xp.name, "q0")

        with self.assertRaises(CalibrationError):
            self.cals.add_schedule(call_xp, num_qubits=1)

        self.cals.add_schedule(xp, num_qubits=1)
        self.cals.add_schedule(call_xp, num_qubits=1)

        self.assertTrue(isinstance(self.cals.get_schedule("call_xp", 2), pulse.ScheduleBlock))

    def test_get_template(self):
        """Test that we can get a registered template and use it."""
        amp = Parameter("amp")

        with pulse.build(name="xp") as xp:
            pulse.play(Gaussian(160, amp, 40), self.d0_)

        self.cals.add_schedule(xp, num_qubits=1)

        registered_xp = self.cals.get_template("xp", (1,))

        self.assertEqual(registered_xp, xp)

        with pulse.build(name="dxp") as dxp:
            pulse.reference(registered_xp.name, "q0")
            pulse.play(Gaussian(160, amp, 40), self.d0_)

        self.cals.add_schedule(dxp, num_qubits=1)
        self.cals.add_parameter_value(0.5, "amp", 3, "xp")

        sched = block_to_schedule(self.cals.get_schedule("dxp", 3))

        self.assertEqual(sched.instructions[0][1], Play(Gaussian(160, 0.5, 40), DriveChannel(3)))
        self.assertEqual(sched.instructions[1][1], Play(Gaussian(160, 0.5, 40), DriveChannel(3)))

        with self.assertRaises(CalibrationError):
            self.cals.get_template("not registered", (1,))

        self.cals.get_template("xp", (3,))


class CrossResonanceTest(QiskitExperimentsTestCase):
    """Setup class for an echoed cross-resonance calibration."""

    def setUp(self):
        """Create the setting to test."""
        super().setUp()

        controls = {
            (3, 2): [ControlChannel(10), ControlChannel(123)],
            (2, 3): [ControlChannel(15), ControlChannel(23)],
        }
        coupling_map = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]]
        self.cals = Calibrations(coupling_map=coupling_map, control_channel_map=controls)

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
                pulse.reference("xp", "q0")
                with pulse.align_left():
                    pulse.play(rotary_m, self.d1_)
                    pulse.play(cr_tone_m, self.c1_)
                pulse.reference("xp", "q0")

        # Mimic a tunable coupler pulse that is just a pulse on a control channel.
        with pulse.build(name="tcp") as tcp:
            pulse.play(GaussianSquare(640, self.amp_tcp, self.sigma, self.width), self.c1_)

        self.cals.add_schedule(xp, num_qubits=1)
        self.cals.add_schedule(cr, num_qubits=2)
        self.cals.add_schedule(tcp, num_qubits=2)

        self.cals.add_parameter_value(ParameterValue(40, self.date_time), "σ", schedule="xp")
        self.cals.add_parameter_value(ParameterValue(0.1, self.date_time), "amp", (3,), "xp")
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

    def test_calibration_save_json(self):
        """Test that the calibration under test can be serialized through JSON."""
        filename = self.__class__.__name__

        try:
            self.cals.save(file_type="json", file_prefix=filename)
            loaded = self.cals.load(file_path=f"{filename}.json")
            self.assertEqual(self.cals, loaded)
        finally:
            if os.path.exists(f"{filename}.json"):
                os.remove(f"{filename}.json")

    def test_get_schedule(self):
        """Check that we can get a CR schedule with a built in Call."""

        with pulse.build(name="cr") as cr_32:
            with pulse.align_sequential():
                with pulse.align_left():
                    pulse.play(GaussianSquare(640, 0.2, 40, 20), DriveChannel(2))  # Rotary tone
                    pulse.play(GaussianSquare(640, 0.3, 40, 20), ControlChannel(10))  # CR tone.
                pulse.play(Gaussian(160, 0.1, 40), DriveChannel(3))
                with pulse.align_left():
                    pulse.play(GaussianSquare(640, -0.2, 40, 20), DriveChannel(2))  # Rotary tone
                    pulse.play(GaussianSquare(640, -0.3, 40, 20), ControlChannel(10))  # CR tone.
                pulse.play(Gaussian(160, 0.1, 40), DriveChannel(3))

        # We inline to make the schedules comparable with the construction directly above.
        schedule = self.cals.get_schedule("cr", (3, 2))
        inline_schedule = inline_subroutines(schedule)
        for idx, inst in enumerate(inline_schedule.instructions):
            self.assertTrue(inst == cr_32.instructions[idx])

        self.assertEqual(set(schedule.parameters), set())

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

        self.assertEqual(set(schedule.parameters), set())

    def test_free_parameters(self):
        """Test that we can get a schedule with free parameters."""

        assign_params = {("amp", (3, 2), "cr"): self.amp_cr}
        schedule = self.cals.get_schedule("cr", (3, 2), assign_params=assign_params)

        self.assertEqual(set(schedule.parameters), {self.amp_cr})

    def test_single_control_channel(self):
        """Test that getting a correct pulse on a control channel only works."""

        with pulse.build(name="tcp") as expected:
            pulse.play(GaussianSquare(640, 0.8, 40, 20), ControlChannel(10))

        self.assertEqual(self.cals.get_schedule("tcp", (3, 2)), expected)

    def test_inst_map_stays_consistent(self):
        """Check that get schedule and inst map are in sync in a complex ECR case.

        Test that when a parameter value is updated for a parameter that is used in a
        schedule nested inside a call instruction of an outer schedule that that outer
        schedule is also updated in the instruction schedule map. For example, this test
        will fail if the coupling_map and the control_channel_map are not consistent
        with each other. This is because the coupling_map is used to build the
        _operated_qubits variable which determines the qubits of the instruction to
        which a schedule is associated.
        """

        # Check that the ECR schedules from get_schedule and the instmap are the same
        sched_inst = self.cals.default_inst_map.get("cr", (2, 3))
        self.assertEqual(sched_inst, self.cals.get_schedule("cr", (2, 3)))

        # Ensure that amp is 0.15
        insts = block_to_schedule(sched_inst).filter(channels=[DriveChannel(2)]).instructions
        self.assertEqual(insts[0][1].pulse.amp, 0.15)

        # Update amp to 0.25 and check that change is propagated through.
        date_time2 = datetime.strptime("15/09/19 10:22:35", "%d/%m/%y %H:%M:%S")
        self.cals.add_parameter_value(ParameterValue(0.25, date_time2), "amp", (2,), schedule="xp")

        sched_inst = self.cals.default_inst_map.get("cr", (2, 3))
        self.assertEqual(sched_inst, self.cals.get_schedule("cr", (2, 3)))
        insts = block_to_schedule(sched_inst).filter(channels=[DriveChannel(2)]).instructions
        self.assertEqual(insts[0][1].pulse.amp, 0.25)

        # Test linked parameters.
        self.cals.add_parameter_value(ParameterValue(2, date_time2), "σ", (2,), schedule="xp")

        sched_inst = self.cals.default_inst_map.get("cr", (2, 3))
        self.assertEqual(sched_inst, self.cals.get_schedule("cr", (2, 3)))
        insts = block_to_schedule(sched_inst).filter(channels=[DriveChannel(2)]).instructions
        self.assertEqual(insts[0][1].pulse.sigma, 2)


class TestAssignment(QiskitExperimentsTestCase):
    """Test simple assignment"""

    def setUp(self):
        """Create the setting to test."""
        super().setUp()

        controls = {(3, 2): [ControlChannel(10)]}
        coupling_map = [[2, 3], [3, 2]]
        self.cals = Calibrations(coupling_map=coupling_map, control_channel_map=controls)

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
                pulse.reference("xp", "q0")
                pulse.reference("xp", "q1")

        self.xp_ = xp
        self.cals.add_schedule(xp, num_qubits=1)
        self.cals.add_schedule(xpxp, num_qubits=2)

        self.cals.add_parameter_value(0.2, "amp", (2,), "xp")
        self.cals.add_parameter_value(0.3, "amp", (3,), "xp")
        self.cals.add_parameter_value(40, "σ", (), "xp")

    def test_calibration_save_json(self):
        """Test that the calibration under test can be serialized through JSON."""
        filename = self.__class__.__name__

        try:
            self.cals.save(file_type="json", file_prefix=filename)
            loaded = self.cals.load(file_path=f"{filename}.json")
            self.assertEqual(self.cals, loaded)
        finally:
            if os.path.exists(f"{filename}.json"):
                os.remove(f"{filename}.json")

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

    def test_assign_to_parameter_in_reference(self):
        """Test assigning to a Parameter instance in a reference."""
        with pulse.build(name="call_xp") as call_xp:
            pulse.reference(self.xp_.name, "q0")
        self.cals.add_schedule(call_xp, num_qubits=1)

        my_amp = Parameter("my_amp")
        sched = self.cals.get_schedule("call_xp", (2,), assign_params={("amp", (2,), "xp"): my_amp})
        sched = block_to_schedule(sched)

        with pulse.build(name="xp") as expected:
            pulse.play(Gaussian(160, my_amp, 40), DriveChannel(2))
        expected = block_to_schedule(expected)

        self.assertEqual(sched, expected)

    def test_assign_to_parameter_in_reference_and_to_value_in_referencer(self):
        """Test assigning to a Parameter instances in a reference and referencer."""
        with pulse.build(name="call_xp_xp") as call_xp_xp:
            pulse.reference(self.xp_.name, "q0")
            pulse.play(Gaussian(160, self.amp_xp, self.sigma), self.d0_)
        self.cals.add_schedule(call_xp_xp, num_qubits=1)

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
            pulse.reference(self.xp_.name, "q0")
            pulse.play(Gaussian(160, self.amp_xp, self.sigma), self.d0_)
        self.cals.add_schedule(call_xp_xp, num_qubits=1)

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


class TestReplaceScheduleAndCall(QiskitExperimentsTestCase):
    """A test to ensure that inconsistencies are picked up when a schedule is reassigned."""

    def setUp(self):
        """Create the setting to test."""
        super().setUp()

        self.cals = Calibrations(coupling_map=[])

        self.amp = Parameter("amp")
        self.dur = Parameter("duration")
        self.sigma = Parameter("σ")
        self.beta = Parameter("β")
        self.ch0 = Parameter("ch0")

        with pulse.build(name="xp") as xp:
            pulse.play(Gaussian(self.dur, self.amp, self.sigma), DriveChannel(self.ch0))

        with pulse.build(name="call_xp") as call_xp:
            pulse.reference(xp.name, "q0")

        self.cals.add_schedule(xp, num_qubits=1)
        self.cals.add_schedule(call_xp, num_qubits=1)

        self.cals.add_parameter_value(0.2, "amp", (4,), "xp")
        self.cals.add_parameter_value(160, "duration", (4,), "xp")
        self.cals.add_parameter_value(40, "σ", (), "xp")

    def test_calibration_save_json(self):
        """Test that the calibration under test can be serialized through JSON."""
        filename = self.__class__.__name__

        try:
            self.cals.save(file_type="json", file_prefix=filename)
            loaded = self.cals.load(file_path=f"{filename}.json")
            self.assertEqual(self.cals, loaded)
        finally:
            if os.path.exists(f"{filename}.json"):
                os.remove(f"{filename}.json")

    def test_reference_replaced(self):
        """Test that we get an error when there is an inconsistency in subroutines."""

        sched = self.cals.get_schedule("call_xp", (4,))

        with pulse.build(name="xp") as expected:
            pulse.play(Gaussian(160, 0.2, 40), DriveChannel(4))

        self.assertEqual(block_to_schedule(sched), block_to_schedule(expected))

        # Now update the xp pulse without updating the call_xp schedule and ensure consistency.
        with pulse.build(name="xp") as drag:
            pulse.play(Drag(self.dur, self.amp, self.sigma, self.beta), DriveChannel(self.ch0))

        self.cals.add_schedule(drag, num_qubits=1)
        self.cals.add_parameter_value(10.0, "β", (4,), "xp")

        sched = self.cals.get_schedule("call_xp", (4,))

        with pulse.build(name="xp") as expected:
            pulse.play(Drag(160, 0.2, 40, 10.0), DriveChannel(4))

        self.assertEqual(block_to_schedule(sched), block_to_schedule(expected))


class TestCoupledAssigning(QiskitExperimentsTestCase):
    """Test that assigning parameters works when they are coupled in calls."""

    def setUp(self):
        """Create the setting to test."""
        super().setUp()

        controls = {(3, 2): [ControlChannel(10)]}
        coupling_map = [[2, 3], [3, 2]]
        self.cals = Calibrations(coupling_map=coupling_map, control_channel_map=controls)

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
                pulse.reference(cr_p.name, "q0", "q1")
                pulse.reference(xp.name, "q0")
                pulse.reference(cr_m.name, "q0", "q1")

        with pulse.build(name="cr_echo_both") as cr_echo_both:
            with pulse.align_sequential():
                pulse.reference(cr_p.name, "q0", "q1")
                with pulse.align_left():
                    pulse.reference(xp.name, "q0")
                    pulse.reference(xp.name, "q1")
                pulse.reference(cr_m.name, "q0", "q1")

        self.cals.add_schedule(cr_p, num_qubits=2)
        self.cals.add_schedule(cr_m, num_qubits=2)
        self.cals.add_schedule(xp, num_qubits=1)
        self.cals.add_schedule(ecr, num_qubits=2)
        self.cals.add_schedule(cr_echo_both, num_qubits=2)

        self.cals.add_parameter_value(0.3, "amp", (3, 2), "cr_p")
        self.cals.add_parameter_value(0.2, "amp", (3,), "xp")
        self.cals.add_parameter_value(0.4, "amp", (2,), "xp")
        self.cals.add_parameter_value(40, "σ", (), "xp")
        self.cals.add_parameter_value(640, "w", (3, 2), "cr_p")
        self.cals.add_parameter_value(800, "duration", (3, 2), "cr_p")

    def test_calibration_save_json(self):
        """Test that the calibration under test can be serialized through JSON."""
        filename = self.__class__.__name__

        try:
            self.cals.save(file_type="json", file_prefix=filename)
            loaded = self.cals.load(file_path=f"{filename}.json")
            self.assertEqual(self.cals, loaded)
        finally:
            if os.path.exists(f"{filename}.json"):
                os.remove(f"{filename}.json")

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


class TestFiltering(QiskitExperimentsTestCase):
    """Test that the filtering works as expected."""

    def setUp(self):
        """Setup a calibration."""
        super().setUp()

        self.cals = Calibrations(coupling_map=[])

        self.sigma = Parameter("σ")
        self.amp = Parameter("amp")
        self.drive = DriveChannel(Parameter("ch0"))

        # Define and add template schedules.
        with pulse.build(name="xp") as xp:
            pulse.play(Gaussian(160, self.amp, self.sigma), self.drive)

        self.cals.add_schedule(xp, num_qubits=1)

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

    def test_calibration_save_json(self):
        """Test that the calibration under test can be serialized through JSON."""
        filename = self.__class__.__name__

        try:
            self.cals.save(file_type="json", file_prefix=filename)
            loaded = self.cals.load(file_path=f"{filename}.json")
            self.assertEqual(self.cals, loaded)
        finally:
            if os.path.exists(f"{filename}.json"):
                os.remove(f"{filename}.json")

    def test_parameter_table_most_recent(self):
        """Test the most_recent argument to the parameter_table method."""

        table = self.cals.parameters_table(parameters=["amp"], most_recent_only=False)
        self.assertTrue(len(table["data"]), 2)

        table = self.cals.parameters_table(parameters=["amp"], most_recent_only=True)
        self.assertTrue(len(table["data"]), 1)
        self.assertTrue(table["data"][0]["value"], 0.2)

    def test_get_parameter_value(self):
        """Test that getting parameter values functions properly."""

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

    def setUp(self):
        """Setup the test."""
        self._prefix = str(uuid.uuid4())
        super().setUp()

    def tearDown(self):
        """Clean-up after the test."""
        super().tearDown()

        for file in ["parameter_values.csv", "parameter_config.csv", "schedules.csv", ".json"]:
            if os.path.exists(self._prefix + file):
                os.remove(self._prefix + file)

    def test_save_load_parameter_values_csv(self):
        """Test that we can save and load parameter values."""
        # NOTE: This is a legacy test that can be removed when csv support is
        # removed from Calibrations.save

        # Expect user warning about schedules, deprecation warning about csv
        with self.assertWarns((UserWarning, DeprecationWarning)):
            self.cals.save("csv", overwrite=True, file_prefix=self._prefix)
        self.assertEqual(self.cals.get_parameter_value("amp", (3,), "xp"), 0.1)

        self.cals._params = defaultdict(list)

        with self.assertRaises(CalibrationError):
            self.cals.get_parameter_value("amp", (3,), "xp")

        # Load the parameters, check value and type.
        with self.assertWarns(DeprecationWarning):
            self.cals.load_parameter_values(self._prefix + "parameter_values.csv")

        val = self.cals.get_parameter_value("amp", (3,), "xp")
        self.assertEqual(val, 0.1)

        val = self.cals.get_parameter_value("σ", (3,), "xp")
        self.assertEqual(val, 40)
        self.assertTrue(isinstance(val, int))

        val = self.cals.get_parameter_value("amp", (3, 2), "cr")
        self.assertEqual(val, 0.3)
        self.assertTrue(isinstance(val, float))

        # Check that we cannot rewrite files as they already exist.
        with self.assertRaises(CalibrationError):
            with self.assertWarns((UserWarning, DeprecationWarning)):
                self.cals.save("csv", file_prefix=self._prefix)

        with self.assertWarns((UserWarning, DeprecationWarning)):
            self.cals.save("csv", overwrite=True, file_prefix=self._prefix)

    def test_alternate_date_formats(self):
        """Test that we can reload dates with or without time-zone."""

        new_date = datetime.strptime("16/09/20 10:21:35.012+0200", "%d/%m/%y %H:%M:%S.%f%z")
        value = ParameterValue(0.222, date_time=new_date)
        self.cals.add_parameter_value(value, "amp", (3,), "xp")

        self.cals.save("json", overwrite=True, file_prefix=self._prefix)
        self.cals.load(self._prefix + ".json")

    def test_save_load_library_csv(self):
        """Test that we can load and save a library.

        These libraries contain both parameters with schedules and parameters without
        any schedules (e.g. frequencies for qubits and readout).
        """

        library = FixedFrequencyTransmon()
        backend = FakeArmonkV2()
        cals = Calibrations.from_backend(backend, libraries=[library])

        cals.parameters_table()

        with self.assertWarns((UserWarning, DeprecationWarning)):
            cals.save(file_type="csv", overwrite=True, file_prefix=self._prefix)

        with self.assertWarns(DeprecationWarning):
            cals.load_parameter_values(self._prefix + "parameter_values.csv")

        # Test the value of a few loaded params.
        self.assertEqual(cals.get_parameter_value("amp", (0,), "x"), 0.5)
        self.assertEqual(
            cals.get_parameter_value("drive_freq", (0,)),
            BackendData(backend).drive_freqs[0],
        )

    def test_save_load_library(self):
        """Test that we can load and save a library.

        These libraries contain both parameters with schedules and parameters without
        any schedules (e.g. frequencies for qubits and readout).
        """

        library = FixedFrequencyTransmon()
        backend = FakeArmonkV2()
        cals = Calibrations.from_backend(backend, libraries=[library])

        cals.parameters_table()

        cals.save(file_type="json", overwrite=True, file_prefix=self._prefix)

        loaded = Calibrations.load(self._prefix + ".json")

        # Test the value of a few loaded params.
        self.assertEqual(loaded.get_parameter_value("amp", (0,), "x"), 0.5)
        self.assertEqual(
            loaded.get_parameter_value("drive_freq", (0,)),
            BackendData(backend).drive_freqs[0],
        )

    def test_json_round_trip(self):
        """Test round trip test for JSON file format.

        This method guarantees full equality including parameterized template schedules
        and we can still generate schedules with loaded calibration instance,
        even though calibrations is instantiated outside built-in library.
        """
        self.cals.save(file_type="json", overwrite=True, file_prefix=self._prefix)
        loaded = self.cals.load(file_path=self._prefix + ".json")
        self.assertEqual(self.cals, loaded)

        original_sched = self.cals.get_schedule("cr", (3, 2))
        roundtrip_sched = loaded.get_schedule("cr", (3, 2))
        self.assertEqual(original_sched, roundtrip_sched)

    def test_overwrite(self):
        """Test that overwriting errors unless overwrite flag is used"""
        self.cals.save(file_type="json", overwrite=True, file_prefix=self._prefix)
        with self.assertRaises(CalibrationError):
            self.cals.save(file_type="json", overwrite=False, file_prefix=self._prefix)

        # Add a value to make sure data is really overwritten and not carried
        # over from first write
        self.cals.add_parameter_value(0.45, "amp", (3,), "xp")
        self.cals.save(file_type="json", overwrite=True, file_prefix=self._prefix)
        loaded = Calibrations.load(file_path=self._prefix + ".json")
        self.assertEqual(self.cals, loaded)


class TestInstructionScheduleMap(QiskitExperimentsTestCase):
    """Class to test the functionality of a Calibrations"""

    def test_setup_withLibrary(self):
        """Test that we can setup with a library."""

        cals = Calibrations.from_backend(
            FakeArmonkV2(),
            libraries=[
                FixedFrequencyTransmon(basis_gates=["x", "sx"], default_values={"duration": 320})
            ],
        )

        # Check the x gate
        with pulse.build(name="x") as expected:
            pulse.play(pulse.Drag(duration=320, amp=0.5, sigma=80, beta=0), pulse.DriveChannel(0))

        self.assertEqual(cals.get_schedule("x", (0,)), expected)

        # Check the sx gate
        with pulse.build(name="sx") as expected:
            pulse.play(pulse.Drag(duration=320, amp=0.25, sigma=80, beta=0), pulse.DriveChannel(0))

        self.assertEqual(cals.get_schedule("sx", (0,)), expected)

    def test_instruction_schedule_map_export(self):
        """Test that exporting the inst map works as planned."""

        backend = FakeBelemV2()

        cals = Calibrations.from_backend(
            backend,
            libraries=[FixedFrequencyTransmon(basis_gates=["sx"])],
        )

        u_chan = pulse.ControlChannel(Parameter("ch0.1"))
        with pulse.build(name="cr") as cr:
            pulse.play(pulse.GaussianSquare(640, 0.5, 64, 384), u_chan)

        cals.add_schedule(cr, num_qubits=2)
        cals.update_inst_map({"cr"})

        for qubit in range(BackendData(backend).num_qubits):
            self.assertTrue(cals.default_inst_map.has("sx", (qubit,)))

        # based on coupling map of Belem to keep the test robust.
        expected_pairs = [(0, 1), (1, 0), (1, 2), (2, 1), (1, 3), (3, 1), (3, 4), (4, 3)]
        coupling_map = set(tuple(pair) for pair in BackendData(backend).coupling_map)

        for pair in expected_pairs:
            self.assertTrue(pair in coupling_map)
            self.assertTrue(cals.default_inst_map.has("cr", pair), pair)

    def test_inst_map_transpilation(self):
        """Test that we can use the inst_map to inject the cals into the circuit."""

        cals = Calibrations.from_backend(
            FakeArmonkV2(),
            libraries=[FixedFrequencyTransmon(basis_gates=["x"])],
        )

        param = Parameter("amp")
        cals.inst_map_add("Rabi", (0,), "x", assign_params={"amp": param})

        circ = QuantumCircuit(1)
        circ.x(0)
        circ.append(Gate("Rabi", num_qubits=1, params=[param]), (0,))

        circs, amps = [], [0.12, 0.25]

        for amp in amps:
            new_circ = circ.assign_parameters({param: amp}, inplace=False)
            circs.append(new_circ)

        # Check that calibrations are absent
        for circ in circs:
            self.assertEqual(len(circ.calibrations), 0)

        # Transpile to inject the cals.
        circs = transpile(circs, inst_map=cals.default_inst_map)

        # Check that we have the expected schedules.
        with pulse.build() as x_expected:
            pulse.play(pulse.Drag(160, 0.5, 40, 0), pulse.DriveChannel(0))

        for idx, circ in enumerate(circs):
            amp = amps[idx]

            with pulse.build() as rabi_expected:
                pulse.play(pulse.Drag(160, amp, 40, 0), pulse.DriveChannel(0))

            self.assertEqual(circ.calibrations["x"][((0,), ())], x_expected)

            circ_rabi = next(iter(circ.calibrations["Rabi"].values()))
            self.assertEqual(circ_rabi, rabi_expected)

        # Test the removal of the Rabi instruction
        self.assertTrue(cals.default_inst_map.has("Rabi", (0,)))

        cals.default_inst_map.remove("Rabi", (0,))

        self.assertFalse(cals.default_inst_map.has("Rabi", (0,)))

    def test_inst_map_updates(self):
        """Test that updating a parameter will force an inst map update."""

        cals = Calibrations.from_backend(
            FakeBelemV2(),
            libraries=[FixedFrequencyTransmon(basis_gates=["sx", "x"])],
        )

        # Test the schedules before the update.
        for qubit in range(5):
            for gate, amp in [("x", 0.5), ("sx", 0.25)]:
                with pulse.build() as expected:
                    pulse.play(pulse.Drag(160, amp, 40, 0), pulse.DriveChannel(qubit))

                self.assertEqual(cals.default_inst_map.get(gate, qubit), expected)

        # Update the duration, this should impact all gates.
        cals.add_parameter_value(200, "duration", schedule="sx")

        # Test that all schedules now have an updated duration in the inst_map
        for qubit in range(5):
            for gate, amp in [("x", 0.5), ("sx", 0.25)]:
                with pulse.build() as expected:
                    pulse.play(pulse.Drag(200, amp, 40, 0), pulse.DriveChannel(qubit))

                self.assertEqual(cals.default_inst_map.get(gate, qubit), expected)

        # Update the amp on a single qubit, this should only update one gate in the inst_map
        cals.add_parameter_value(0.8, "amp", qubits=(4,), schedule="sx")

        # Test that all schedules now have an updated duration in the inst_map
        for qubit in range(5):
            for gate, amp in [("x", 0.5), ("sx", 0.25)]:

                if gate == "sx" and qubit == 4:
                    amp = 0.8

                with pulse.build() as expected:
                    pulse.play(pulse.Drag(200, amp, 40, 0), pulse.DriveChannel(qubit))

                self.assertEqual(cals.default_inst_map.get(gate, qubit), expected)

    def test_cx_cz_case(self):
        """Test the case where the coupling map has CX and CZ on different qubits.

        We use FakeBelem which has a linear coupling map and will restrict ourselves to
        qubits 0, 1, and 2. The Cals will define a template schedule for CX and CZ. We will
        mock this with GaussianSquare and Gaussian pulses since the nature of the schedules
        is irrelevant here. The parameters for CX will only have values for qubits 0 and 1 while
        the parameters for CZ will only have values for qubits 1 and 2. We therefore will have
        a CX on qubits 0, 1 in the inst. map and a CZ on qubits 1, 2.
        """

        cals = Calibrations.from_backend(FakeBelemV2())

        sig = Parameter("σ")
        dur = Parameter("duration")
        width = Parameter("width")
        amp_cx = Parameter("amp")
        amp_cz = Parameter("amp")
        uchan = Parameter("ch1.0")

        with pulse.build(name="cx") as cx:
            pulse.play(
                pulse.GaussianSquare(duration=dur, amp=amp_cx, sigma=sig, width=width),
                pulse.ControlChannel(uchan),
            )

        with pulse.build(name="cz") as cz:
            pulse.play(
                pulse.Gaussian(duration=dur, amp=amp_cz, sigma=sig), pulse.ControlChannel(uchan)
            )

        cals.add_schedule(cx, num_qubits=2)
        cals.add_schedule(cz, num_qubits=2)

        cals.add_parameter_value(640, "duration", schedule="cx")
        cals.add_parameter_value(64, "σ", schedule="cx")
        cals.add_parameter_value(320, "width", qubits=(0, 1), schedule="cx")
        cals.add_parameter_value(320, "width", qubits=(1, 0), schedule="cx")
        cals.add_parameter_value(0.1, "amp", qubits=(0, 1), schedule="cx")
        cals.add_parameter_value(0.8, "amp", qubits=(1, 0), schedule="cx")
        cals.add_parameter_value(0.1, "amp", qubits=(2, 1), schedule="cz")
        cals.add_parameter_value(0.8, "amp", qubits=(1, 2), schedule="cz")

        # CX only defined for qubits (0, 1) and (1,0)?
        self.assertTrue(cals.default_inst_map.has("cx", (0, 1)))
        self.assertTrue(cals.default_inst_map.has("cx", (1, 0)))
        self.assertFalse(cals.default_inst_map.has("cx", (2, 1)))
        self.assertFalse(cals.default_inst_map.has("cx", (1, 2)))

        # CZ only defined for qubits (2, 1) and (1,2)?
        self.assertTrue(cals.default_inst_map.has("cz", (2, 1)))
        self.assertTrue(cals.default_inst_map.has("cz", (1, 2)))
        self.assertFalse(cals.default_inst_map.has("cz", (0, 1)))
        self.assertFalse(cals.default_inst_map.has("cz", (1, 0)))

    def test_alternate_initialization(self):
        """Test that we can initialize without a backend object."""

        backend = FakeBelemV2()
        library = FixedFrequencyTransmon(basis_gates=["sx", "x"])

        backend_data = BackendData(backend)
        control_channel_map = {}
        for qargs in backend_data.coupling_map:
            control_channel_map[tuple(qargs)] = backend_data.control_channel(qargs)

        cals1 = Calibrations.from_backend(backend, libraries=[library])
        cals2 = Calibrations(
            libraries=[library],
            control_channel_map=control_channel_map,
            coupling_map=backend_data.coupling_map,
        )

        self.assertEqual(str(cals1.get_schedule("x", 1)), str(cals2.get_schedule("x", 1)))


class TestSerialization(QiskitExperimentsTestCase):
    """Test the serialization of the Calibrations."""

    def test_serialization(self):
        """Test the serialization."""

        backend = FakeBelemV2()
        library = FixedFrequencyTransmon(basis_gates=["sx", "x"])

        cals = Calibrations.from_backend(backend, libraries=[library])
        cals.add_parameter_value(0.12345, "amp", 3, "x")

        self.assertRoundTripSerializable(cals)
