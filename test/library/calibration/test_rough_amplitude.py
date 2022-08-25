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

"""Test rough amplitude calibration experiment classes."""

from test.base import QiskitExperimentsTestCase
import numpy as np

from qiskit import transpile
import qiskit.pulse as pulse
from qiskit.circuit import Parameter
from qiskit.providers.fake_provider import FakeArmonkV2

from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.library import EFRoughXSXAmplitudeCal, RoughXSXAmplitudeCal
from qiskit_experiments.test.mock_iq_backend import MockIQBackend
from qiskit_experiments.test.mock_iq_helpers import MockIQRabiHelper as RabiHelper


class TestRoughAmpCal(QiskitExperimentsTestCase):
    """A class to test the rough amplitude calibration experiments."""

    def setUp(self):
        """Setup the tests."""
        super().setUp()
        library = FixedFrequencyTransmon()

        self.backend = FakeArmonkV2()
        self.cals = Calibrations.from_backend(self.backend, libraries=[library])

    def test_circuits(self):
        """Test the quantum circuits."""
        test_amps = [-0.5, 0, 0.5]
        rabi = RoughXSXAmplitudeCal(0, self.cals, amplitudes=test_amps)

        circs = transpile(rabi.circuits(), self.backend, inst_map=self.cals.default_inst_map)

        for circ, amp in zip(circs, test_amps):
            self.assertEqual(circ.count_ops()["Rabi"], 1)

            d0 = pulse.DriveChannel(0)
            with pulse.build(name="x") as expected_x:
                pulse.play(pulse.Drag(160, amp, 40, 0), d0)

            self.assertEqual(circ.calibrations["Rabi"][((0,), (amp,))], expected_x)

    def test_update(self):
        """Test that the calibrations update properly."""

        self.assertTrue(np.allclose(self.cals.get_parameter_value("amp", 0, "x"), 0.5))
        self.assertTrue(np.allclose(self.cals.get_parameter_value("amp", 0, "sx"), 0.25))

        rabi_ef = RoughXSXAmplitudeCal(0, self.cals)
        expdata = rabi_ef.run(MockIQBackend(RabiHelper(amplitude_to_angle=np.pi * 1.5)))
        self.assertExperimentDone(expdata)

        tol = 0.002
        self.assertTrue(abs(self.cals.get_parameter_value("amp", 0, "x") - 0.333) < tol)
        self.assertTrue(abs(self.cals.get_parameter_value("amp", 0, "sx") - 0.333 / 2) < tol)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = RoughXSXAmplitudeCal(0, self.cals)
        config = exp.config()
        loaded_exp = RoughXSXAmplitudeCal.from_config(config)
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqual(config, loaded_exp.config())


class TestSpecializations(QiskitExperimentsTestCase):
    """Test the specialized versions of the calibration."""

    def setUp(self):
        """Setup the tests"""
        super().setUp()

        library = FixedFrequencyTransmon()

        self.backend = FakeArmonkV2()
        self.cals = Calibrations.from_backend(self.backend, libraries=[library])

        # Add some pulses on the 1-2 transition.
        d0 = pulse.DriveChannel(0)
        with pulse.build(name="x12") as x12:
            with pulse.frequency_offset(-300e6, d0):
                pulse.play(pulse.Drag(160, Parameter("amp"), 40, 0.0), d0)

        with pulse.build(name="sx12") as sx12:
            with pulse.frequency_offset(-300e6, d0):
                pulse.play(pulse.Drag(160, Parameter("amp"), 40, 0.0), d0)

        self.cals.add_schedule(x12, 0)
        self.cals.add_schedule(sx12, 0)
        self.cals.add_parameter_value(0.4, "amp", 0, "x12")
        self.cals.add_parameter_value(0.2, "amp", 0, "sx12")

    def test_ef_circuits(self):
        """Test that we get the expected circuits with calibrations for the EF experiment."""

        test_amps = [-0.5, 0, 0.5]
        rabi_ef = EFRoughXSXAmplitudeCal(0, self.cals, amplitudes=test_amps)

        circs = transpile(rabi_ef.circuits(), self.backend, inst_map=self.cals.default_inst_map)

        for circ, amp in zip(circs, test_amps):

            self.assertEqual(circ.count_ops()["x"], 1)
            self.assertEqual(circ.count_ops()["Rabi"], 1)

            d0 = pulse.DriveChannel(0)
            with pulse.build(name="x") as expected_x:
                pulse.play(pulse.Drag(160, 0.5, 40, 0), d0)

            with pulse.build(name="x12") as expected_x12:
                with pulse.frequency_offset(-300e6, d0):
                    pulse.play(pulse.Drag(160, amp, 40, 0), d0)

            self.assertEqual(circ.calibrations["x"][((0,), ())], expected_x)
            self.assertEqual(circ.calibrations["Rabi"][((0,), (amp,))], expected_x12)

    def test_ef_update(self):
        """Tes that we properly update the pulses on the 1<->2 transition."""

        self.assertTrue(np.allclose(self.cals.get_parameter_value("amp", 0, "x12"), 0.4))
        self.assertTrue(np.allclose(self.cals.get_parameter_value("amp", 0, "sx12"), 0.2))

        rabi_ef = EFRoughXSXAmplitudeCal(0, self.cals)
        expdata = rabi_ef.run(MockIQBackend(RabiHelper(amplitude_to_angle=np.pi * 1.5)))
        self.assertExperimentDone(expdata)

        tol = 0.002
        self.assertTrue(abs(self.cals.get_parameter_value("amp", 0, "x12") - 0.333) < tol)
        self.assertTrue(abs(self.cals.get_parameter_value("amp", 0, "sx12") - 0.333 / 2) < tol)
