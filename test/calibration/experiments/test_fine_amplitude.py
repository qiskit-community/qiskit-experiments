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
from qiskit.test.mock import FakeArmonk

from qiskit_experiments.library import FineAmplitude, FineXAmplitude
from qiskit_experiments.test.mock_iq_backend import MockFineAmp
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management import BackendCalibrations
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon


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

        expdata = amp_cal.run(backend)
        expdata.block_for_results()
        result = expdata.analysis_results(1)
        d_theta = result.value.value

        tol = 0.04

        self.assertTrue(abs(d_theta - backend.angle_error) < tol)
        self.assertEqual(result.quality, "good")

    def test_end_to_end_over_rotation(self):
        """Test the experiment end to end."""

        amp_cal = FineAmplitude(0)
        amp_cal.set_schedule(
            schedule=self.x_plus, angle_per_gate=np.pi, add_xp_circuit=True, add_sx=True
        )
        amp_cal.set_analysis_options(number_guesses=6)

        backend = MockFineAmp(np.pi * 0.07, np.pi, "xp")

        expdata = amp_cal.run(backend)
        expdata.block_for_results()
        result = expdata.analysis_results(1)
        d_theta = result.value.value

        tol = 0.04

        self.assertTrue(abs(d_theta - backend.angle_error) < tol)
        self.assertEqual(result.quality, "good")

    def test_zero_angle_per_gate(self):
        """Test that we cannot set angle per gate to zero."""
        amp_cal = FineAmplitude(0)

        with self.assertRaises(CalibrationError):
            amp_cal.set_schedule(
                schedule=self.x_plus, angle_per_gate=0.0, add_xp_circuit=True, add_sx=True
            )

    def test_update_calibrations(self):
        """Test that calibrations are updated."""

        library = FixedFrequencyTransmon(basis_gates=["x", "sx"], default_values={"duration": 320})
        cals = BackendCalibrations(FakeArmonk(), library=library)

        pre_cal_amp = cals.get_parameter_value("amp", (0,), "x")

        target_angle = np.pi
        backend = MockFineAmp(target_angle * 0.01, target_angle, "x")
        exp_data = FineXAmplitude(0, calibrations=cals).run(backend)

        result = [
            r for r in exp_data.analysis_results() if r.name.startswith("@Parameters_")
        ][0]
        d_theta = result.value.value[result.extra["popt_keys"].index("d_theta")]

        post_cal_amp = cals.get_parameter_value("amp", (0,), "x")

        self.assertEqual(post_cal_amp, pre_cal_amp * target_angle / (target_angle + d_theta))

        # Test that the circuit has a calibration for the sx and x gate.
        circs = FineXAmplitude(0, calibrations=cals).circuits()
        self.assertTrue("sx" in circs[3].calibrations)
        self.assertTrue("x" in circs[3].calibrations)


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
