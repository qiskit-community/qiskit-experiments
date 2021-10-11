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

from qiskit.circuit.library import XGate, SXGate
from qiskit.test import QiskitTestCase
from qiskit.pulse import DriveChannel, Drag
import qiskit.pulse as pulse

from qiskit_experiments.library import FineAmplitude, FineXAmplitude, FineSXAmplitude
from qiskit_experiments.test.mock_iq_backend import MockFineAmp
from qiskit_experiments.exceptions import CalibrationError


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


class TestSpecializations(QiskitTestCase):
    """Test the options of the specialized classes."""

    def test_fine_x_amp(self):
        """Test the fine X amplitude."""

        exp = FineXAmplitude(0)

        self.assertTrue(exp.experiment_options.add_sx)
        self.assertTrue(exp.experiment_options.add_xp_circuit)
        self.assertEqual(exp.analysis_options.angle_per_gate, np.pi)
        self.assertEqual(exp.analysis_options.phase_offset, np.pi / 2)
        self.assertEqual(exp.experiment_options.gate_type, XGate)

    def test_fine_sx_amp(self):
        """Test the fine SX amplitude."""

        exp = FineSXAmplitude(0)

        self.assertFalse(exp.experiment_options.add_sx)
        self.assertFalse(exp.experiment_options.add_xp_circuit)

        expected = [1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 21, 23, 25]
        self.assertEqual(exp.experiment_options.repetitions, expected)
        self.assertEqual(exp.analysis_options.angle_per_gate, np.pi / 2)
        self.assertEqual(exp.analysis_options.phase_offset, 0)
        self.assertEqual(exp.experiment_options.gate_type, SXGate)

    def test_end_to_end_no_schedule(self):
        """Test the experiment end to end."""

        amp_cal = FineXAmplitude(0)
        backend = MockFineAmp(-np.pi * 0.07, np.pi, "x")

        expdata = amp_cal.run(backend).block_for_results()
        result = expdata.analysis_results(1)
        d_theta = result.value.value

        tol = 0.04

        self.assertTrue(abs(d_theta - backend.angle_error) < tol)
        self.assertEqual(result.quality, "good")
        self.assertIsNone(amp_cal.experiment_options.schedule)
