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

"""Test drag calibration experiment."""

from test.base import QiskitExperimentsTestCase
import unittest
import numpy as np

from qiskit.circuit import Parameter
from qiskit.exceptions import QiskitError
from qiskit.pulse import DriveChannel, Drag
import qiskit.pulse as pulse
from qiskit.qobj.utils import MeasLevel
from qiskit import transpile

from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.library import RoughDrag, RoughDragCal
from qiskit_experiments.test.mock_iq_backend import DragBackend
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.calibration_management import Calibrations


class TestDragEndToEnd(QiskitExperimentsTestCase):
    """Test the drag experiment."""

    def setUp(self):
        """Setup some schedules."""
        super().setUp()

        beta = Parameter("β")

        with pulse.build(name="xp") as xp:
            pulse.play(Drag(duration=160, amp=0.208519, sigma=40, beta=beta), DriveChannel(0))

        self.x_plus = xp
        self.test_tol = 0.05

    def test_reps(self):
        """Test that setting reps raises and error if reps is not of length three."""

        drag = RoughDrag(0, self.x_plus)

        with self.assertRaises(CalibrationError):
            drag.set_experiment_options(reps=[1, 2, 3, 4])

    def test_end_to_end(self):
        """Test the drag experiment end to end."""

        backend = DragBackend(gate_name="Drag(xp)")

        drag = RoughDrag(1, self.x_plus)

        expdata = drag.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results(1)

        self.assertTrue(abs(result.value.n - backend.ideal_beta) < self.test_tol)
        self.assertEqual(result.quality, "good")

        # Small leakage will make the curves very flat, in this case one should
        # rather increase beta.
        backend = DragBackend(error=0.0051, gate_name="Drag(xp)")

        drag = RoughDrag(0, self.x_plus)
        drag.analysis.set_options(p0={"beta": 1.2})
        exp_data = drag.run(backend)
        self.assertExperimentDone(exp_data)
        result = exp_data.analysis_results(1)

        self.assertTrue(abs(result.value.n - backend.ideal_beta) < self.test_tol)
        self.assertEqual(result.quality, "good")

        # Large leakage will make the curves oscillate quickly.
        backend = DragBackend(error=0.05, gate_name="Drag(xp)")

        drag = RoughDrag(1, self.x_plus, betas=np.linspace(-4, 4, 31))
        drag.set_run_options(shots=200)
        drag.analysis.set_options(p0={"beta": 1.8, "freq0": 0.08, "freq1": 0.16, "freq2": 0.32})
        exp_data = drag.run(backend)
        self.assertExperimentDone(exp_data)
        result = exp_data.analysis_results(1)

        meas_level = exp_data.experiment_config().run_options["meas_level"]

        self.assertEqual(meas_level, MeasLevel.CLASSIFIED)
        self.assertTrue(abs(result.value.n - backend.ideal_beta) < self.test_tol)
        self.assertEqual(result.quality, "good")


class TestDragCircuits(QiskitExperimentsTestCase):
    """Test the circuits of the drag calibration."""

    def setUp(self):
        """Setup some schedules."""
        super().setUp()

        beta = Parameter("β")

        with pulse.build(name="xp") as xp:
            pulse.play(Drag(duration=160, amp=0.208519, sigma=40, beta=beta), DriveChannel(0))

        self.x_plus = xp

    def test_default_circuits(self):
        """Test the default circuit."""

        backend = DragBackend(error=0.005, gate_name="Drag(xp)")

        drag = RoughDrag(0, self.x_plus)
        drag.set_experiment_options(reps=[2, 4, 8])
        drag.backend = DragBackend(gate_name="Drag(xp)")
        circuits = drag.circuits()

        for idx, expected in enumerate([4, 8, 16]):
            ops = transpile(circuits[idx * 51], backend).count_ops()
            self.assertEqual(ops["Drag(xp)"], expected)

    def test_raise_multiple_parameter(self):
        """Check that the experiment raises with unassigned parameters."""

        beta = Parameter("β")
        amp = Parameter("amp")

        with pulse.build(name="xp") as xp:
            pulse.play(Drag(duration=160, amp=amp, sigma=40, beta=beta), DriveChannel(0))

        with self.assertRaises(QiskitError):
            RoughDrag(1, xp, betas=np.linspace(-3, 3, 21))


class TestRoughDragCalUpdate(QiskitExperimentsTestCase):
    """Test that a Drag calibration experiment properly updates the calibrations."""

    def setUp(self):
        """Setup the tests"""
        super().setUp()

        library = FixedFrequencyTransmon()

        self.backend = DragBackend(gate_name="Drag(x)")
        self.cals = Calibrations.from_backend(self.backend, library)
        self.test_tol = 0.05

    def test_update(self):
        """Test that running RoughDragCal updates the calibrations."""

        qubit = 0
        prev_beta = self.cals.get_parameter_value("β", (0,), "x")
        self.assertEqual(prev_beta, 0)

        expdata = RoughDragCal(qubit, self.cals, backend=self.backend).run()
        self.assertExperimentDone(expdata)

        new_beta = self.cals.get_parameter_value("β", (0,), "x")
        self.assertTrue(abs(new_beta - self.backend.ideal_beta) < self.test_tol)
        self.assertTrue(abs(new_beta) > self.test_tol)

    def test_dragcal_experiment_config(self):
        """Test RoughDragCal config can round trip"""
        exp = RoughDragCal(0, self.cals, backend=self.backend)
        loaded_exp = RoughDragCal.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    @unittest.skip("Calibration experiments are not yet JSON serializable")
    def test_dragcal_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = RoughDragCal(0, self.cals)
        self.assertRoundTripSerializable(exp, self.json_equiv)

    def test_drag_experiment_config(self):
        """Test RoughDrag config can roundtrip"""
        with pulse.build(name="xp") as sched:
            pulse.play(pulse.Drag(160, 0.5, 40, Parameter("β")), pulse.DriveChannel(0))
        exp = RoughDrag(0, backend=self.backend, schedule=sched)
        loaded_exp = RoughDrag.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    @unittest.skip("Schedules are not yet JSON serializable")
    def test_drag_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        with pulse.build(name="xp") as sched:
            pulse.play(pulse.Drag(160, 0.5, 40, Parameter("β")), pulse.DriveChannel(0))
        exp = RoughDrag(0, backend=self.backend, schedule=sched)
        self.assertRoundTripSerializable(exp, self.json_equiv)
