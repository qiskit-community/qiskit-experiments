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

"""Test fine drag calibration experiment."""

from test.base import QiskitExperimentsTestCase
import numpy as np

from qiskit import pulse
from qiskit.circuit import Gate
from qiskit_ibm_runtime.fake_provider import FakeArmonkV2

from qiskit_experiments.library import FineDrag, FineXDrag, FineDragCal
from qiskit_experiments.test.mock_iq_backend import MockIQBackend
from qiskit_experiments.test.mock_iq_helpers import MockIQFineDragHelper as FineDragHelper
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon


class TestFineDrag(QiskitExperimentsTestCase):
    """Tests of the fine DRAG experiment."""

    def setUp(self):
        """Setup test variables."""
        super().setUp()

        with pulse.build(name="Drag") as schedule:
            pulse.play(pulse.Drag(160, 0.5, 40, 0.3), pulse.DriveChannel(0))

        self.schedule = schedule

    def test_circuits(self):
        """Test the circuits of the experiment."""

        drag = FineDrag([0], Gate("Drag", num_qubits=1, params=[]))
        drag.set_experiment_options(schedule=self.schedule)
        drag.backend = FakeArmonkV2()
        for circuit in drag.circuits()[1:]:
            for idx, name in enumerate(["Drag", "rz", "Drag", "rz"]):
                self.assertEqual(circuit.data[idx][0].name, name)

    def test_end_to_end(self):
        """A simple test to check if the experiment will run and fit data."""

        drag = FineDrag([0], Gate("Drag", num_qubits=1, params=[]))
        drag.set_experiment_options(schedule=self.schedule)
        drag.set_transpile_options(basis_gates=["rz", "Drag", "sx"])
        exp_data = drag.run(MockIQBackend(FineDragHelper()))
        self.assertExperimentDone(exp_data)

        self.assertEqual(exp_data.analysis_results("d_theta").quality, "good")

    def test_end_to_end_no_schedule(self):
        """Test that we can run without a schedule."""
        exp_data = FineXDrag([0]).run(MockIQBackend(FineDragHelper()))
        self.assertExperimentDone(exp_data)

        self.assertEqual(exp_data.analysis_results("d_theta").quality, "good")

    def test_circuits_roundtrip_serializable(self):
        """Test circuits serialization of the experiment."""
        drag = FineDrag([0], Gate("Drag", num_qubits=1, params=[]))
        drag.set_experiment_options(schedule=self.schedule)
        drag.backend = FakeArmonkV2()
        self.assertRoundTripSerializable(drag._transpiled_circuits())

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = FineDrag([0], Gate("Drag", num_qubits=1, params=[]))
        config = exp.config()
        loaded_exp = FineDrag.from_config(config)
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqual(config, loaded_exp.config())


class TestFineDragCal(QiskitExperimentsTestCase):
    """Test the calibration version of the fine drag experiment."""

    def setUp(self):
        """Setup the test."""
        super().setUp()

        library = FixedFrequencyTransmon()
        self.backend = MockIQBackend(FineDragHelper())
        self.cals = Calibrations.from_backend(self.backend, libraries=[library])

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = FineDragCal([0], self.cals, schedule_name="x")
        config = exp.config()
        loaded_exp = FineDragCal.from_config(config)
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqual(config, loaded_exp.config())

    def test_update_cals(self):
        """Test that the calibrations are updated."""

        init_beta = 0.0

        drag_cal = FineDragCal([0], self.cals, "x", self.backend)

        circs = drag_cal._transpiled_circuits()

        with pulse.build(name="x") as expected_x:
            pulse.play(pulse.Drag(160, 0.5, 40, 0), pulse.DriveChannel(0))

        with pulse.build(name="sx") as expected_sx:
            pulse.play(pulse.Drag(160, 0.25, 40, 0), pulse.DriveChannel(0))

        self.assertEqual(circs[5].calibrations["x"][((0,), ())], expected_x)
        self.assertEqual(circs[5].calibrations["sx"][((0,), ())], expected_sx)

        # run the calibration experiment. This should update the beta parameter of x which we test.
        exp_data = drag_cal.run(self.backend)
        self.assertExperimentDone(exp_data)
        d_theta = exp_data.analysis_results("d_theta").value.n
        sigma = 40
        target_angle = np.pi
        new_beta = -np.sqrt(np.pi) * d_theta * sigma / target_angle**2

        circs = drag_cal._transpiled_circuits()

        x_cal = circs[5].calibrations["x"][((0,), ())]

        # Requires allclose due to numerical precision.
        self.assertTrue(np.allclose(x_cal.blocks[0].pulse.beta, new_beta))
        self.assertFalse(np.allclose(x_cal.blocks[0].pulse.beta, init_beta))
        self.assertEqual(circs[5].calibrations["sx"][((0,), ())], expected_sx)
