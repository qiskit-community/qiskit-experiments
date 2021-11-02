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

"""Test the calibration update library."""

from test.test_qubit_spectroscopy import SpectroscopyBackend
import numpy as np

from qiskit.circuit import Parameter
from qiskit.test import QiskitTestCase
from qiskit.qobj.utils import MeasLevel
import qiskit.pulse as pulse
from qiskit.test.mock import FakeAthens

from qiskit_experiments.library import FineXDrag, DragCal, QubitSpectroscopy
from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.calibration_management.update_library import (
    Frequency,
    Drag,
    FineDragUpdater,
)
from qiskit_experiments.calibration_management.backend_calibrations import BackendCalibrations
from qiskit_experiments.test.mock_iq_backend import DragBackend
from .experiments.test_fine_drag import FineDragTestBackend


class TestAmplitudeUpdate(QiskitTestCase):
    """Test the update functions in the update library."""

    def setUp(self):
        """Setup amplitude values."""
        super().setUp()
        self.cals = Calibrations()
        self.qubit = 1

        axp = Parameter("amp")
        chan = Parameter("ch0")
        with pulse.build(name="xp") as xp:
            pulse.play(pulse.Gaussian(duration=160, amp=axp, sigma=40), pulse.DriveChannel(chan))

        ax90p = Parameter("amp")
        with pulse.build(name="x90p") as x90p:
            pulse.play(pulse.Gaussian(duration=160, amp=ax90p, sigma=40), pulse.DriveChannel(chan))

        self.x90p = x90p

        self.cals.add_schedule(xp, num_qubits=1)
        self.cals.add_schedule(x90p, num_qubits=1)
        self.cals.add_parameter_value(0.2, "amp", self.qubit, "xp")
        self.cals.add_parameter_value(0.1, "amp", self.qubit, "x90p")


class TestFrequencyUpdate(QiskitTestCase):
    """Test the frequency update function in the update library."""

    def test_frequency(self):
        """Test calibrations update from spectroscopy."""

        qubit = 1
        peak_offset = 5.0e6
        backend = SpectroscopyBackend(line_width=2e6, freq_offset=peak_offset)
        freq01 = backend.defaults().qubit_freq_est[qubit]
        frequencies = np.linspace(freq01 - 10.0e6, freq01 + 10.0e6, 21) / 1e6

        spec = QubitSpectroscopy(qubit, frequencies, unit="MHz")
        spec.set_run_options(meas_level=MeasLevel.CLASSIFIED)
        exp_data = spec.run(backend)
        exp_data.block_for_results()
        result = exp_data.analysis_results(1)
        value = result.value.value

        self.assertTrue(freq01 + peak_offset - 2e6 < value < freq01 + peak_offset + 2e6)
        self.assertEqual(result.quality, "good")

        # Test the integration with the BackendCalibrations
        cals = BackendCalibrations(FakeAthens())
        self.assertNotEqual(cals.get_qubit_frequencies()[qubit], value)
        Frequency.update(cals, exp_data)
        self.assertEqual(cals.get_qubit_frequencies()[qubit], value)


class TestDragUpdate(QiskitTestCase):
    """Test the frequency update function in the update library."""

    def test_drag(self):
        """Test calibrations update from drag."""

        backend = DragBackend(gate_name="xp")
        beta = Parameter("β")
        qubit = 1
        test_tol = 0.02
        chan = Parameter("ch0")

        with pulse.build(backend=backend, name="xp") as x_plus:
            pulse.play(
                pulse.Drag(duration=160, amp=0.208519, sigma=40, beta=beta),
                pulse.DriveChannel(chan),
            )

        # Setup the calibrations
        cals = BackendCalibrations(backend)

        cals.add_schedule(x_plus, num_qubits=1)

        cals.add_parameter_value(0.2, "β", qubit, x_plus)
        cals.inst_map_add("xp", (qubit,))

        # Check that the inst_map has the default beta
        beta_val = cals.default_inst_map.get("xp", (qubit,)).blocks[0].pulse.beta
        self.assertEqual(beta_val, 0.2)

        # Run a Drag calibration experiment.
        drag = DragCal(qubit)
        drag.set_experiment_options(
            schedule=cals.get_schedule("xp", qubit, assign_params={"β": beta}),
        )

        exp_data = drag.run(backend)
        exp_data.block_for_results()
        result = exp_data.analysis_results(1)

        # Test the fit for good measure.
        self.assertTrue(abs(result.value.value - backend.ideal_beta) < test_tol)
        self.assertEqual(result.quality, "good")

        # Check schedules pre-update
        expected = x_plus.assign_parameters({beta: 0.2, chan: 1}, inplace=False)
        self.assertEqual(cals.get_schedule("xp", qubit), expected)

        Drag.update(cals, exp_data, parameter="β", schedule="xp")

        # Check schedules post-update
        expected = x_plus.assign_parameters({beta: result.value.value, chan: 1}, inplace=False)
        self.assertEqual(cals.get_schedule("xp", qubit), expected)

        # Check the inst map post update
        beta_val = cals.default_inst_map.get("xp", (qubit,)).blocks[0].pulse.beta
        self.assertTrue(np.allclose(beta_val, result.value.value))


class TestFineDragUpdate(QiskitTestCase):
    """A class to test fine DRAG updates."""

    def test_fine_drag(self):
        """Test that we can update from a fine DRAG experiment."""

        d_theta = 0.03  # rotation error per single gate.
        backend = FineDragTestBackend(error=d_theta)

        qubit = 0
        test_tol = 0.005
        beta = Parameter("β")
        chan = Parameter("ch0")

        with pulse.build(backend=backend, name="xp") as x_plus:
            pulse.play(
                pulse.Drag(duration=160, amp=0.208519, sigma=40, beta=beta),
                pulse.DriveChannel(chan),
            )

        # Setup the calibrations
        cals = BackendCalibrations(backend)

        cals.add_schedule(x_plus, num_qubits=1)

        old_beta = 0.2
        cals.add_parameter_value(old_beta, "β", qubit, x_plus)
        cals.inst_map_add("xp", (qubit,))

        # Check that the inst_map has the default beta
        beta_val = cals.default_inst_map.get("xp", (qubit,)).blocks[0].pulse.beta
        self.assertEqual(beta_val, old_beta)

        # Run a Drag calibration experiment.
        drag = FineXDrag(qubit)
        drag.set_experiment_options(schedule=cals.get_schedule("xp", qubit))
        drag.set_transpile_options(basis_gates=["rz", "xp", "ry"])
        exp_data = drag.run(backend).block_for_results()

        result = exp_data.analysis_results(1)

        # Test the fit for good measure.
        self.assertTrue(abs(result.value.value - d_theta) < test_tol)
        self.assertEqual(result.quality, "good")

        # Check schedules pre-update
        expected = x_plus.assign_parameters({beta: 0.2, chan: qubit}, inplace=False)
        self.assertEqual(cals.get_schedule("xp", qubit), expected)

        FineDragUpdater.update(cals, exp_data, parameter="β", schedule="xp")

        # Check schedules post-update. Here the FineDragTestBackend has a leakage
        # of 0.03 per gate so the DRAG update rule
        # -np.sqrt(np.pi) * d_theta * sigma / target_angle ** 2 should give a new beta of
        # 0.2 - np.sqrt(np.pi) * 0.03 * 40 / (np.pi ** 2)
        new_beta = old_beta - np.sqrt(np.pi) * result.value.value * 40 / np.pi ** 2
        expected = x_plus.assign_parameters({beta: new_beta, chan: qubit}, inplace=False)
        self.assertEqual(cals.get_schedule("xp", qubit), expected)

        # Check the inst map post update
        beta_val = cals.default_inst_map.get("xp", (qubit,)).blocks[0].pulse.beta
        self.assertTrue(np.allclose(beta_val, new_beta))
