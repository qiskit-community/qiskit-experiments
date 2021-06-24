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

from test.calibration.experiments.test_rabi import RabiBackend
from test.test_qubit_spectroscopy import SpectroscopyBackend
import numpy as np

from qiskit.circuit import Parameter
from qiskit.test import QiskitTestCase
from qiskit.qobj.utils import MeasLevel
import qiskit.pulse as pulse
from qiskit.test.mock import FakeAthens

from qiskit_experiments.calibration.experiments.rabi import Rabi
from qiskit_experiments.calibration.calibrations import Calibrations
from qiskit_experiments.calibration.exceptions import CalibrationError
from qiskit_experiments.calibration.update_library import Frequency, Amplitude
from qiskit_experiments.calibration.backend_calibrations import BackendCalibrations
from qiskit_experiments.characterization.qubit_spectroscopy import QubitSpectroscopy
from qiskit_experiments.analysis import get_opt_value


class TestCalibrationsUpdate(QiskitTestCase):
    """Test the update functions in the update library."""

    def test_amplitude(self):
        """Test amplitude update from Rabi."""

        cals = Calibrations()

        amp = Parameter("amp")
        chan = Parameter("ch0")
        with pulse.build(name="xp") as xp:
            pulse.play(pulse.Gaussian(duration=160, amp=amp, sigma=40), pulse.DriveChannel(chan))

        amp = Parameter("amp")
        with pulse.build(name="x90p") as x90p:
            pulse.play(pulse.Gaussian(duration=160, amp=amp, sigma=40), pulse.DriveChannel(chan))

        cals.add_schedule(xp)
        cals.add_schedule(x90p)

        qubit = 1
        rabi = Rabi(qubit)
        rabi.set_experiment_options(amplitudes=np.linspace(-0.95, 0.95, 21))
        exp_data = rabi.run(RabiBackend())

        for qubit_ in [0, 1]:
            with self.assertRaises(CalibrationError):
                cals.get_schedule("xp", qubits=qubit_)

        to_update = [(np.pi, "amp", "xp"), (np.pi / 2, "amp", x90p)]

        self.assertEqual(len(cals.parameters_table()), 0)

        Amplitude.update(cals, exp_data, angles_schedules=to_update)

        with self.assertRaises(CalibrationError):
            cals.get_schedule("xp", qubits=0)

        self.assertEqual(len(cals.parameters_table()), 2)

        # Now check the corresponding schedules
        result = exp_data.analysis_result(-1)
        rate = 2 * np.pi * result["popt"][1]
        amp = np.round(np.pi / rate, decimals=8)
        with pulse.build(name="xp") as expected:
            pulse.play(pulse.Gaussian(160, amp, 40), pulse.DriveChannel(qubit))

        self.assertEqual(cals.get_schedule("xp", qubits=qubit), expected)

        amp = np.round(0.5 * np.pi / rate, decimals=8)
        with pulse.build(name="xp") as expected:
            pulse.play(pulse.Gaussian(160, amp, 40), pulse.DriveChannel(qubit))

        self.assertEqual(cals.get_schedule("x90p", qubits=qubit), expected)

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
        result = exp_data.analysis_result(0)

        value = get_opt_value(result, "freq")

        self.assertTrue(freq01 + peak_offset - 2e6 < value < freq01 + peak_offset + 2e6)
        self.assertEqual(result["quality"], "computer_good")

        # Test the integration with the BackendCalibrations
        cals = BackendCalibrations(FakeAthens())
        self.assertNotEqual(cals.get_qubit_frequencies()[qubit], result["popt"][2])
        Frequency.update(cals, exp_data)
        self.assertEqual(cals.get_qubit_frequencies()[qubit], result["popt"][2])
