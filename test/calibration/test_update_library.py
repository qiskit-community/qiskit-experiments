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
from test.base import QiskitExperimentsTestCase
from test.test_qubit_spectroscopy import SpectroscopyBackend
import numpy as np

from qiskit.circuit import Parameter
from qiskit.qobj.utils import MeasLevel
import qiskit.pulse as pulse
from qiskit.test.mock import FakeAthens

from qiskit_experiments.library import QubitSpectroscopy
from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.calibration_management.update_library import Frequency


class TestAmplitudeUpdate(QiskitExperimentsTestCase):
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


class TestFrequencyUpdate(QiskitExperimentsTestCase):
    """Test the frequency update function in the update library."""

    def test_frequency(self):
        """Test calibrations update from spectroscopy."""

        qubit = 1
        peak_offset = 5.0e6
        backend = SpectroscopyBackend(line_width=2e6, freq_offset=peak_offset)
        freq01 = backend.defaults().qubit_freq_est[qubit]
        frequencies = np.linspace(freq01 - 10.0e6, freq01 + 10.0e6, 21)

        spec = QubitSpectroscopy(qubit, frequencies)
        spec.set_run_options(meas_level=MeasLevel.CLASSIFIED)
        exp_data = spec.run(backend)
        self.assertExperimentDone(exp_data)
        result = exp_data.analysis_results(1)
        value = result.value.n

        self.assertTrue(freq01 + peak_offset - 2e6 < value < freq01 + peak_offset + 2e6)
        self.assertEqual(result.quality, "good")

        # Test the integration with the Calibrations
        cals = Calibrations.from_backend(FakeAthens())
        self.assertNotEqual(cals.get_parameter_value(cals.__drive_freq_parameter__, qubit), value)
        Frequency.update(cals, exp_data)
        self.assertEqual(cals.get_parameter_value(cals.__drive_freq_parameter__, qubit), value)
