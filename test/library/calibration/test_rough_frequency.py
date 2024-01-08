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

"""Rough frequency calibration tests."""
from test.base import QiskitExperimentsTestCase

import numpy as np

from qiskit_experiments.framework import BackendData
from qiskit_experiments.library import RoughFrequencyCal
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend


class TestRoughFrequency(QiskitExperimentsTestCase):
    """Tests for the rough frequency calibration experiment."""

    def setUp(self):
        """Setup the tests."""
        super().setUp()
        self.backend = SingleTransmonTestBackend(noise=False, atol=1e-3)

    def test_init(self):
        """Test that initialization."""

        qubit = 0
        cals = Calibrations.from_backend(self.backend)
        frequencies = [1000, 2000, 3000]
        auto_update = False
        absolute = False

        freq = RoughFrequencyCal(
            [qubit], cals, frequencies, auto_update=auto_update, absolute=absolute
        )

        self.assertEqual(freq.physical_qubits, (qubit,))
        self.assertEqual(freq._frequencies, frequencies)
        self.assertEqual(freq._absolute, False)
        self.assertEqual(freq.auto_update, False)

    def test_update_calibrations(self):
        """Test that we can properly update an instance of Calibrations."""

        freq01 = BackendData(self.backend).drive_freqs[0]

        backend_5mhz = SingleTransmonTestBackend(
            qubit_frequency=freq01 + 5e6, noise=False, atol=1e-3
        )

        library = FixedFrequencyTransmon()
        cals = Calibrations.from_backend(self.backend, libraries=[library])

        prev_freq = cals.get_parameter_value("drive_freq", (0,))
        self.assertEqual(prev_freq, freq01)

        frequencies = np.linspace(freq01 - 10.0e6, freq01 + 10.0e6, 11)

        spec = RoughFrequencyCal([0], cals, frequencies, backend=backend_5mhz)
        spec.set_experiment_options(amp=0.005)
        expdata = spec.run()
        self.assertExperimentDone(expdata)

        # Check the updated frequency which should be shifted by 5MHz.
        post_freq = cals.get_parameter_value("drive_freq", (0,))
        self.assertTrue(abs(post_freq - freq01 - 5e6) < 1e6)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        cals = Calibrations.from_backend(self.backend)
        frequencies = [1, 2, 3]
        exp = RoughFrequencyCal([0], cals, frequencies)
        loaded_exp = RoughFrequencyCal.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqualExtended(exp, loaded_exp)
