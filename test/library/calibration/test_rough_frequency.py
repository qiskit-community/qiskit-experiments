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

from qiskit.providers.fake_provider import FakeArmonkV2

from qiskit_experiments.framework import BackendData
from qiskit_experiments.library import RoughFrequencyCal
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.test.mock_iq_backend import MockIQBackend
from qiskit_experiments.test.mock_iq_helpers import MockIQSpectroscopyHelper as SpectroscopyHelper


class TestRoughFrequency(QiskitExperimentsTestCase):
    """Tests for the rough frequency calibration experiment."""

    def test_init(self):
        """Test that initialization."""

        qubit = 1
        cals = Calibrations.from_backend(FakeArmonkV2())
        frequencies = [1000, 2000, 3000]
        auto_update = False
        absolute = False

        freq = RoughFrequencyCal(
            qubit, cals, frequencies, auto_update=auto_update, absolute=absolute
        )

        self.assertEqual(freq.physical_qubits, (qubit,))
        self.assertEqual(freq._frequencies, frequencies)
        self.assertEqual(freq._absolute, False)
        self.assertEqual(freq.auto_update, False)

    def test_update_calibrations(self):
        """Test that we can properly update an instance of Calibrations."""

        freq01 = BackendData(FakeArmonkV2()).drive_freqs[0]

        backend = MockIQBackend(
            experiment_helper=SpectroscopyHelper(
                freq_offset=5e6,
                line_width=2e6,
                iq_cluster_centers=[((-1.0, -1.0), (1.0, 1.0))],
                iq_cluster_width=[0.2],
            ),
        )
        backend._configuration.basis_gates = ["x"]
        backend._configuration.timing_constraints = {"granularity": 16}

        backend.defaults().qubit_freq_est = [freq01, freq01]

        library = FixedFrequencyTransmon(basis_gates=["x", "sx"])
        cals = Calibrations.from_backend(FakeArmonkV2(), libraries=[library])

        prev_freq = cals.get_parameter_value(cals.__drive_freq_parameter__, (0,))
        self.assertEqual(prev_freq, freq01)

        frequencies = np.linspace(freq01 - 10.0e6, freq01 + 10.0e6, 21)

        expdata = RoughFrequencyCal(0, cals, frequencies).run(backend)
        self.assertExperimentDone(expdata)

        # Check the updated frequency which should be shifted by 5MHz.
        post_freq = cals.get_parameter_value(cals.__drive_freq_parameter__, (0,))
        self.assertTrue(abs(post_freq - freq01 - 5e6) < 1e6)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        cals = Calibrations.from_backend(FakeArmonkV2())
        frequencies = [1, 2, 3]
        exp = RoughFrequencyCal(0, cals, frequencies)
        loaded_exp = RoughFrequencyCal.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))
