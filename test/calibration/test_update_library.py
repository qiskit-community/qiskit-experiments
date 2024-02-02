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
import numpy as np

from qiskit.qobj.utils import MeasLevel
from qiskit_ibm_runtime.fake_provider import FakeAthensV2

from qiskit_experiments.framework import BackendData
from qiskit_experiments.library import QubitSpectroscopy
from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.calibration_management.update_library import Frequency
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.test.mock_iq_backend import MockIQBackend
from qiskit_experiments.test.mock_iq_helpers import MockIQSpectroscopyHelper as SpectroscopyHelper


class TestFrequencyUpdate(QiskitExperimentsTestCase):
    """Test the frequency update function in the update library."""

    def test_frequency(self):
        """Test calibrations update from spectroscopy."""

        qubit = 1
        peak_offset = 5.0e6
        backend = MockIQBackend(
            experiment_helper=SpectroscopyHelper(
                freq_offset=peak_offset,
                iq_cluster_centers=[((-1.0, -1.0), (1.0, 1.0))],
                iq_cluster_width=[0.2],
            ),
        )

        freq01 = BackendData(backend).drive_freqs[qubit]
        frequencies = np.linspace(freq01 - 10.0e6, freq01 + 10.0e6, 21)

        spec = QubitSpectroscopy([qubit], frequencies)
        spec.set_run_options(meas_level=MeasLevel.CLASSIFIED)
        exp_data = spec.run(backend)
        self.assertExperimentDone(exp_data)
        result = exp_data.analysis_results("f01")
        value = result.value.n

        self.assertTrue(freq01 + peak_offset - 2e6 < value < freq01 + peak_offset + 2e6)
        self.assertEqual(result.quality, "good")

        # Test the integration with the Calibrations
        cals = Calibrations.from_backend(FakeAthensV2(), libraries=[FixedFrequencyTransmon()])
        self.assertNotEqual(cals.get_parameter_value("drive_freq", qubit), value)
        Frequency.update(cals, exp_data)
        self.assertEqual(cals.get_parameter_value("drive_freq", qubit), value)
