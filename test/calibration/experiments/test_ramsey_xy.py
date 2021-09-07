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

"""Test Ramsey XY calibration experiment."""

from qiskit.test import QiskitTestCase
import qiskit.pulse as pulse

from qiskit_experiments.library import RamseyXY
from qiskit_experiments.test.mock_iq_backend import MockRamseyXY


class TestRamseyXY(QiskitTestCase):
    """Tests for the Ramsey XY experiment."""

    def test_end_to_end(self):
        """Test that we can run on a mock backend and perform a fit.

        This test also checks that we can pickup frequency shifts with different signs.
        """

        test_tol = 0.01

        ramsey = RamseyXY(0)
        ramsey.set_experiment_options(schedule=pulse.ScheduleBlock())

        for freq_shift in [2e6, -3e6]:
            test_data = ramsey.run(MockRamseyXY(freq_shift=freq_shift)).block_for_results()
            meas_shift = test_data.analysis_results(1).value.value
            self.assertTrue((meas_shift - freq_shift) < abs(test_tol * freq_shift))
