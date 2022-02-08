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

"""Test Ramsey XY experiments."""

import unittest
from test.base import QiskitExperimentsTestCase
from qiskit.test.mock import FakeArmonk

from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.library import RamseyXY, FrequencyCal
from qiskit_experiments.test.mock_iq_backend import MockRamseyXY


class TestRamseyXY(QiskitExperimentsTestCase):
    """Tests for the Ramsey XY experiment."""

    def setUp(self):
        """Initialize some cals."""
        super().setUp()

        library = FixedFrequencyTransmon()
        self.cals = Calibrations.from_backend(FakeArmonk(), library)

    def test_end_to_end(self):
        """Test that we can run on a mock backend and perform a fit.

        This test also checks that we can pickup frequency shifts with different signs.
        """

        test_tol = 0.01

        ramsey = RamseyXY(0)

        for freq_shift in [2e6, -3e6]:
            test_data = ramsey.run(MockRamseyXY(freq_shift=freq_shift))
            self.assertExperimentDone(test_data)
            meas_shift = test_data.analysis_results(1).value.value
            self.assertTrue((meas_shift - freq_shift) < abs(test_tol * freq_shift))

    def test_update_calibrations(self):
        """Test that the calibration version of the experiment updates the cals."""

        tol = 1e4  # 10 kHz resolution

        freq_name = self.cals.__drive_freq_parameter__

        # Check qubit frequency before running the cal
        f01 = self.cals.get_parameter_value(freq_name, 0)
        self.assertTrue(len(self.cals.parameters_table(parameters=[freq_name])["data"]), 1)
        self.assertEqual(f01, FakeArmonk().defaults().qubit_freq_est[0])

        freq_shift = 4e6
        osc_shift = 2e6
        backend = MockRamseyXY(freq_shift=freq_shift + osc_shift)  # oscillation with 6 MHz
        expdata = FrequencyCal(0, self.cals, backend, osc_freq=osc_shift).run()
        self.assertExperimentDone(expdata)

        # Check that qubit frequency after running the cal is shifted by freq_shift, i.e. 4 MHz.
        f01 = self.cals.get_parameter_value(freq_name, 0)
        self.assertTrue(len(self.cals.parameters_table(parameters=[freq_name])["data"]), 2)
        self.assertTrue(abs(f01 - (freq_shift + FakeArmonk().defaults().qubit_freq_est[0])) < tol)

    def test_ramseyxy_experiment_config(self):
        """Test RamseyXY config roundtrips"""
        exp = RamseyXY(0)
        loaded_exp = RamseyXY.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_ramseyxy_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = RamseyXY(0)
        self.assertRoundTripSerializable(exp, self.json_equiv)

    def test_cal_experiment_config(self):
        """Test FrequencyCal config roundtrips"""
        exp = FrequencyCal(0, self.cals)
        loaded_exp = FrequencyCal.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    @unittest.skip("Cal experiments are not yet JSON serializable")
    def test_freqcal_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = FrequencyCal(0, self.cals)
        self.assertRoundTripSerializable(exp, self.json_equiv)
