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

from ddt import ddt, data, named_data
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeArmonkV2

from qiskit_experiments.calibration_management.calibrations import Calibrations
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.framework import AnalysisStatus, BackendData, BaseAnalysis
from qiskit_experiments.library import RamseyXY, FrequencyCal
from qiskit_experiments.test.mock_iq_backend import MockIQBackend
from qiskit_experiments.test.mock_iq_helpers import MockIQRamseyXYHelper as RamseyXYHelper


@ddt
class TestRamseyXY(QiskitExperimentsTestCase):
    """Tests for the Ramsey XY experiment."""

    def setUp(self):
        """Initialize some cals."""
        super().setUp()

        library = FixedFrequencyTransmon()
        self.cals = Calibrations.from_backend(FakeArmonkV2(), libraries=[library])

    @named_data(
        ["no_backend", None], ["fake_backend", FakeArmonkV2()], ["aer_backend", AerSimulator()]
    )
    def test_circuits(self, backend: str):
        """Test circuit generation does not error"""
        delays = [1e-6, 5e-6, 10e-6]
        circs = RamseyXY([0], delays=delays, backend=backend).circuits()
        # Deduplicate xvals
        xvals = sorted({c.metadata["xval"] for c in circs})
        for delay, xval in zip(delays, xvals):
            self.assertAlmostEqual(delay, xval)

    @data(2e6, -3e6, 1e3, 0.0, 0.2e6, 0.3e6)
    def test_end_to_end(self, freq_shift: float):
        """Test that we can run on a mock backend and perform a fit.

        This test also checks that we can pickup frequency shifts with different signs.
        """
        test_tol = 0.03
        abs_tol = max(1e3, abs(freq_shift) * test_tol)

        exp_helper = RamseyXYHelper()
        ramsey = RamseyXY([0])
        ramsey.backend = MockIQBackend(exp_helper)

        exp_helper.freq_shift = freq_shift
        test_data = ramsey.run()
        self.assertExperimentDone(test_data)

        freq_est_data = test_data.analysis_results("freq")
        self.assertAlmostEqual(freq_est_data.value.n, freq_shift, delta=abs_tol)
        self.assertLess(freq_est_data.chisq, 3.0)

    def test_update_calibrations(self):
        """Test that the calibration version of the experiment updates the cals."""

        tol = 1e4  # 10 kHz resolution

        freq_name = "drive_freq"

        # Check qubit frequency before running the cal
        f01 = self.cals.get_parameter_value(freq_name, 0)
        self.assertTrue(len(self.cals.parameters_table(parameters=[freq_name])["data"]), 1)
        self.assertEqual(f01, BackendData(FakeArmonkV2()).drive_freqs[0])

        freq_shift = 4e6
        osc_shift = 2e6

        # oscillation with 6 MHz
        backend = MockIQBackend(RamseyXYHelper(freq_shift=freq_shift + osc_shift))
        expdata = FrequencyCal([0], self.cals, backend, osc_freq=osc_shift).run()
        self.assertExperimentDone(expdata)

        # Check that qubit frequency after running the cal is shifted by freq_shift, i.e. 4 MHz.
        f01 = self.cals.get_parameter_value(freq_name, 0)
        self.assertTrue(len(self.cals.parameters_table(parameters=[freq_name])["data"]), 2)
        self.assertLess(abs(f01 - (freq_shift + BackendData(FakeArmonkV2()).drive_freqs[0])), tol)

    def test_update_with_failed_analysis(self):
        """Test that calibration update handles analysis producing no results

        Here we test that the experiment does not raise an unexpected exception
        or hang indefinitely. Since there are no analysis results, we expect
        that the calibration update will result in an ERROR status.
        """
        backend = MockIQBackend(RamseyXYHelper(freq_shift=0))

        class NoResults(BaseAnalysis):
            """Simple analysis class that generates no results"""

            def _run_analysis(self, experiment_data):
                return ([], [])

        expt = FrequencyCal([0], self.cals, backend, auto_update=True)
        expt.analysis = NoResults()
        expdata = expt.run().block_for_results(timeout=3)
        self.assertEqual(expdata.analysis_status(), AnalysisStatus.ERROR)

    def test_ramseyxy_experiment_config(self):
        """Test RamseyXY config roundtrips"""
        exp = RamseyXY([0])
        loaded_exp = RamseyXY.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqualExtended(exp, loaded_exp)

    def test_ramseyxy_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = RamseyXY([0])
        self.assertRoundTripSerializable(exp)

    def test_circuit_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        backend = FakeArmonkV2()
        exp = RamseyXY([0], backend=backend)
        self.assertRoundTripSerializable(exp._transpiled_circuits())

    def test_cal_experiment_config(self):
        """Test FrequencyCal config roundtrips"""
        exp = FrequencyCal([0], self.cals)
        loaded_exp = FrequencyCal.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqualExtended(exp, loaded_exp)

    def test_residual_plot(self):
        """Test if plot is changing due to residual plotting."""
        freq_shift = 1e3

        exp_helper = RamseyXYHelper()
        ramsey = RamseyXY([0])
        ramsey.backend = MockIQBackend(exp_helper)

        exp_helper.freq_shift = freq_shift
        ramsey.analysis.set_options(plot_residuals=True)
        test_data = ramsey.run().block_for_results()
        test_data_figure_bounds = test_data.figure(0).figure.figbbox.bounds

        ramsey.analysis.set_options(plot_residuals=False)
        test_data2 = ramsey.run().block_for_results()
        test_data2_figure_bounds = test_data2.figure(0).figure.figbbox.bounds

        self.assertNotEqual(test_data_figure_bounds[3], test_data2_figure_bounds[3])

    @unittest.skip("Cal experiments are not yet JSON serializable")
    def test_freqcal_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = FrequencyCal([0], self.cals)
        self.assertRoundTripSerializable(exp)
