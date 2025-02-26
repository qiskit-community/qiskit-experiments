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

from test.base import QiskitExperimentsTestCase

from ddt import ddt, data, named_data
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeArmonkV2

from qiskit_experiments.library import RamseyXY
from qiskit_experiments.test.mock_iq_backend import MockIQBackend
from qiskit_experiments.test.mock_iq_helpers import MockIQRamseyXYHelper as RamseyXYHelper


@ddt
class TestRamseyXY(QiskitExperimentsTestCase):
    """Tests for the Ramsey XY experiment."""

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
