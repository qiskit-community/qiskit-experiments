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

"""Test restless fine amplitude characterization and calibration experiments."""
from test.base import QiskitExperimentsTestCase

import numpy as np
from ddt import ddt, data

from qiskit_experiments.library import (
    FineXAmplitude,
)

from qiskit_experiments.test.mock_iq_backend import MockRestlessFineAmp

from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.nodes import Probability, RestlessToCounts


@ddt
class TestFineAmpEndToEndRestless(QiskitExperimentsTestCase):
    """Test the fine amplitude experiment in a restless measurement setting."""

    @data(-0.03, -0.02, -0.01, 0.02, 0.04)
    def test_end_to_end_restless(self, pi_ratio):
        """Test the restless experiment end to end."""

        amp_exp = FineXAmplitude(0)
        # enable a restless measurement setting.
        amp_exp.enable_restless(rep_delay=1e-6)

        error = -np.pi * pi_ratio
        backend = MockRestlessFineAmp(error, np.pi, "x")

        expdata = amp_exp.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results(1)
        d_theta = result.value.n

        self.assertAlmostEqual(d_theta, error, delta=0.01)
        self.assertEqual(result.quality, "good")

        # check that the fit amplitude is almost 1 as expected.
        amp_fit = expdata.analysis_results(0).value[0]
        self.assertAlmostEqual(amp_fit, 1.0, delta=0.02)

    @data(-0.02, 0.03, 0.04)
    def test_end_to_end_restless_standard_processor(self, pi_ratio):
        """Test the restless experiment with a standard processor end to end."""

        amp_exp = FineXAmplitude(0)
        # standard data processor.
        standard_processor = DataProcessor("counts", [Probability("1")])
        amp_exp.analysis.set_options(data_processor=standard_processor)
        # set restless run options.
        amp_exp.set_run_options(rep_delay=1e-6, meas_level=2, memory=True, init_qubits=False)

        error = -np.pi * pi_ratio
        backend = MockRestlessFineAmp(error, np.pi, "x")

        expdata = amp_exp.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results(1)
        d_theta = result.value.n

        self.assertTrue(abs(d_theta - error) > 0.01)

        # check that the fit amplitude is much smaller than 1.
        amp_fit = expdata.analysis_results(0).value[0]
        self.assertTrue(amp_fit < 0.05)