# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test standard analysis base classes."""

from test.base import QiskitExperimentsTestCase
import numpy as np
from ddt import ddt, data

from qiskit_experiments.curve_analysis import ErrorAmplificationAnalysis
from qiskit_experiments.data_processing import DataProcessor
from qiskit_experiments.data_processing.nodes import Probability
from qiskit_experiments.framework import Options, ExperimentData


@ddt
class TestErrorAmplificationAnalysis(QiskitExperimentsTestCase):
    """Test for error amplification analysis."""

    @staticmethod
    def create_data(xvals, d_theta, shots=1000, noise_seed=123):
        """Create experiment data for testing."""
        np.random.seed(noise_seed)
        noise = np.random.normal(0, 0.03, len(xvals))
        yvals = 0.5 * np.cos((d_theta + np.pi) * xvals - np.pi / 2) + 0.5 + noise

        results = []
        for x, y in zip(xvals, yvals):
            n1 = int(shots * y)
            results.append({"counts": {"0": shots - n1, "1": n1}, "metadata": {"xval": x}})

        expdata = ExperimentData()
        expdata.add_data(results)

        return expdata

    @data(-0.1, -0.08, -0.06, -0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1)
    def test_fit_vals(self, d_theta_targ):
        """Test for fitting."""

        class FakeAmpAnalysis(ErrorAmplificationAnalysis):
            """Analysis class for testing."""

            @classmethod
            def _default_options(cls) -> Options:
                """Default analysis options."""
                options = super()._default_options()
                options.fixed_parameters = {
                    "angle_per_gate": np.pi,
                    "phase_offset": np.pi / 2,
                    "amp": 1.0,
                }

                return options

        reps = np.arange(0, 21, 1)
        fake_data = self.create_data(reps, d_theta=d_theta_targ)
        processor = DataProcessor("counts", data_actions=[Probability("1")])

        analysis = FakeAmpAnalysis()
        analysis.set_options(data_processor=processor)

        fake_data = analysis.run(fake_data)
        self.assertExperimentDone(fake_data)

        self.assertAlmostEqual(
            fake_data.analysis_results("d_theta", dataframe=True).iloc[0].value.n,
            d_theta_targ,
            delta=0.01,
        )
