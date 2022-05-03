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
"""
Analysis classes for Stark tone experiments.
"""

from qiskit_experiments.framework import Options
import qiskit_experiments.curve_analysis as curve


class StarkRamseyAnalysis(curve.DumpedOscillationAnalysis):

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_options()
        options.curve_drawer.set_options(
            xlabel="Delay",
            ylabel="P(1)",
            xval_unit="s",
        )
        options.result_parameters = [
            curve.ParameterRepr("freq", "stark_shift", "Hz"),
        ]

        return options

    def _evaluate_quality(self, fit_data: curve.FitData) -> str:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three
            - relative error of freq is less than 10 percent
        """
        freq = fit_data.fitval("freq")

        criteria = [
            fit_data.reduced_chisq < 3,
            curve.is_error_not_significant(freq, fraction=0.1),
        ]

        if all(criteria):
            return "good"

        return "bad"
