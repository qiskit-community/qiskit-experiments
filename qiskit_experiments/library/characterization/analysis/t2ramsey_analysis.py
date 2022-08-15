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
T2Ramsey Experiment class.
"""
from typing import Union
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import Options


class T2RamseyAnalysis(curve.DampedOscillationAnalysis):
    """T2 Ramsey result analysis class.

    # section: see_also
        qiskit_experiments.curve_analysis.standard_analysis.oscillation.DampedOscillationAnalysis

    """

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
            curve.ParameterRepr("freq", "Frequency", "Hz"),
            curve.ParameterRepr("tau", "T2star", "s"),
        ]

        return options

    def _evaluate_quality(self, fit_data: curve.CurveFitResult) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three
            - relative error of amp is less than 10 percent
            - relative error of tau is less than 10 percent
            - relative error of freq is less than 10 percent
        """
        amp = fit_data.ufloat_params["amp"]
        tau = fit_data.ufloat_params["tau"]
        freq = fit_data.ufloat_params["freq"]

        criteria = [
            fit_data.reduced_chisq < 3,
            curve.utils.is_error_not_significant(amp, fraction=0.1),
            curve.utils.is_error_not_significant(tau, fraction=0.1),
            curve.utils.is_error_not_significant(freq, fraction=0.1),
        ]

        if all(criteria):
            return "good"

        return "bad"
