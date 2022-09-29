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
"""
T2 Hahn echo Analysis class.
"""
from typing import Union

import numpy as np

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.data_processing import DataProcessor, Probability

from qiskit_experiments.framework import Options


class T2HahnAnalysis(curve.DecayAnalysis):
    r"""A class to analyze T2Hahn experiments.

    # section: see_also
        qiskit_experiments.curve_analysis.standard_analysis.decay.DecayAnalysis

    """

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_options()
        options.curve_drawer.set_options(
            xlabel="Delay",
            ylabel="P(0)",
            xval_unit="s",
        )
        options.data_processor = DataProcessor(
            input_key="counts", data_actions=[Probability(outcome="0")]
        )
        options.bounds = {
            "amp": (0.0, 1.0),
            "tau": (0.0, np.inf),
            "base": (0.0, 1.0),
        }
        options.result_parameters = [curve.ParameterRepr("tau", "T2", "s")]

        return options

    def _evaluate_quality(self, fit_data: curve.CurveFitResult) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three
            - absolute amp is within [0.4, 0.6]
            - base is less is within [0.4, 0.6]
            - amp error is less than 0.1
            - tau error is less than its value
            - base error is less than 0.1
        """
        amp = fit_data.ufloat_params["amp"]
        tau = fit_data.ufloat_params["tau"]
        base = fit_data.ufloat_params["base"]

        criteria = [
            fit_data.reduced_chisq < 3,
            abs(amp.nominal_value - 0.5) < 0.1,
            abs(base.nominal_value - 0.5) < 0.1,
            curve.utils.is_error_not_significant(amp, absolute=0.1),
            curve.utils.is_error_not_significant(tau),
            curve.utils.is_error_not_significant(base, absolute=0.1),
        ]

        if all(criteria):
            return "good"

        return "bad"
