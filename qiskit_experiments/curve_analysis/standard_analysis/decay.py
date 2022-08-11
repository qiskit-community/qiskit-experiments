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
"""Decay analysis class."""

from typing import List, Union, Optional

import lmfit
import numpy as np

import qiskit_experiments.curve_analysis as curve


class DecayAnalysis(curve.CurveAnalysis):
    r"""A class to analyze general exponential decay curve.

    # section: fit_model

    The fit is based on the following decay function.

    .. math::
        F(x) = {\rm amp} \cdot e^{-x/\tau} + {\rm base}

    # section: fit_parameters

        defpar \rm amp:
           desc: Height of the decay curve.
           init_guess: Determined by :py:func:`~qiskit_experiments.curve_analysis.guess.min_height`.
           bounds: None

        defpar \rm base:
           desc: Base line of the decay curve.
           init_guess: Determined by the difference of minimum and maximum points.
           bounds: None

        defpar \tau:
           desc: This is the fit parameter of main interest.
           init_guess: Determined by :py:func:`~qiskit_experiments.curve_analysis.guess.exp_decay`.
           bounds: None

    """

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="amp * exp(-x/tau) + base",
                    name="exp_decay",
                )
            ],
            name=name,
        )

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Create algorithmic initial fit guess from analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        user_opt.p0.set_if_empty(base=curve.guess.min_height(curve_data.y)[0])

        alpha = curve.guess.exp_decay(curve_data.x, curve_data.y)

        if alpha != 0.0:
            user_opt.p0.set_if_empty(
                tau=-1 / alpha,
                amp=curve.guess.max_height(curve_data.y)[0] - user_opt.p0["base"],
            )
        else:
            # Likely there is no slope. Cannot fit constant line with this model.
            # Set some large enough number against to the scan range.
            user_opt.p0.set_if_empty(
                tau=100 * np.max(curve_data.x),
                amp=curve.guess.max_height(curve_data.y)[0] - user_opt.p0["base"],
            )
        return user_opt

    def _evaluate_quality(self, fit_data: curve.CurveFitResult) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three
            - tau error is less than its value
        """
        tau = fit_data.ufloat_params["tau"]

        criteria = [
            fit_data.reduced_chisq < 3,
            curve.utils.is_error_not_significant(tau),
        ]

        if all(criteria):
            return "good"

        return "bad"
