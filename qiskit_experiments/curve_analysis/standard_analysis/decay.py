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

from typing import List, Union

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

    __series__ = [
        curve.SeriesDef(
            fit_func=lambda x, amp, base, tau: curve.fit_function.exponential_decay(
                x,
                amp=amp,
                lamb=1 / tau,
                baseline=base,
            ),
            plot_color="blue",
            model_description=r"amp \exp(-x/tau) + base",
            plot_fit_uncertainty=True,
        )
    ]

    def _generate_fit_guesses(
        self, user_opt: curve.FitOptions
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Compute the initial guesses.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        curve_data = self._data()

        user_opt.p0.set_if_empty(base=curve.guess.min_height(curve_data.y)[0])

        user_opt.p0.set_if_empty(
            tau=-1 / curve.guess.exp_decay(curve_data.x, curve_data.y),
            amp=curve.guess.max_height(curve_data.y)[0] - user_opt.p0["base"],
        )

        return user_opt

    def _evaluate_quality(self, fit_data: curve.FitData) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three
            - absolute amp is within [0.9, 1.1]
            - base is less than 0.1
            - amp error is less than 0.1
            - tau error is less than its value
            - base error is less than 0.1
        """
        amp = fit_data.fitval("amp")
        tau = fit_data.fitval("tau")
        base = fit_data.fitval("base")

        criteria = [
            fit_data.reduced_chisq < 3,
            abs(amp.value - 1.0) < 0.1,
            abs(base.value) < 0.1,
            amp.stderr is None or amp.stderr < 0.1,
            tau.stderr is None or tau.stderr < tau.value,
            base.stderr is None or base.stderr < 0.1,
        ]

        if all(criteria):
            return "good"

        return "bad"
