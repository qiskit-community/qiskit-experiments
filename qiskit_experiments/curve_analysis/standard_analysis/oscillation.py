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

"""Analyze oscillating data such as a Rabi amplitude experiment."""

from typing import List, Union, Optional

import lmfit
import numpy as np

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.warnings import deprecated_class


class OscillationAnalysis(curve.CurveAnalysis):
    r"""Oscillation analysis class based on a fit of the data to a cosine function.

    # section: fit_model

        Analyse oscillating data by fitting it to a cosine function

        .. math::

            y = {\rm amp} \cos\left(2 \pi\cdot {\rm freq}\cdot x + {\rm phase}\right) + {\rm base}

    # section: fit_parameters
        defpar \rm amp:
            desc: Amplitude of the oscillation.
            init_guess: Calculated by :func:`~qiskit_experiments.curve_analysis.\
            guess.sinusoidal_freq_offset_amp`.
            bounds: [-2 * maximum Y, 2 * maximum Y].

        defpar \rm base:
            desc: Base line.
            init_guess: Calculated by :func:`~qiskit_experiments.curve_analysis.\
            guess.sinusoidal_freq_offset_amp`.
            bounds: [-maximum Y, maximum Y].

        defpar \rm freq:
            desc: Frequency of the oscillation. This is the fit parameter of interest.
            init_guess: Calculated by :func:`~qiskit_experiments.curve_analysis.\
            guess.sinusoidal_freq_offset_amp`.
            bounds: [0, inf].

        defpar \rm phase:
            desc: Phase of the oscillation.
            init_guess: Multiple points between [-pi, pi].
            bounds: [-pi, pi].
    """

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="amp * cos(2 * pi * freq * x + phase) + base",
                    name="cos",
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
        max_abs_y, _ = curve.guess.max_height(curve_data.y, absolute=True)

        user_opt.bounds.set_if_empty(
            amp=(-2 * max_abs_y, 2 * max_abs_y),
            freq=(0, np.inf),
            phase=(-np.pi, np.pi),
            base=(-max_abs_y, max_abs_y),
        )

        options = []
        for delay_ratio in (0.1, 0.2, 0.3):
            for phase_guess in np.linspace(-np.pi, np.pi, 5):
                tmp_opt = user_opt.copy()
                freq, base, amp = curve.guess.sinusoidal_freq_offset_amp(
                    x=curve_data.x,
                    y=curve_data.y,
                    delay=int(delay_ratio * len(curve_data.x)),
                )
                tmp_opt.p0.set_if_empty(freq=freq, base=base, amp=amp, phase=phase_guess)
                options.append(tmp_opt)

        return options

    def _evaluate_quality(self, fit_data: curve.CurveFitResult) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three,
            - more than a quarter of a full period,
            - less than 10 full periods, and
            - an error on the fit frequency lower than the fit frequency.
        """
        fit_freq = fit_data.ufloat_params["freq"]

        criteria = [
            fit_data.reduced_chisq < 3,
            1.0 / 4.0 < fit_freq.nominal_value < 10.0,
            curve.utils.is_error_not_significant(fit_freq),
        ]

        if all(criteria):
            return "good"

        return "bad"


class DampedOscillationAnalysis(curve.CurveAnalysis):
    r"""A class to analyze general exponential decay curve with sinusoidal oscillation.

    # section: fit_model
        This class is based on the fit model of sinusoidal signal with exponential decay.
        This model is often used for the oscillation signal in the dissipative system.

        .. math::

            F(x) = {\rm amp} \cdot e^{-x/\tau}
                \cos(2\pi \cdot {\rm freq} \cdot t + \phi) + {\rm base}

    # section: fit_parameters

        defpar \rm amp:
            desc: Amplitude of the oscillation.
            init_guess: Calculated by :func:`~qiskit_experiments.curve_analysis.\
            guess.sinusoidal_freq_offset_amp`.
            bounds: [-2 * maximum Y, 2 * maximum Y].

        defpar \rm base:
            desc: Base line.
            init_guess: Calculated by :func:`~qiskit_experiments.curve_analysis.\
            guess.sinusoidal_freq_offset_amp`.
            bounds: [-maximum Y, maximum Y].

        defpar \rm freq:
            desc: Frequency of the oscillation. This is the fit parameter of interest.
            init_guess: Calculated by :func:`~qiskit_experiments.curve_analysis.\
            guess.sinusoidal_freq_offset_amp`.
            bounds: [0, inf].

        defpar \tau:
            desc: Represents the rate of decay.
            init_guess: Determined by :py:func:`~qiskit_experiments.curve_analysis.\
                guess.exp_decay`
            bounds: [0, inf]

        defpar \rm phase:
            desc: Phase of the oscillation.
            init_guess: Multiple points between [-pi, pi].
            bounds: [-pi, pi].
    """

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="amp * exp(-x / tau) * cos(2 * pi * freq * x + phi) + base",
                    name="cos_decay",
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
        max_abs_y, _ = curve.guess.max_height(curve_data.y, absolute=True)

        user_opt.bounds.set_if_empty(
            amp=(-2 * max_abs_y, 2 * max_abs_y),
            freq=(0, np.inf),
            tau=(0, np.inf),
            phase=(-np.pi, np.pi),
            base=(-max_abs_y, max_abs_y),
        )
        alpha = curve.guess.exp_decay(curve_data.x, curve_data.y)
        tau_guess = -1 / min(alpha, -1 / (100 * max(curve_data.x)))

        options = []
        for delay_ratio in (0.1, 0.2, 0.3):
            for phase_guess in np.linspace(0, np.pi, 5):
                tmp_opt = user_opt.copy()
                freq, base, amp = curve.guess.sinusoidal_freq_offset_amp(
                    x=curve_data.x,
                    y=curve_data.y,
                    delay=int(delay_ratio * len(curve_data.x)),
                )
                tmp_opt.p0.set_if_empty(
                    freq=freq,
                    base=base,
                    amp=amp,
                    phase=phase_guess,
                    tau=tau_guess,
                )
                options.append(tmp_opt)

        return options

    def _evaluate_quality(self, fit_data: curve.CurveFitResult) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three
            - relative error of tau is less than its value
            - relative error of freq is less than its value
        """
        tau = fit_data.ufloat_params["tau"]
        freq = fit_data.ufloat_params["freq"]

        criteria = [
            fit_data.reduced_chisq < 3,
            curve.utils.is_error_not_significant(tau),
            curve.utils.is_error_not_significant(freq),
        ]

        if all(criteria):
            return "good"

        return "bad"


@deprecated_class("0.5", new_cls=DampedOscillationAnalysis)
class DumpedOscillationAnalysis:
    """Deprecated."""

    pass
