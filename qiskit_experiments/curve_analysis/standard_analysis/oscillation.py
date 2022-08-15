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
            init_guess: Calculated by :func:`~qiskit_experiments.curve_analysis.guess.max_height`.
            bounds: [-2, 2] scaled to the maximum signal value.

        defpar \rm base:
            desc: Base line.
            init_guess: Calculated by :func:`~qiskit_experiments.curve_analysis.\
            guess.constant_sinusoidal_offset`.
            bounds: [-1, 1] scaled to the maximum signal value.

        defpar \rm freq:
            desc: Frequency of the oscillation. This is the fit parameter of interest.
            init_guess: Calculated by :func:`~qiskit_experiments.curve_analysis.\
            guess.frequency`.
            bounds: [0, inf].

        defpar \rm phase:
            desc: Phase of the oscillation.
            init_guess: Zero.
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
        user_opt.p0.set_if_empty(
            freq=curve.guess.frequency(curve_data.x, curve_data.y),
            base=curve.guess.constant_sinusoidal_offset(curve_data.y),
        )
        user_opt.p0.set_if_empty(
            amp=curve.guess.max_height(curve_data.y - user_opt.p0["base"], absolute=True)[0],
        )

        options = []
        for phase_guess in np.linspace(0, np.pi, 5):
            new_opt = user_opt.copy()
            new_opt.p0.set_if_empty(phase=phase_guess)
            options.append(new_opt)

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
            desc: Amplitude. Height of the decay curve.
            init_guess: 0.5
            bounds: [0, 1.5],

        defpar \rm base:
            desc: Offset. Base line of the decay curve.
            init_guess: Determined by :py:func:`~qiskit_experiments.curve_analysis.\
                guess.constant_sinusoidal_offset`
            bounds: [0, 1.5]

        defpar \tau:
            desc: Represents the rate of decay.
            init_guess: Determined by :py:func:`~qiskit_experiments.curve_analysis.\
                guess.oscillation_exp_decay`
            bounds: [0, None]

        defpar \rm freq:
            desc: Oscillation frequency.
            init_guess: Determined by :py:func:`~qiskit_experiments.curve_analysis.guess.frequency`
            bounds: [0, 10 freq]

        defpar \phi:
            desc: Phase. Relative shift of the sinusoidal function from the origin.
            init_guess: Set multiple guesses within [-pi, pi]
            bounds: [-pi, pi]
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
        user_opt.p0.set_if_empty(
            amp=0.5,
            base=curve.guess.constant_sinusoidal_offset(curve_data.y),
        )

        # frequency resolution of this scan
        df = 1 / ((curve_data.x[1] - curve_data.x[0]) * len(curve_data.x))

        if user_opt.p0["freq"] is not None:
            # If freq guess is provided
            freq_guess = user_opt.p0["freq"]

            freqs = [freq_guess]
        else:
            freq_guess = curve.guess.frequency(curve_data.x, curve_data.y - user_opt.p0["base"])

            # The FFT might be up to 1/2 bin off
            if freq_guess > df:
                freqs = [freq_guess - df, freq_guess, freq_guess + df]
            else:
                freqs = [0.0, freq_guess]

        # Set guess for decay parameter based on estimated frequency
        if freq_guess > df:
            alpha = curve.guess.oscillation_exp_decay(
                curve_data.x, curve_data.y - user_opt.p0["base"], freq_guess=freq_guess
            )
        else:
            # Very low frequency. Assume standard exponential decay
            alpha = curve.guess.exp_decay(curve_data.x, curve_data.y)

        if alpha != 0.0:
            user_opt.p0.set_if_empty(tau=-1 / alpha)
        else:
            # Likely there is no slope. Cannot fit constant line with this model.
            # Set some large enough number against to the scan range.
            user_opt.p0.set_if_empty(tau=100 * np.max(curve_data.x))

        user_opt.bounds.set_if_empty(
            amp=[0, 1.5],
            base=[0, 1.5],
            tau=(0, np.inf),
            freq=(0, 10 * freq_guess),
            phi=(-np.pi, np.pi),
        )

        # more robust estimation
        options = []
        for freq in freqs:
            for phi in np.linspace(-np.pi, np.pi, 5)[:-1]:
                new_opt = user_opt.copy()
                new_opt.p0.set_if_empty(freq=freq, phi=phi)
                options.append(new_opt)

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
