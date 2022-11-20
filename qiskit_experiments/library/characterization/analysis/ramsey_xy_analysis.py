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

"""The analysis class for the Ramsey XY experiment."""

from typing import List, Union

import lmfit
import numpy as np

import qiskit_experiments.curve_analysis as curve


class RamseyXYAnalysis(curve.CurveAnalysis):
    r"""Ramsey XY analysis based on a fit to a cosine function and a sine function.

    # section: fit_model

        Analyze a Ramsey XY experiment by fitting the X and Y series to a cosine and sine
        function, respectively. The two functions share the frequency and amplitude parameters.

        .. math::

            y_X = {\rm amp}e^{-x/\tau}\cos\left(2\pi\cdot{\rm freq}_i\cdot x\right) + {\rm base} \\
            y_Y = {\rm amp}e^{-x/\tau}\sin\left(2\pi\cdot{\rm freq}_i\cdot x\right) + {\rm base}

    # section: fit_parameters
        defpar \rm amp:
            desc: Amplitude of both series.
            init_guess: Half of the maximum y value less the minimum y value. When the
                oscillation frequency is low, it uses an averaged difference of
                Ramsey X data - Ramsey Y data.
            bounds: [0, 2 * average y peak-to-peak]

        defpar \tau:
            desc: The exponential decay of the curve.
            init_guess: The initial guess is obtained by fitting an exponential to the
                square root of (X data)**2 + (Y data)**2.
            bounds: [0, inf]

        defpar \rm base:
            desc: Base line of both series.
            init_guess: Roughly the average of the data. When the oscillation frequency is low,
                it uses an averaged data of Ramsey Y experiment.
            bounds: [min y - average y peak-to-peak, max y + average y peak-to-peak]

        defpar \rm freq:
            desc: Frequency of both series. This is the parameter of interest.
            init_guess: The frequency with the highest power spectral density.
            bounds: [-inf, inf]

        defpar \rm phase:
            desc: Common phase offset.
            init_guess: 0
            bounds: [-pi, pi]
    """

    def __init__(self):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="amp * exp(-x / tau) * cos(2 * pi * freq * x + phase) + base",
                    name="X",
                ),
                lmfit.models.ExpressionModel(
                    expr="amp * exp(-x / tau) * sin(2 * pi * freq * x + phase) + base",
                    name="Y",
                ),
            ]
        )

    @classmethod
    def _default_options(cls):
        """Return the default analysis options.

        See :meth:`~qiskit_experiment.curve_analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """
        default_options = super()._default_options()
        default_options.data_subfit_map = {
            "X": {"series": "X"},
            "Y": {"series": "Y"},
        }
        default_options.plotter.set_figure_options(
            xlabel="Delay",
            ylabel="Signal (arb. units)",
            xval_unit="s",
        )
        default_options.result_parameters = ["freq"]

        return default_options

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
        ramx_data = curve_data.get_subset_of("X")
        ramy_data = curve_data.get_subset_of("Y")

        # At very low frequency, y value of X (Y) curve stay at P=1.0 (0.5) for all x values.
        # Computing y peak-to-peak with combined data gives fake amplitude of 0.25.
        # Same for base, i.e. P=0.75 is often estimated in this case.
        full_y_ptp = np.ptp(curve_data.y)
        avg_y_ptp = 0.5 * (np.ptp(ramx_data.y) + np.ptp(ramy_data.y))
        max_y = np.max(curve_data.y)
        min_y = np.min(curve_data.y)

        user_opt.bounds.set_if_empty(
            amp=(0, full_y_ptp * 2),
            tau=(0, np.inf),
            base=(min_y - avg_y_ptp, max_y + avg_y_ptp),
            phase=(-np.pi, np.pi),
        )

        if avg_y_ptp < 0.5 * full_y_ptp:
            # When X and Y curve don't oscillate, X (Y) usually stays at P(1) = 1.0 (0.5).
            # So peak-to-peak of full data is something around P(1) = 0.75, while
            # single curve peak-to-peak is almost zero.
            avg_x = np.average(ramx_data.y)
            avg_y = np.average(ramy_data.y)

            user_opt.p0.set_if_empty(
                amp=np.abs(avg_x - avg_y),
                tau=100 * np.max(curve_data.x),
                base=avg_y,
                phase=0.0,
                freq=0.0,
            )
            return user_opt

        base_guess_x = curve.guess.constant_sinusoidal_offset(ramx_data.y)
        base_guess_y = curve.guess.constant_sinusoidal_offset(ramy_data.y)
        base_guess = 0.5 * (base_guess_x + base_guess_y)
        user_opt.p0.set_if_empty(
            amp=0.5 * full_y_ptp,
            base=base_guess,
            phase=0.0,
        )

        # Guess the exponential decay by combining both curves
        ramx_unbiased = ramx_data.y - user_opt.p0["base"]
        ramy_unbiased = ramy_data.y - user_opt.p0["base"]
        decay_data = ramx_unbiased**2 + ramy_unbiased**2
        if np.ptp(decay_data) < 0.95 * 0.5 * full_y_ptp:
            # When decay is less than 95 % of peak-to-peak value, ignore decay and
            # set large enough tau value compared with the measured x range.
            user_opt.p0.set_if_empty(tau=1000 * np.max(curve_data.x))
        else:
            user_opt.p0.set_if_empty(tau=-1 / curve.guess.exp_decay(ramx_data.x, decay_data))

        # Guess the oscillation frequency, remove offset to eliminate DC peak
        freq_guess_x = curve.guess.frequency(ramx_data.x, ramx_unbiased)
        freq_guess_y = curve.guess.frequency(ramy_data.x, ramy_unbiased)
        freq_val = 0.5 * (freq_guess_x + freq_guess_y)

        # FFT might be up to 1/2 bin off
        df = 2 * np.pi / (np.min(np.diff(ramx_data.x)) * ramx_data.x.size)
        freq_guesses = [freq_val - df, freq_val + df, freq_val]

        # Ramsey XY is frequency sign sensitive.
        # Since experimental data is noisy, correct sign is hardly estimated with phase velocity.
        # Try both positive and negative frequency to find the best fit.
        opts = []
        for sign in (1, -1):
            for freq_guess in freq_guesses:
                opt = user_opt.copy()
                opt.p0.set_if_empty(freq=sign * freq_guess)
                opts.append(opt)

        return opts

    def _evaluate_quality(self, fit_data: curve.CurveFitResult) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three,
            - an error on the frequency smaller than the frequency.
        """
        fit_freq = fit_data.ufloat_params["freq"]

        criteria = [
            fit_data.reduced_chisq < 3,
            curve.utils.is_error_not_significant(fit_freq),
        ]

        if all(criteria):
            return "good"

        return "bad"
