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

from qiskit.qobj.utils import MeasLevel

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import Options, ExperimentData


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
            init_guess: Calculated by :func:`~qiskit_experiments.curve_analysis.\
                guess.sinusoidal_freq_offset_amp`. When frequency is too low,
                the difference of averages of Ramsey X and Y outcomes is set.
            bounds: [0, 2 * average y peak-to-peak]

        defpar \tau:
            desc: The exponential decay of the curve.
            init_guess: Determined by :py:func:`~qiskit_experiments.curve_analysis.\
                guess.exp_decay`. The square root of (X data)**2 + (Y data)**2 is used to
                estimate the decay. When frequency is too low, large value is set.
            bounds: [0, inf]

        defpar \rm base:
            desc: Base line of both series.
            init_guess: Calculated by :func:`~qiskit_experiments.curve_analysis.\
                guess.sinusoidal_freq_offset_amp`. When frequency is too low,
                the average of Ramsey Y outcomes is set.
            bounds: [min y - average y peak-to-peak, max y + average y peak-to-peak]

        defpar \rm freq:
            desc: Frequency of both series. This is the parameter of interest.
            init_guess: Calculated by :func:`~qiskit_experiments.curve_analysis.\
                guess.sinusoidal_freq_offset_amp`. When frequency is too low, zero is set.
            bounds: [-Nyquist frequency, Nyquist frequency] where Nyquist frequency is
                a half of maximum sampling frequency of the X values.

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
                    data_sort_key={"series": "X"},
                ),
                lmfit.models.ExpressionModel(
                    expr="amp * exp(-x / tau) * sin(2 * pi * freq * x + phase) + base",
                    name="Y",
                    data_sort_key={"series": "Y"},
                ),
            ]
        )

    @classmethod
    def _default_options(cls) -> Options:
        """Return the default analysis options.

        See :meth:`~qiskit_experiment.curve_analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """
        default_options = super()._default_options()
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
        nyquist_freq = 1 / np.min(np.diff(ramx_data.x)) / 2

        user_opt.bounds.set_if_empty(
            amp=(0, full_y_ptp * 2),
            tau=(0, np.inf),
            base=(min_y - avg_y_ptp, max_y + avg_y_ptp),
            phase=(-np.pi, np.pi),
            freq=(-nyquist_freq, nyquist_freq),
        )
        user_opt.p0.set_if_empty(phase=0.0)

        if avg_y_ptp < 0.5 * full_y_ptp:
            # When X and Y curve don't oscillate, X (Y) usually stays at P(1) = 1.0 (0.5).
            # So peak-to-peak of full data is something around P(1) = 0.75, while
            # single curve peak-to-peak is almost zero.
            # Tau guess is not applicable because we cannot distinguish the decay curve
            # from the sinusoidal oscillation.
            avg_x = np.average(ramx_data.y)
            avg_y = np.average(ramy_data.y)

            user_opt.p0.set_if_empty(
                amp=np.abs(avg_x - avg_y),
                tau=100 * np.max(curve_data.x),
                base=avg_y,
                freq=0.0,
            )
            return user_opt

        alpha = curve.guess.exp_decay(ramx_data.x, np.sqrt(ramx_data.y**2 + ramy_data.y**2))
        tau_guess = -1 / min(alpha, -1 / (100 * max(curve_data.x)))

        # Generate guess iterators for Ramsey X and Y and average the estimate
        pre_estimate = user_opt.p0.copy()
        del pre_estimate["tau"]

        ramx_guess_iter = curve.guess.composite_sinusoidal_estimate(
            x=ramx_data.x,
            y=ramx_data.y,
            **pre_estimate,
        )
        ramy_guess_iter = curve.guess.composite_sinusoidal_estimate(
            x=ramy_data.x,
            y=ramy_data.y,
            **pre_estimate,
        )

        options = []
        for ramx_guess, ramy_guess in zip(ramx_guess_iter, ramy_guess_iter):
            amp_x, freq_x, base_x, _ = ramx_guess
            amp_y, freq_y, base_y, _ = ramy_guess
            for sign in (-1, 1):
                # Ramsey XY is frequency sign sensitive.
                # Since experimental data is noisy, correct sign is hardly estimated with
                # phase velocity. Try both positive and negative frequency to find the best fit.
                tmp_opt = user_opt.copy()
                tmp_opt.p0.set_if_empty(
                    amp=0.5 * (amp_x + amp_y),
                    freq=sign * 0.5 * (freq_x + freq_y),
                    base=0.5 * (base_x + base_y),
                    tau=tau_guess,
                )
                options.append(tmp_opt)

        if len(options) == 0:
            return user_opt
        return options

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

    def _initialize(
        self,
        experiment_data: ExperimentData,
    ):
        super()._initialize(experiment_data)

        if experiment_data.metadata.get("meas_level", MeasLevel.CLASSIFIED) == MeasLevel.CLASSIFIED:
            init_guess = self.options.get("p0", {})
            bounds = self.options.get("bounds", {})

            init_guess["base"] = init_guess.get("base", 0.5)
            bounds["base"] = bounds.get("base", (0.0, 1.0))
            self.set_options(p0=init_guess, bounds=bounds)
