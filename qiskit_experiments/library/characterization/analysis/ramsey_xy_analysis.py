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
            init_guess: Half of the maximum y value less the minimum y value.
            bounds: [-2, 2] scaled to the maximum signal value.

        defpar \tau:
            desc: The exponential decay of the curve.
            init_guess: The initial guess is obtained by fitting an exponential to the
                square root of (X data)**2 + (Y data)**2.
            bounds: [0, inf].

        defpar \rm base:
            desc: Base line of both series.
            init_guess: Roughly the average of the data.
            bounds: [minimum Y data, maximum Y data]

        defpar \rm freq:
            desc: Frequency of both series. This is the parameter of interest.
            init_guess: The frequency with the highest power spectral density.
            bounds: [-inf, inf].

        defpar \rm phase:
            desc: Common phase offset.
            init_guess: 0
            bounds: [-pi, pi].
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
    def _default_options(cls):
        """Return the default analysis options.

        See :meth:`~qiskit_experiment.curve_analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """
        default_options = super()._default_options()
        default_options.curve_drawer.set_options(
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
        y_max = np.max(curve_data.y)
        y_min = np.min(curve_data.y)
        y_ptp = y_max - y_min

        user_opt.bounds.set_if_empty(
            amp=(0, y_ptp),
            tau=(0, np.inf),
            base=(y_min, y_max),
            phase=(-np.pi, np.pi),
        )

        # Default guess values
        freq_guesses, base_guesses = [], []
        for series in ["X", "Y"]:
            data = curve_data.get_subset_of(series)
            freq_guesses.append(curve.guess.frequency(data.x, data.y))
            base_guesses.append(curve.guess.constant_sinusoidal_offset(data.y))

        freq_val = float(np.average(freq_guesses))
        user_opt.p0.set_if_empty(base=np.average(base_guesses))

        # Guess the exponential decay by combining both curves
        data_x = curve_data.get_subset_of("X")
        data_y = curve_data.get_subset_of("Y")
        decay_data = (data_x.y - user_opt.p0["base"]) ** 2 + (data_y.y - user_opt.p0["base"]) ** 2

        user_opt.p0.set_if_empty(
            tau=-curve.guess.exp_decay(data_x.x, decay_data),
            amp=0.5 * y_ptp,
            phase=0.0,
        )

        opt_fp = user_opt.copy()
        opt_fp.p0.set_if_empty(freq=freq_val)

        opt_fm = user_opt.copy()
        opt_fm.p0.set_if_empty(freq=-freq_val)

        return [opt_fp, opt_fm]

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
