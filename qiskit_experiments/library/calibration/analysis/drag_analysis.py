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

"""DRAG pulse calibration experiment."""

from typing import Any, Dict, List, Union
import numpy as np

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.fit_function import cos


class DragCalAnalysis(curve.CurveAnalysis):
    r"""Drag calibration analysis based on a fit to a cosine function.

    # section: fit_model

        Analyse a Drag calibration experiment by fitting three series each to a cosine function.
        The three functions share the phase parameter (i.e. beta) but each have their own amplitude,
        baseline, and frequency parameters (which therefore depend on the number of repetitions of
        xp-xm). Several initial guesses are tried if the user does not provide one.

        .. math::

            y = {\rm amp} \cos\left(2 \pi\cdot {\rm freq}_i\cdot x - 2 \pi \beta\right) + {\rm base}

    # section: fit_parameters
        defpar \rm amp:
            desc: Amplitude of all series.
            init_guess: The maximum y value less the minimum y value. 0.5 is also tried.
            bounds: [-2, 2] scaled to the maximum signal value.

        defpar \rm base:
            desc: Base line of all series.
            init_guess: The average of the data. 0.5 is also tried.
            bounds: [-1, 1] scaled to the maximum signal value.

        defpar {\rm freq}_i:
            desc: Frequency of the :math:`i` th oscillation.
            init_guess: The frequency with the highest power spectral density.
            bounds: [0, inf].

        defpar \beta:
            desc: Common beta offset. This is the parameter of interest.
            init_guess: Linearly spaced between the maximum and minimum scanned beta.
            bounds: [-min scan range, max scan range].
    """

    __series__ = [
        curve.SeriesDef(
            fit_func=lambda x, amp, freq0, freq1, freq2, beta, base: cos(
                x, amp=amp, freq=freq0, phase=-2 * np.pi * freq0 * beta, baseline=base
            ),
            plot_color="blue",
            name="series-0",
            filter_kwargs={"series": 0},
            plot_symbol="o",
        ),
        curve.SeriesDef(
            fit_func=lambda x, amp, freq0, freq1, freq2, beta, base: cos(
                x, amp=amp, freq=freq1, phase=-2 * np.pi * freq1 * beta, baseline=base
            ),
            plot_color="green",
            name="series-1",
            filter_kwargs={"series": 1},
            plot_symbol="^",
        ),
        curve.SeriesDef(
            fit_func=lambda x, amp, freq0, freq1, freq2, beta, base: cos(
                x, amp=amp, freq=freq2, phase=-2 * np.pi * freq2 * beta, baseline=base
            ),
            plot_color="red",
            name="series-2",
            filter_kwargs={"series": 2},
            plot_symbol="v",
        ),
    ]

    @classmethod
    def _default_options(cls):
        """Return the default analysis options.

        See :meth:`~qiskit_experiment.curve_analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """
        default_options = super()._default_options()
        default_options.p0 = {
            "amp": None,
            "freq0": None,
            "freq1": None,
            "freq2": None,
            "beta": None,
            "base": None,
        }
        default_options.bounds = {
            "amp": None,
            "freq0": None,
            "freq1": None,
            "freq2": None,
            "beta": None,
            "base": None,
        }
        default_options.result_parameters = ["beta"]
        default_options.xlabel = "Beta"
        default_options.ylabel = "Signal (arb. units)"

        return default_options

    def _setup_fitting(self, **options) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Compute the initial guesses."""
        user_p0 = self._get_option("p0")
        user_bounds = self._get_option("bounds")

        # Use a fast Fourier transform to guess the frequency.
        x_data = self._data("series-0").x
        delta_beta = x_data[1] - x_data[0]

        min_beta, max_beta = min(x_data), max(x_data)

        freq_guess = []
        for series in ["series-0", "series-1", "series-2"]:
            y_data = self._data(series).y
            fft = np.abs(np.fft.fft(y_data - np.average(y_data)))
            freqs = np.linspace(0.0, 1.0 / (2.0 * delta_beta), len(fft))
            freq_guess.append(freqs[np.argmax(fft[0 : len(fft) // 2])])

        if user_p0.get("beta", None) is not None:
            p_guesses = [user_p0["beta"]]
        else:
            p_guesses = np.linspace(min_beta, max_beta, 20)

        user_amp = user_p0.get("amp", None)
        user_base = user_p0.get("base", None)

        # Drag curves can sometimes be very flat, i.e. averages of y-data
        # and min-max do not always make good initial guesses. We therefore add
        # 0.5 to the initial guesses.
        guesses = [(0.5, 0.5)]

        if user_amp is not None and user_base is not None:
            guesses.append((user_amp, user_base))

        max_abs_y = np.max(np.abs(self._data().y))

        freq_guess0 = user_p0.get("freq0", None) or freq_guess[0]
        freq_bound = max(10 / freq_guess0, max(x_data))

        fit_options = []
        for amp_guess, b_guess in guesses:
            for p_guess in p_guesses:
                fit_option = {
                    "p0": {
                        "amp": amp_guess,
                        "freq0": freq_guess0,
                        "freq1": user_p0.get("freq1", None) or freq_guess[1],
                        "freq2": user_p0.get("freq2", None) or freq_guess[2],
                        "beta": p_guess,
                        "base": b_guess,
                    },
                    "bounds": {
                        "amp": user_bounds.get("amp", None) or (-2 * max_abs_y, 2 * max_abs_y),
                        "freq0": user_bounds.get("freq0", None) or (0, np.inf),
                        "freq1": user_bounds.get("freq1", None) or (0, np.inf),
                        "freq2": user_bounds.get("freq2", None) or (0, np.inf),
                        "beta": user_bounds.get("beta", None) or (-freq_bound, freq_bound),
                        "base": user_bounds.get("base", None) or (-1 * max_abs_y, 1 * max_abs_y),
                    },
                }

                fit_option.update(options)
                fit_options.append(fit_option)

        return fit_options

    def _evaluate_quality(self, fit_data: curve.FitData) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three,
            - a DRAG parameter value within the first period of the lowest number of repetitions,
            - an error on the drag beta smaller than the beta.
        """
        fit_beta = fit_data.fitval("beta").value
        fit_beta_err = fit_data.fitval("beta").stderr
        fit_freq0 = fit_data.fitval("freq0").value

        criteria = [
            fit_data.reduced_chisq < 3,
            fit_beta < 1 / fit_freq0,
            fit_beta_err < abs(fit_beta),
        ]

        if all(criteria):
            return "good"

        return "bad"
