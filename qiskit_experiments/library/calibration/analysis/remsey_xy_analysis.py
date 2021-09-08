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

from typing import Any, Dict, List, Union
import numpy as np

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis import fit_function


class RamseyXYAnalysis(curve.CurveAnalysis):
    r"""The Ramsey XY analysis is based on a fit to a cosine function and a sine function.

    # section: fit_model

        Analyse a Ramsey XY experiment by fitting the X and Y series to a cosine and sine
        function, respectively. The two functions share the frequency and amplitude parameters
        (i.e. beta).

        .. math::

            y_X = {\rm amp}e^{x/\tau}\cos\left(2\pi\cdot{\rm freq}_i\cdot x-\pi/2\right)+{\rm base}
            y_Y = {\rm amp}e^{x/\tau}\cos\left(2\pi\cdot{\rm freq}_i\cdot x) + {\rm base}

    # section: fit_parameters
        defpar \rm amp:
            desc: Amplitude of both series.
            init_guess: The maximum y value less the minimum y value. 0.5 is also tried.
            bounds: [-2, 2] scaled to the maximum signal value.

        defpar \tau:
            desc: The exponential decay of the curve.
            init_guess: The initial guess is obtained by fitting an exponential to the
                square root of (X data)**2 + (Y data)**2.
            bounds: [0, inf].

        defpar \rm base:
            desc: Base line of both series.
            init_guess: The average of the data. 0.5 is also tried.
            bounds: [-1, 1] scaled to the maximum signal value.

        defpar \rm freq:
            desc: Frequency of both series. This is the parameter of interest.
            init_guess: The frequency with the highest power spectral density.
            bounds: [0, inf].

        defpar \rm phase:
            desc: Common phase offset.
            init_guess: Linearly spaced between the maximum and minimum scanned beta.
            bounds: [-min scan range, max scan range].
    """

    __series__ = [
        curve.SeriesDef(
            fit_func=lambda x, amp, tau, freq, base, phase: fit_function.cos_decay(
                x, amp=amp, tau=tau, freq=freq, phase=phase, baseline=base
            ),
            plot_color="blue",
            name="X",
            filter_kwargs={"series": "X"},
            plot_symbol="o",
            model_description=r"{\rm amp} e^{-x/\tau} \cos\left(2 \pi\cdot {\rm freq}\cdot x "
            r"+ {\rm phase}) + {\rm base}",
        ),
        curve.SeriesDef(
            fit_func=lambda x, amp, tau, freq, base, phase: fit_function.cos_decay(
                x, amp=amp, tau=tau, freq=freq, phase=phase - np.pi / 2, baseline=base
            ),
            plot_color="green",
            name="Y",
            filter_kwargs={"series": "Y"},
            plot_symbol="^",
            model_description=r"{\rm amp} e^{-x/\tau} \cos\left(2 \pi\cdot {\rm freq}\cdot x "
            r"+{\rm phase} - \pi/2\right) + {\rm base}",
        ),
    ]

    @classmethod
    def _default_options(cls):
        """Return the default analysis options.

        See :meth:`~qiskit_experiment.curve_analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """
        default_options = super()._default_options()
        default_options.result_parameters = ["freq"]
        default_options.xlabel = "Delay (s)"
        default_options.ylabel = "Signal (arb. units)"

        return default_options

    def _setup_fitting(self, **extra_options) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Compute the initial guesses."""
        user_p0 = self._get_option("p0")

        # Default guess values
        freq_guesses, offset_guesses = [], []

        for series in ["X", "Y"]:
            data = self._data(series)
            freq_guesses.append(curve.guess.frequency(data.x, data.y))
            offset_guesses.append(curve.guess.constant_sinusoidal_offset(data.y))

        # Guess the exponential decay by combining both curves
        data_x = self._data("X")
        data_y = self._data("Y")
        decay_data = (data_x.y - offset_guesses[0]) ** 2 + (data_y.y - offset_guesses[1]) ** 2
        guess_decay = -curve.guess.exp_decay(data_x.x, decay_data)

        freq_guess = user_p0.get("freq", None) or np.average(freq_guesses)
        fit_options = []

        for freq in [-freq_guess, freq_guess]:
            fit_options.append(
                {
                    "p0": {
                        "amp": user_p0.get("amp", None) or 0.5,
                        "tau": guess_decay,
                        "freq": freq,
                        "base": user_p0.get("base", None) or np.average(offset_guesses),
                        "phase": user_p0.get("phase", None) or 0.0,
                    },
                }
            )

        return fit_options

    def _evaluate_quality(self, fit_data: curve.FitData) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three,
            - an error on the frequency smaller than the frequency.
        """
        fit_freq = fit_data.fitval("freq").value
        fit_freq_err = fit_data.fitval("freq").stderr

        criteria = [
            fit_data.reduced_chisq < 3,
            fit_freq_err < abs(fit_freq),
        ]

        if all(criteria):
            return "good"

        return "bad"
