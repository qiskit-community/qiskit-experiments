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
Analysis class for ZZ Ramsey experiment
"""

from typing import List, Tuple, Union

import lmfit
import numpy as np

from qiskit.providers.options import Options

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis import CurveAnalysis, CurveData, CurveFitResult, FitOptions
from qiskit_experiments.curve_analysis.utils import is_error_not_significant


class ZZRamseyAnalysis(CurveAnalysis):
    # Disable long line check because we can't break the long math lines
    # pylint: disable=line-too-long
    r"""A class to analyze a :math:`ZZ` Ramsey experiment.

    # section: fit_model

        Analyze a :math:`ZZ` Ramsey experiment by fitting the :code:`'0'` and
        :code:`'1'` series to sinusoidal functions as defined in the
        :class:`ZZRamsey` experiment. The two functions share the frequency,
        amplitude, decay constant, baseline, and phase parameters.

        .. math::

            y_0 = - {\rm amp} \cdot e^{-x/\tau} \cos\left(2 \pi\cdot {\rm freq - zz / 2}\cdot x + {\rm phase}\right) + {\rm base} \\

            y_1 = - {\rm amp} \cdot e^{-x/\tau} \cos\left(2 \pi\cdot {\rm freq + zz / 2}\cdot x + {\rm phase}\right) + {\rm base}

        :math:`freq` is the same as the virtual frequency :math:`f` mentioned
        in :class:`ZZRamsey`.

    # section: fit_parameters

        defpar \rm amp:
            desc: Amplitude of the sinusoidal curves.
            init_guess: Half of the maximum y value less the minimum y value.
            bounds: [0, the peak to peak range of the data]
        defpar \tau:
            desc: The exponential decay of the curve amplitudes.
            init_guess: Inferred by comparing the peak to peak amplitude for
                longer delay values with that of shorter delay values and
                assuming an exponential decay in amplitude.
            bounds: [1/4 of the typical time spacing,
                10 times the maximum delay time].
        defpar \rm base:
            desc: Base line of both series.
            init_guess: The average of the data, excluding outliers
            bounds: [the minimum amplitude less the peak to peak of the data,
                the maximum amplitude plus the peak to peak of the data]
        defpar \rm freq:
            desc: Average frequency of both series.
            init_guess: The average of the frequencies with the highest power
                spectral density for each series.
            bounds: [0, the Nyquist frequency of the data].
        defpar \rm zz:
            desc: The :math:`ZZ` value for the qubit pair. In terms of the fit,
                this is frequency difference between series 1 and series 0.
            init_guess: The difference between the frequencies with the highest
                power spectral density for each series
            bounds: [-inf, inf].
        defpar \rm phase:
            desc: Common phase offset.
            init_guess: Zero
            bounds: [-pi, pi].
    """
    # pylint: enable=line-too-long

    def __init__(self):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="-amp * exp(-x / tau) * cos(2 * pi * (freq - zz / 2) * x + phase) + base",
                    name="0",
                ),
                lmfit.models.ExpressionModel(
                    expr="-amp * exp(-x / tau) * cos(2 * pi * (freq + zz / 2) * x + phase) + base",
                    name="1",
                ),
            ]
        )

    @classmethod
    def _default_options(cls) -> Options:
        """Return the default analysis options.

        See
        :meth:`~qiskit_experiment.curve_analysis.CurveAnalysis._default_options`
        for descriptions of analysis options.
        """
        default_options = super()._default_options()
        default_options.result_parameters = ["zz"]
        default_options.plotter.set_figure_options(
            xlabel="Delay",
            xval_unit="s",
            ylabel="P(1)",
        )
        default_options.data_subfit_map = {
            "0": {"series": "0"},
            "1": {"series": "1"},
        }

        return default_options

    def _generate_fit_guesses(
        self,
        user_opt: FitOptions,
        curve_data: CurveData,
    ) -> Union[FitOptions, List[FitOptions]]:
        """Compute the initial guesses.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Preprocessed data to be fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        y_max = np.max(curve_data.y)
        y_min = np.min(curve_data.y)
        y_ptp = y_max - y_min
        x_max = np.max(curve_data.x)

        data_0 = curve_data.get_subset_of("0")
        data_1 = curve_data.get_subset_of("1")

        def typical_step(arr):
            """Find the typical step size of an array"""
            steps = np.diff(np.sort(arr))
            # If points are not unique, there will be 0's that don't count as
            # steps
            steps = steps[steps != 0]
            return np.median(steps)

        x_step = max(typical_step(data_0.x), typical_step(data_1.x))

        user_opt.bounds.set_if_empty(
            amp=(0, y_max - y_min),
            tau=(x_step / 4, 10 * x_max),
            base=(y_min - y_ptp, y_max + y_ptp),
            phase=(-np.pi, np.pi),
            freq=(0, 1 / 2 / x_step),
        )

        freq_guesses = [
            curve.guess.frequency(data_0.x, data_0.y),
            curve.guess.frequency(data_1.x, data_1.y),
        ]
        base_guesses = [
            curve.guess.constant_sinusoidal_offset(data_0.y),
            curve.guess.constant_sinusoidal_offset(data_1.y),
        ]

        def rough_sinusoidal_decay_constant(
            x_data: np.ndarray, y_data: np.ndarray, bounds: Tuple[float, float]
        ) -> float:
            """Estimate the decay constant of y_data vs x_data

            This function assumes the data is roughly evenly spaced and that
            the y_data goes through a few periods so that the peak to peak
            value early in the data can be compared to the peak to peak later
            in the data to estimate the decay constant.

            Args:
                x_data: x-axis data
                y_data: y-axis data
                bounds: minimum and maximum allowed decay constant

            Returns:
                The bounded guess of the decay constant
            """
            x_median = np.median(x_data)
            i_left = x_data < x_median
            i_right = x_data > x_median

            y_left = np.ptp(y_data[i_left])
            y_right = np.ptp(y_data[i_right])
            x_left = np.average(x_data[i_left])
            x_right = np.average(x_data[i_right])

            # Now solve y_left = exp(-x_left / tau) and
            # y_right = exp(-x_right / tau) for tau
            denom = np.log(y_right / y_left)
            if denom < 0:
                tau = (x_left - x_right) / denom
            else:
                # If amplitude is constant or growing from left to right, bound
                # to the maximum allowed tau
                tau = bounds[1]

            return max(min(tau, bounds[1]), bounds[0])

        user_opt.p0.set_if_empty(
            tau=rough_sinusoidal_decay_constant(curve_data.x, curve_data.y, user_opt.bounds["tau"]),
            amp=y_ptp / 2,
            phase=0.0,
            freq=float(np.average(freq_guesses)),
            base=np.average(base_guesses),
            zz=freq_guesses[1] - freq_guesses[0],
        )

        return user_opt

    def _evaluate_quality(self, fit_data: CurveFitResult) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three
            - an error on the frequency smaller than the frequency

        Args:
            fit_data: The fit result of the analysis

        Returns:
            The automated fit quality assessment as a string
        """
        freq = fit_data.ufloat_params["freq"]
        zz = fit_data.ufloat_params["zz"]
        amp = fit_data.ufloat_params["amp"]
        base = fit_data.ufloat_params["base"]

        rough_freq_magnitude = 1 / (fit_data.x_range[1] - fit_data.x_range[0])

        criteria = [
            is_error_not_significant(amp, fraction=0.2),
            is_error_not_significant(base, absolute=0.2 * amp.nominal_value),
            is_error_not_significant(freq, absolute=0.2 * rough_freq_magnitude),
            is_error_not_significant(zz, absolute=0.2 * rough_freq_magnitude),
        ]

        if all(criteria):
            return "good"

        return "bad"
