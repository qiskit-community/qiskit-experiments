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

"""Resonance analysis class."""

from typing import Any, Dict, List, Union

import numpy as np
import qiskit_experiments.curve_analysis as curve


class ResonanceAnalysis(curve.CurveAnalysis):
    r"""A class to analyze a resonance, typically seen as a peak.

    Overview
        This analysis takes only single series. This series is fit by the Gaussian function.

    Fit Model
        The fit is based on the following Gaussian function.

        .. math::

            F(x) = a \exp(-(x-f)^2/(2\sigma^2)) + b

    Fit Parameters
        - :math:`a`: Peak height.
        - :math:`b`: Base line.
        - :math:`f`: Center frequency. This is the fit parameter of main interest.
        - :math:`\sigma`: Standard deviation of Gaussian function.

    Initial Guesses
        - :math:`a`: Calculated by :func:`~qiskit_experiments.curve_analysis.guess.max_height`.
        - :math:`b`: Calculated by :func:`~qiskit_experiments.curve_analysis.guess.\
          constant_spectral_offset`.
        - :math:`f`: Frequency at max height position calculated by
          :func:`~qiskit_experiments.curve_analysis.guess.max_height`.
        - :math:`\sigma`: Calculated from FWHM of peak :math:`w`
          such that :math:`w / \sqrt{8} \ln{2}`, where FWHM is calculated by
          :func:`~qiskit_experiments.curve_analysis.guess.full_width_half_max`.

    Bounds
        - :math:`a`: [-2, 2] scaled with maximum signal value.
        - :math:`b`: [-1, 1] scaled with maximum signal value.
        - :math:`f`: [min(x), max(x)] of frequency scan range.
        - :math:`\sigma`: [0, :math:`\Delta x`] where :math:`\Delta x`
          represents frequency scan range.

    """

    __series__ = [
        curve.SeriesDef(
            fit_func=lambda x, a, sigma, freq, b: curve.fit_function.gaussian(
                x, amp=a, sigma=sigma, x0=freq, baseline=b
            ),
            plot_color="blue",
        )
    ]

    @classmethod
    def _default_options(cls):
        """Return default data processing options.

        See :meth:`~qiskit_experiment.curve_analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """
        default_options = super()._default_options()
        default_options.p0 = {"a": None, "sigma": None, "freq": None, "b": None}
        default_options.bounds = {"a": None, "sigma": None, "freq": None, "b": None}
        default_options.reporting_parameters = {"freq": ("frequency", "Hz")}
        default_options.normalization = True

        return default_options

    def _setup_fitting(self, **options) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Fitter options."""
        user_p0 = self._get_option("p0")
        user_bounds = self._get_option("bounds")

        curve_data = self._data()

        b_guess = curve.guess.constant_spectral_offset(curve_data.y)
        y_ = curve_data.y - b_guess

        _, peak_idx = curve.guess.max_height(y_, absolute=True)
        a_guess = curve_data.y[peak_idx] - b_guess
        f_guess = curve_data.x[peak_idx]
        s_guess = curve.guess.full_width_half_max(curve_data.x, y_, peak_idx) / np.sqrt(
            8 * np.log(2)
        )

        max_abs_y = np.max(np.abs(curve_data.y))

        fit_option = {
            "p0": {
                "a": user_p0["a"] or a_guess,
                "sigma": user_p0["sigma"] or s_guess,
                "freq": user_p0["freq"] or f_guess,
                "b": user_p0["b"] or b_guess,
            },
            "bounds": {
                "a": user_bounds["a"] or (-2 * max_abs_y, 2 * max_abs_y),
                "sigma": user_bounds["sigma"] or (0.0, max(curve_data.x) - min(curve_data.x)),
                "freq": user_bounds["freq"] or (min(curve_data.x), max(curve_data.x)),
                "b": user_bounds["b"] or (-max_abs_y, max_abs_y),
            },
        }
        fit_option.update(options)

        return fit_option

    def _evaluate_quality(self, fit_data: curve.FitData) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared less than 3,
            - a peak within the scanned frequency range,
            - a standard deviation that is not larger than the scanned frequency range,
            - a standard deviation that is wider than the smallest frequency increment,
            - a signal-to-noise ratio, defined as the amplitude of the peak divided by the
              square root of the median y-value less the fit offset, greater than a
              threshold of two, and
            - a standard error on the sigma of the Gaussian that is smaller than the sigma.
        """
        curve_data = self._data()

        max_freq = np.max(curve_data.x)
        min_freq = np.min(curve_data.x)
        freq_increment = np.mean(np.diff(curve_data.x))

        fit_a = fit_data.fitval("a").value
        fit_b = fit_data.fitval("b").value
        fit_freq = fit_data.fitval("freq").value
        fit_sigma = fit_data.fitval("sigma").value
        fit_sigma_err = fit_data.fitval("sigma").stderr

        snr = abs(fit_a) / np.sqrt(abs(np.median(curve_data.y) - fit_b))
        fit_width_ratio = fit_sigma / (max_freq - min_freq)

        criteria = [
            min_freq <= fit_freq <= max_freq,
            1.5 * freq_increment < fit_sigma,
            fit_width_ratio < 0.25,
            fit_data.reduced_chisq < 3,
            (fit_sigma_err is None or fit_sigma_err < fit_sigma),
            snr > 2,
        ]

        if all(criteria):
            return "good"

        return "bad"
