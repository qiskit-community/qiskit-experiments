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

"""Fine amplitude calibration analysis."""

from typing import Any, Dict, List, Union
import numpy as np

from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.analysis import (
    CurveAnalysis,
    CurveAnalysisResult,
    SeriesDef,
    fit_function,
    get_opt_value,
)


class FineAmplitudeAnalysis(CurveAnalysis):
    r"""Fine amplitude analysis class based on a fit to a cosine function.

    Analyse a fine amplitude calibration experiment by fitting the data to a cosine function.
    The user must also specify the intended rotation angle per gate, here labeled,
    :math:`{\rm apg}`. The parameter of interest in the
    fit is the deviation from the intended rotation angle per gate labeled :math:`{\rm d}\theta`.
    The fit function is

    .. math::
        y = {\rm amp}/2\cos\left(x[{\rm d}\theta + {\rm apg} ]+{\rm phase\_offset}\right)+baseline

    Fit Parameters
        - :math:`amp`: Amplitude of the oscillation.
        - :math:`baseline`: Base line.
        - :math:`{\rm d}\theta`: The angle offset in the gate that we wish to measure.

    Initial Guesses
        - :math:`amp`: The maximum y value less the minimum y value.
        - :math:`baseline`: The average of the data.
        - :math:`{\rm d}\theta`: Zero.

    Bounds
        - :math:`amp`: [-2, 2] scaled to the maximum signal value.
        - :math:`baseline`: [-1, 1] scaled to the maximum signal value.
        - :math:`{\rm d}\theta`: [-pi, pi].

    Fixed-value parameters:
        - :math:`{\rm apg}` The angle per gate is set by the user, for example pi for a pi-pulse.
        - :math:`{\rm phase\_offset}` The phase offset in the cosine oscillation, for example,
          :math:`\pi/2` if a square-root of X gate is added before the repeated gates.
    """

    __series__ = [
        SeriesDef(
            fit_func=lambda x, amp, d_theta, phase_offset, baseline, angle_per_gate: fit_function.cos(
                x,
                amp=0.5 * amp,
                freq=(d_theta + angle_per_gate) / (2 * np.pi),
                phase=phase_offset,
                baseline=baseline,
            ),
            plot_color="blue",
        )
    ]

    # The intended angle per gat of the gate being calibrated, e.g. pi for a pi-pulse.
    __fixed_parameters__ = ["angle_per_gate", "phase_offset"]

    @classmethod
    def _default_options(cls):
        """Return the default analysis options.

        See :meth:`~qiskit_experiment.analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """
        default_options = super()._default_options()
        default_options.p0 = {"amp": None, "d_theta": None, "phase": None, "baseline": None}
        default_options.bounds = {"amp": None, "d_theta": None, "phase": None, "baseline": None}
        default_options.fit_reports = {"d_theta": "d_theta"}
        default_options.xlabel = "Number of gates (n)"
        default_options.ylabel = "Population"
        default_options.angle_per_gate = None
        default_options.phase_offset = 0.0
        default_options.number_guesses = 21
        default_options.max_good_angle_error = np.pi / 2

        return default_options

    def _setup_fitting(self, **options) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Fitter options."""
        user_p0 = self._get_option("p0")
        user_bounds = self._get_option("bounds")
        n_guesses = self._get_option("number_guesses")

        max_y, min_y = np.max(self._data().y), np.min(self._data().y)
        b_guess = (max_y + min_y) / 2
        a_guess = max_y - min_y

        max_abs_y = np.max(np.abs(self._data().y))

        # Base the initial guess on the intended angle_per_gate.
        angle_per_gate = self._get_option("angle_per_gate")

        if angle_per_gate is None:
            raise CalibrationError("The angle_per_gate was not specified in the analysis options.")

        guess_range = max(abs(angle_per_gate), np.pi / 2)

        fit_options = []

        for angle in np.linspace(-guess_range, guess_range, n_guesses):
            fit_option = {
                "p0": {
                    "amp": user_p0["amp"] or a_guess,
                    "d_theta": angle,
                    "baseline": b_guess,
                },
                "bounds": {
                    "amp": user_bounds.get("amp", None) or (-2 * max_abs_y, 2 * max_abs_y),
                    "d_theta": user_bounds.get("d_theta", None) or (-np.pi, np.pi),
                    "baseline": user_bounds.get("d_theta", None) or (-1 * max_abs_y, 1 * max_abs_y),
                },
            }

            fit_options.append(fit_option)

        return fit_options

    def _post_analysis(self, analysis_result: CurveAnalysisResult) -> CurveAnalysisResult:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three,
            - a measured angle error that is smaller than the allowed maximum good angle error.
              This quantity is set in the analysis options.
        """
        fit_d_theta = get_opt_value(analysis_result, "d_theta")
        max_good_angle_error = self._get_option("max_good_angle_error")

        criteria = [
            analysis_result["reduced_chisq"] < 3,
            abs(fit_d_theta) < abs(max_good_angle_error),
        ]

        if all(criteria):
            analysis_result["quality"] = "computer_good"
        else:
            analysis_result["quality"] = "computer_bad"

        return analysis_result
