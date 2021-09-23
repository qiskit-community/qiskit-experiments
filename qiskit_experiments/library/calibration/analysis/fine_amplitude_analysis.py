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

from typing import List, Union

import numpy as np

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.exceptions import CalibrationError


class FineAmplitudeAnalysis(curve.CurveAnalysis):
    r"""Fine amplitude analysis class based on a fit to a cosine function.

    # section: fit_model

        Analyse a fine amplitude calibration experiment by fitting the data to a cosine function.
        The user must also specify the intended rotation angle per gate, here labeled,
        :math:`{\rm apg}`. The parameter of interest in the
        fit is the deviation from the intended rotation angle per gate labeled
        :math:`{\rm d}\theta`. The fit function is

        .. math::
            y = \frac{{\rm amp}}{2}\cos\left(x[{\rm d}\theta + {\rm apg} ] \
            +{\rm phase\_offset}\right)+{\rm base}

    # section: fit_parameters
        defpar \rm amp:
            desc: Amplitude of the oscillation.
            init_guess: The maximum y value less the minimum y value.
            bounds: [-2, 2] scaled to the maximum signal value.

        defpar \rm base:
            desc: Base line.
            init_guess: The average of the data.
            bounds: [-1, 1] scaled to the maximum signal value.

        defpar d\theta:
            desc: The angle offset in the gate that we wish to measure.
            init_guess: Multiple initial guesses are tried ranging from -a to a
                where a is given by :code:`max(abs(angle_per_gate), np.pi / 2)`.
            bounds: [-pi, pi].

    # section: note

        The following is a list of fixed-valued parameters that enter the fit.

        * :math:`{\rm apg}` The angle per gate is set by the user, for example pi for a pi-pulse.
        * :math:`{\rm phase\_offset}` The phase offset in the cosine oscillation, for example,
          :math:`\pi/2` if a square-root of X gate is added before the repeated gates.
    """

    __series__ = [
        curve.SeriesDef(
            # pylint: disable=line-too-long
            fit_func=lambda x, amp, d_theta, phase_offset, base, angle_per_gate: curve.fit_function.cos(
                x,
                amp=0.5 * amp,
                freq=(d_theta + angle_per_gate) / (2 * np.pi),
                phase=phase_offset,
                baseline=base,
            ),
            plot_color="blue",
            model_description=r"\frac{{\rm amp}}{2}\cos\left(x[{\rm d}\theta + {\rm apg} ] "
            r"+ {\rm phase\_offset}\right)+{\rm base}",
        )
    ]

    # The intended angle per gat of the gate being calibrated, e.g. pi for a pi-pulse.
    __fixed_parameters__ = ["angle_per_gate", "phase_offset"]

    @classmethod
    def _default_options(cls):
        r"""Return the default analysis options.

        See :meth:`~qiskit_experiment.curve_analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.

        Analysis Options:
            angle_per_gate (float): The ideal angle per repeated gate.
                The user must set this option as it defaults to None.
            phase_offset (float): A phase offset for the analysis. This phase offset will be
                :math:`\pi/2` if the square-root of X gate is added before the repeated gates.
                This is decided for the user in :meth:`set_schedule` depending on whether the
                sx gate is included in the experiment.
            number_of_guesses (int): The number of initial guesses to try.
            max_good_angle_error (float): The maximum angle error for which the fit is
                considered as good. Defaults to :math:`\pi/2`.
        """
        default_options = super()._default_options()
        default_options.result_parameters = ["d_theta"]
        default_options.xlabel = "Number of gates (n)"
        default_options.ylabel = "Population"
        default_options.angle_per_gate = None
        default_options.phase_offset = 0.0
        default_options.number_guesses = 21
        default_options.max_good_angle_error = np.pi / 2

        return default_options

    def _generate_fit_guesses(
        self, opt: curve.FitOptions
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Compute the initial guesses.

        Args:
            opt: Fit options filled with user provided guess and bounds.

        Returns:
            List of fit options that are passed to the fitter function.

        Raises:
            CalibrationError: When ``angle_per_gate`` is missing.
        """
        n_guesses = self._get_option("number_guesses")

        curve_data = self._data()
        max_abs_y, _ = curve.guess.max_height(curve_data.y, absolute=True)

        opt.bounds.set_if_empty(
            amp=(-2 * max_abs_y, 2 * max_abs_y),
            d_theta=(-np.pi, np.pi),
            base=(-max_abs_y, max_abs_y),
        )

        max_y, min_y = np.max(curve_data.y), np.min(curve_data.y)

        opt.p0.set_if_empty(amp=max_y - min_y, base=(max_y + min_y) / 2)

        # Base the initial guess on the intended angle_per_gate.
        angle_per_gate = self._get_option("angle_per_gate")

        if angle_per_gate is None:
            raise CalibrationError("The angle_per_gate was not specified in the analysis options.")

        guess_range = max(abs(angle_per_gate), np.pi / 2)
        options = []
        for d_theta_guess in np.linspace(-guess_range, guess_range, n_guesses):
            new_opt = opt.copy()
            new_opt.p0.set_if_empty(d_theta=d_theta_guess)
            options.append(new_opt)

        return options

    def _evaluate_quality(self, fit_data: curve.FitData) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three,
            - a measured angle error that is smaller than the allowed maximum good angle error.
              This quantity is set in the analysis options.
        """
        fit_d_theta = fit_data.fitval("d_theta").value
        max_good_angle_error = self._get_option("max_good_angle_error")

        criteria = [
            fit_data.reduced_chisq < 3,
            abs(fit_d_theta) < abs(max_good_angle_error),
        ]

        if all(criteria):
            return "good"

        return "bad"
