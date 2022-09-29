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

"""Error amplification analysis."""

from typing import List, Union, Optional

import lmfit
import numpy as np

import qiskit_experiments.curve_analysis as curve


class ErrorAmplificationAnalysis(curve.CurveAnalysis):
    r"""Error amplification analysis class based on a fit to a cosine function.

    # section: fit_model

        Analyse an error amplifying calibration experiment by fitting the data to a cosine
        function. The user must also specify the intended rotation angle per gate, here labeled,
        :math:`{\rm apg}`. The parameter of interest in the fit is the deviation from the
        intended rotation angle per gate labeled :math:`{\rm d}\theta`. The fit function is

        .. math::
            y = \frac{{\rm amp}}{2}\cos\left(x[{\rm d}\theta + {\rm apg} ] \
            -{\rm phase\_offset}\right)+{\rm base}

        To understand how the error is measured we can transformed the function above into

        .. math::
            y = \frac{{\rm amp}}{2} \left(\
            \cos\right({\rm d}\theta \cdot x\left)\
            \cos\right({\rm apg} \cdot x - {\rm phase\_offset}\left) -\
            \sin\right({\rm d}\theta \cdot x\left)\
            \sin\right({\rm apg} \cdot x - {\rm phase\_offset}\left)
            \right) + {\rm base}

        When :math:`{\rm apg} \cdot x - {\rm phase\_offset} = (2n + 1) \pi/2` is satisfied the
        fit model above simplifies to

        .. math::
            y = \mp \frac{{\rm amp}}{2} \sin\left({\rm d}\theta \cdot x\right) + {\rm base}

        In the limit :math:`{\rm d}\theta \ll 1`, the error can be estimated from the curve data

        .. math::
            {\rm d}\theta \simeq \mp \frac{2(y - {\rm base})}{x \cdot {\rm amp}}


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
                Extra guesses are added based on curve data when either :math:`\rm amp` or
                :math:`\rm base` is :math:`\pi/2`. See fit model for details.
            bounds: [-0.8 pi, 0.8 pi]. The bounds do not include plus and minus pi since these values
                often correspond to symmetry points of the fit function. Furthermore,
                this type of analysis is intended for values of :math:`d\theta` close to zero.

    """

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="amp / 2 * cos((d_theta + angle_per_gate) * x - phase_offset) + base",
                    name="ping_pong",
                )
            ],
            name=name,
        )

    @classmethod
    def _default_options(cls):
        r"""Return the default analysis options.

        See :meth:`~qiskit_experiment.curve_analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.

        Analysis Options:
            max_good_angle_error (float): The maximum angle error for which the fit is
                considered as good. Defaults to :math:`\pi/2`.
        """
        default_options = super()._default_options()
        default_options.curve_drawer.set_options(
            xlabel="Number of gates (n)",
            ylabel="Population",
            ylim=(0, 1.0),
        )
        default_options.result_parameters = ["d_theta"]
        default_options.max_good_angle_error = np.pi / 2

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
        fixed_params = self.options.fixed_parameters

        max_abs_y, _ = curve.guess.max_height(curve_data.y, absolute=True)
        max_y, min_y = np.max(curve_data.y), np.min(curve_data.y)

        user_opt.bounds.set_if_empty(
            d_theta=(-0.8 * np.pi, 0.8 * np.pi), base=(-max_abs_y, max_abs_y)
        )
        user_opt.p0.set_if_empty(base=(max_y + min_y) / 2)

        if "amp" in user_opt.p0:
            user_opt.p0.set_if_empty(amp=max_y - min_y)
            user_opt.bounds.set_if_empty(amp=(0, 2 * max_abs_y))
            amp = user_opt.p0["amp"]
        else:
            # Fixed parameter
            amp = fixed_params.get("amp", 1.0)

        # Base the initial guess on the intended angle_per_gate and phase offset.
        apg = user_opt.p0.get("angle_per_gate", fixed_params.get("angle_per_gate", 0.0))
        phi = user_opt.p0.get("phase_offset", fixed_params.get("phase_offset", 0.0))

        # Prepare logical guess for specific condition (often satisfied)
        d_theta_guesses = []

        offsets = apg * curve_data.x + phi
        for i in range(curve_data.x.size):
            xi = curve_data.x[i]
            yi = curve_data.y[i]
            if np.isclose(offsets[i] % np.pi, np.pi / 2) and xi > 0:
                # Condition satisfied: i.e. cos(apg x - phi) = 0
                err = -np.sign(np.sin(offsets[i])) * (yi - user_opt.p0["base"]) / (0.5 * amp)
                # Validate estimate. This is just the first order term of Maclaurin expansion.
                if np.abs(err) < 0.5:
                    d_theta_guesses.append(err / xi)
                else:
                    # Terminate guess generation because larger d_theta x will start to
                    # reduce net y value and underestimate the rotation.
                    break

        # Add naive guess for more coverage
        guess_range = max(abs(apg), np.pi / 2)
        d_theta_guesses.extend(np.linspace(-guess_range, guess_range, 11))

        options = []
        for d_theta_guess in d_theta_guesses:
            new_opt = user_opt.copy()
            new_opt.p0.set_if_empty(d_theta=d_theta_guess)
            options.append(new_opt)

        return options

    def _evaluate_quality(self, fit_data: curve.CurveFitResult) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three,
            - a measured angle error that is smaller than the allowed maximum good angle error.
              This quantity is set in the analysis options.
        """
        fit_d_theta = fit_data.ufloat_params["d_theta"]

        criteria = [
            fit_data.reduced_chisq < 3,
            abs(fit_d_theta.nominal_value) < abs(self.options.max_good_angle_error),
        ]

        if all(criteria):
            return "good"

        return "bad"
