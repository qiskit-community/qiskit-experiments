# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The analysis class for the Stark Ramsey fast experiment."""

from typing import List, Union

import lmfit
import numpy as np

import qiskit_experiments.curve_analysis as curve
import qiskit_experiments.visualization as vis
from qiskit_experiments.framework import ExperimentData


class StarkRamseyFastAnalysis(curve.CurveAnalysis):
    r"""Ramsey XY analysis for the Stark shifted phase sweep.

    # section: overview

        This analysis is a variant of :class:`RamseyXYAnalysis` in which
        the data is fit for a trigonometric function model with a linear phase.
        In this model, the phase is assumed to be a polynominal of the x-data,
        and techniques to compute a good initial guess for these polynominal coefficients
        are not trivial. For example, when the phase is a linear function of the x-data,
        one may apply Fourier transform to the data to obtain the coefficient,
        but this is not the case.

        This analysis assumes the following polynominal for the phase imparted by the Stark shift.

        .. math::

            \theta_{\text Stark}(x) = 2 \pi t_S f_S(x),

        where

        .. math::

            f_S(x) = c_2 x^2 + c_3 x^3 + f_\epsilon,

        denotes the Stark shift. In the perturbation picture,
        the Stark shift is a quadratic function of :math:`x`, but the cubit term and
        offset are also empirically considered to account for the
        effect of the strong drive and frequency mis-calibration, respectively.

    # section: fit_model

        .. math::

            F_{X+} = \text{amp} \cdot \cos \left( dt f_S^+(x) \right) + \text{offset}, \\
            F_{Y+} = \text{amp} \cdot \sin \left( dt f_S^+(x) \right) + \text{offset}, \\
            F_{X-} = \text{amp} \cdot \cos \left( dt f_S^-(x) \right) + \text{offset}, \\
            F_{Y-} = \text{amp} \cdot \sin \left( dt f_S^-(x) \right) + \text{offset},

        where

        .. math ::

            f_S^\nu(x) = c_2^\nu x^2 + c_3^\nu x^3 + f_\epsilon.

        The Stark shift is asymmetric with respect to :math:`x=0`, because of the
        anti-crossings of higher energy levels. In a typical transmon qubit,
        these levels appear only in :math:`f_S < 0` because of the negative anharmonicity.
        To precisely fit the results, this analysis uses different model parameters
        for positive (:math:`x > 0`) and negative (:math:`x < 0`) shift domains.

        To obtain the initial guess, the following calculation is employed in this analysis.
        First, oscillations in each quadrature are normalized and differentiated.

        .. math ::

            \dot{F}_X = \frac{\partial}{\partial x} \bar{F}_X = dt \frac{d}{dx} f_S \bar{F}_Y, \\
            \dot{F}_Y = \frac{\partial}{\partial x} \bar{F}_Y = - dt \frac{d}{dx} f_S \bar{F}_X. \\

        The root sum square of above quantities yields

        .. math ::

            \sqrt{\dot{F}_X + \dot{F}_Y} = dt \frac{d}{dx} f_S = dt (2 c_2 x + 3 c_3 x^2).

        By using this synthesized data, one can estimate the initial guess of the
        polynomial coefficients by the polynomial fit.
        This fit protocol is independently conducted for the experiment data on the
        positive and negative shift domain.

    # section: fit_parameters

        defpar \rm amp:
            desc: Amplitude of both series.
            init_guess: Median of root sum square of Ramsey X and Y oscillation.
            bounds: [0, 1]

        defpar \rm offset:
            desc: Base line of all series.
            init_guess: Roughly the average of the data.
            bounds: [-1, 1]

        defpar dt:
            desc: Fixed parameter of :math:`2 \pi t_S`, where :math:`t_S` is
                the ``stark_length`` experiment option.
            init_guess: Automatically set from metadata when this analysis is run.
            bounds: None

        defpar c_2^+:
            desc: The quadratic term coefficient of the positive Stark shift.
            init_guess: See the fit model description.
            bounds: None

        defpar c_3^+:
            desc: The cubit term coefficient of the positive Stark shift.
            init_guess: See the fit model description.
            bounds: None

        defpar c_2^-:
            desc: The quadratic term coefficient of the negative Stark shift.
            init_guess: See the fit model description.
            bounds: None

        defpar c_3^-:
            desc: The cubic term coefficient of the negative Stark shift.
            init_guess: See the fit model description.
            bounds: None

        defpar f_\epsilon:
            desc: Constant phase accumulation which is independent of the Stark tone amplitude.
            init_guess: 0
            bounds: None

    # section: see_also

        :class:`qiskit_experiments.library.characterization.analysis.ramsey_xy_analysis.RamseyXYAnalysis`

    """

    def __init__(self):
        super().__init__(
            # Ramsey phase := 2π Δf(x) Δt; Δf(x) = c2 x^2 + c3 x^3 + f_err
            # dt := 2π Δt (const.)
            models=[
                lmfit.models.ExpressionModel(
                    expr="amp * cos(dt * (c2_pos * x**2 + c3_pos * x**3 + f_err)) + offset",
                    name="Xpos",
                    data_sort_key={"series": "X", "direction": "pos"},
                ),
                lmfit.models.ExpressionModel(
                    expr="amp * sin(dt * (c2_pos * x**2 + c3_pos * x**3 + f_err)) + offset",
                    name="Ypos",
                    data_sort_key={"series": "Y", "direction": "pos"},
                ),
                lmfit.models.ExpressionModel(
                    expr="amp * cos(dt * (c2_neg * x**2 + c3_neg * x**3 + f_err)) + offset",
                    name="Xneg",
                    data_sort_key={"series": "X", "direction": "neg"},
                ),
                lmfit.models.ExpressionModel(
                    expr="amp * sin(dt * (c2_neg * x**2 + c3_neg * x**3 + f_err)) + offset",
                    name="Yneg",
                    data_sort_key={"series": "Y", "direction": "neg"},
                ),
            ],
        )

    @classmethod
    def _default_options(cls):
        """Default analysis options."""
        ramsey_plotter = vis.CurvePlotter(vis.MplDrawer())
        ramsey_plotter.set_figure_options(
            xlabel="Stark tone amplitude",
            ylabel="P(1)",
            ylim=(0, 1),
            series_params={
                "Xpos": {"color": "blue", "symbol": "o", "label": "Ramsey X(+)"},
                "Ypos": {"color": "blue", "symbol": "^", "label": "Ramsey Y(+)"},
                "Xneg": {"color": "red", "symbol": "o", "label": "Ramsey X(-)"},
                "Yneg": {"color": "red", "symbol": "^", "label": "Ramsey Y(-)"},
            },
        )
        ramsey_plotter.set_options(style=vis.PlotStyle({"figsize": (12, 5)}))

        options = super()._default_options()
        options.update_options(
            result_parameters=[
                curve.ParameterRepr("c2_pos", "stark_pos_coef_o2", "Hz"),
                curve.ParameterRepr("c3_pos", "stark_pos_coef_o3", "Hz"),
                curve.ParameterRepr("c2_neg", "stark_neg_coef_o2", "Hz"),
                curve.ParameterRepr("c3_neg", "stark_neg_coef_o3", "Hz"),
                curve.ParameterRepr("f_err", "stark_offset", "Hz"),
            ],
            plotter=ramsey_plotter,
        )

        return options

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
        # Compute offset guess
        user_opt.p0.set_if_empty(
            offset=np.mean(curve_data.y),
            f_err=0.0,
        )
        user_opt.bounds.set_if_empty(
            offset=(-1, 1),
            amp=(0, 1),
        )
        est_offs = user_opt.p0["offset"]

        # Compute amplitude guess
        amps = np.zeros(0)
        for direction in ("pos", "neg"):
            ram_x_off = curve_data.get_subset_of(f"X{direction}").y - est_offs
            ram_y_off = curve_data.get_subset_of(f"Y{direction}").y - est_offs
            amps = np.concatenate([amps, np.sqrt(ram_x_off**2 + ram_y_off**2)])
        user_opt.p0.set_if_empty(amp=np.median(amps))
        est_a = user_opt.p0["amp"]
        d_const = user_opt.p0["dt"]

        # Compute polynominal coefficients
        for direction in ("pos", "neg"):
            ram_x_data = curve_data.get_subset_of(f"X{direction}")
            ram_y_data = curve_data.get_subset_of(f"Y{direction}")
            xvals = ram_x_data.x

            # Get normalized sinusoidals
            xnorm = (ram_x_data.y - est_offs) / est_a
            ynorm = (ram_y_data.y - est_offs) / est_a

            # Compute derivative to extract polynominals from sinusoidal
            dx = np.diff(xnorm) / np.diff(xvals)
            dy = np.diff(ynorm) / np.diff(xvals)

            # Eliminate sinusoidal
            phase_poly = np.sqrt(dx**2 + dy**2)

            # Do polyfit up to 2rd order.
            # This must correspond to the 3rd order coeff because of the derivative.
            # The intercept is ignored.
            vmat_xpoly = np.vstack((xvals[1:] ** 2, xvals[1:])).T
            coeffs = np.linalg.lstsq(vmat_xpoly, phase_poly, rcond=-1)[0]

            poly_guess = {
                f"c2_{direction}": coeffs[1] / 2 / d_const,
                f"c3_{direction}": coeffs[0] / 3 / d_const,
            }
            user_opt.p0.set_if_empty(**poly_guess)

        return user_opt

    def _initialize(
        self,
        experiment_data: ExperimentData,
    ):
        super()._initialize(experiment_data)

        # Set scaling factor to convert phase to frequency
        scale = 2 * np.pi * experiment_data.metadata["stark_length"]
        self.set_options(fixed_parameters={"dt": scale})
