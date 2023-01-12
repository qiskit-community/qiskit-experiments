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
Stark Ramsey fast experiment analysis.
"""

from typing import List, Union

import lmfit
import numpy as np

import qiskit_experiments.curve_analysis as curve
import qiskit_experiments.visualization as vis
from qiskit_experiments.framework import ExperimentData


class StarkRamseyFastAnalysis(curve.CurveAnalysis):
    """StarkRamseyAnalysis

    TODO write docstring.
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
            ylabel="Ramsey P(1)",
            ylim=(0, 1),
            series_params={
                "Xpos": {"color": "blue", "symbol": "o", "label": "Ramsey X(+)"},
                "Ypos": {"color": "blue", "symbol": "^", "label": "Ramsey Y(+)"},
                "Xneg": {"color": "red", "symbol": "o", "label": "Ramsey X(-)"},
                "Yneg": {"color": "red", "symbol": "^", "label": "Ramsey Y(-)"},
            },
        )
        ramsey_plotter.set_options(
            style=vis.PlotStyle({"figsize": (12, 5)})
        )

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
        est_offs = user_opt.p0["offset"]

        # Compute amplitude guess
        amps = np.zeros(0)
        for direction in ("pos", "neg"):
            ram_x_off = curve_data.get_subset_of(f"X{direction}").y - est_offs
            ram_y_off = curve_data.get_subset_of(f"Y{direction}").y - est_offs
            amps = np.concatenate([amps, np.sqrt(ram_x_off ** 2 + ram_y_off ** 2)])
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
        scale = 2 * np.pi * experiment_data.metadata["stark_delay"]
        self.set_options(fixed_parameters={"dt": scale})
