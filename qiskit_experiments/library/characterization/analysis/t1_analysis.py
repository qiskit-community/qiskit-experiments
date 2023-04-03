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
"""
T1 Analysis class.
"""
from typing import Union

import numpy as np
from uncertainties import unumpy as unp

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import Options
from qiskit_experiments.curve_analysis.curve_data import CurveData


class T1Analysis(curve.DecayAnalysis):
    """A class to analyze T1 experiments."""

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_options()
        options.plotter.set_figure_options(
            xlabel="Delay",
            ylabel="P(1)",
            xval_unit="s",
        )
        options.result_parameters = [curve.ParameterRepr("tau", "T1", "s")]

        return options

    def _evaluate_quality(self, fit_data: curve.CurveFitResult) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three
            - absolute amp is within [0.9, 1.1]
            - base is less than 0.1
            - amp error is less than 0.1
            - tau error is less than its value
            - base error is less than 0.1
        """
        amp = fit_data.ufloat_params["amp"]
        tau = fit_data.ufloat_params["tau"]
        base = fit_data.ufloat_params["base"]

        criteria = [
            fit_data.reduced_chisq < 3,
            abs(amp.nominal_value - 1.0) < 0.1,
            abs(base.nominal_value) < 0.1,
            curve.utils.is_error_not_significant(amp, absolute=0.1),
            curve.utils.is_error_not_significant(tau),
            curve.utils.is_error_not_significant(base, absolute=0.1),
        ]

        if all(criteria):
            return "good"

        return "bad"


class T1KerneledAnalysis(curve.DecayAnalysis):
    """A class to analyze T1 experiments with kerneled data."""

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_options()
        options.plotter.set_figure_options(
            xlabel="Delay",
            ylabel="Normalized Projection on the Main Axis",
            xval_unit="s",
        )
        options.result_parameters = [curve.ParameterRepr("tau", "T1", "s")]
        options.normalization = True

        return options

    def _evaluate_quality(self, fit_data: curve.CurveFitResult) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three
            - absolute amp is within [0.9, 1.1]
            - base is less than 0.1
            - amp error is less than 0.1
            - tau error is less than its value
            - base error is less than 0.1
        """
        amp = fit_data.ufloat_params["amp"]
        tau = fit_data.ufloat_params["tau"]
        base = fit_data.ufloat_params["base"]

        criteria = [
            fit_data.reduced_chisq < 3,
            abs(amp.nominal_value - 1.0) < 0.1,
            abs(base.nominal_value) < 0.1,
            curve.utils.is_error_not_significant(amp, absolute=0.1),
            curve.utils.is_error_not_significant(tau),
            curve.utils.is_error_not_significant(base, absolute=0.1),
        ]

        if all(criteria):
            return "good"

        return "bad"

    def _format_data(
        self,
        curve_data: curve.CurveData,
    ) -> curve.CurveData:
        """Postprocessing for the processed dataset.

        Args:
            curve_data: Processed dataset created from experiment results.

        Returns:
            Formatted data.
        """
        # check if the SVD decomposition categorized 0 as 1 by calculating the average slope
        diff_y = np.diff(unp.nominal_values(curve_data.y), axis=0)
        avg_slope = sum(diff_y) / len(diff_y)
        if avg_slope[0] > 0:
            new_y_data = 1 - curve_data.y
            new_curve_data = CurveData(
                x=curve_data.x,
                y=new_y_data,
                y_err=curve_data.y_err,
                shots=curve_data.shots,
                data_allocation=curve_data.data_allocation,
                labels=curve_data.labels,
            )

            return super()._format_data(new_curve_data)
        return super()._format_data(curve_data)
