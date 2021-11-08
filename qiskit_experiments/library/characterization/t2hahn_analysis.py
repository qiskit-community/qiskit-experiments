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
T2Hahn analysis class.
"""

from typing import List, Optional, Tuple, Dict, Union, Any
import dataclasses
import numpy as np

from qiskit.utils import apply_prefix
from qiskit_experiments.framework import (
    BaseAnalysis,
    Options,
    ExperimentData,
    AnalysisResultData,
    FitVal,
)
from qiskit_experiments.curve_analysis import curve_fit, plot_curve_fit, plot_errorbar, plot_scatter
from qiskit_experiments.curve_analysis.curve_fit import process_curve_data
from qiskit_experiments.curve_analysis.data_processing import level2_probability


# pylint: disable = invalid-name
class T2HahnAnalysis(BaseAnalysis):
    r"""
    T2 Hahn result analysis class.

    # section: fit_model
        This class is used to analyze the results of a T2 Hahn Echo experiment.
        The probability of measuring :math:`|+\rangle` state is assumed to be of the form

        .. math::

            f(t) = a\mathrm{e}^{-2*t / T_2} + b

    # section: fit_parameters

        defpar a:
            desc: Amplitude. Height of the decay curve.
            init_guess: 0.5
            bounds: [-0.5, 1.5]

        defpar b:
            desc: Offset. Base line of the decay curve.
            init_guess: 0.5
            bounds: [-0.5, 1.5]

        defpar T_2:
            desc: Represents the rate of decay.
            init_guess: the mean of the input delays.
            bounds: [0, np.inf]

    """

    @classmethod
    def _default_options(cls):
        r"""Default analysis options.

        Analysis Options:
            user_p0 (List[Float]): user guesses for the fit parameters
                :math:`(a, b, T_2)`.
            user_bounds (Tuple[List[float], List[float]]): Lower and upper bounds
                for the fit parameters.
            plot (bool): Create a graph if and only if True.
        """
        return Options(user_p0=None, user_bounds=None)

    # pylint: disable=arguments-differ, unused-argument
    def _run_analysis(
        self,
        experiment_data: ExperimentData,
        user_p0: Optional[Dict[str, float]] = None,
        user_bounds: Optional[Tuple[List[float], List[float]]] = None,
        plot: bool = False,
        ax: Optional["AxesSubplot"] = None,
        **kwargs,
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        r"""Calculate T2Hahn experiment.

        Args:
            experiment_data (ExperimentData): the experiment data to analyze
            user_p0: contains initial values given by the user, for the
            fit parameters :math:`(a, t2hahn, b)`
            user_bounds: lower and upper bounds on the parameters in p0,
                         given by the user.
                         The first tuple is the lower bounds,
                         The second tuple is the upper bounds.
                         For both params, the order is :math:`a, t2hahn, b`.
            plot: if True, create the plot, otherwise, do not create the plot.
            ax: the plot object
            **kwargs: additional parameters for curve fit.

        Returns:
            The analysis result with the estimated :math:`t2hahn`
            The graph of the function.
        """

        def T2_fit_fun(x, a, t2hahn, c):
            """Decay cosine fit function"""
            return a * np.exp(-2 * x / t2hahn) + c

        def _format_plot(ax, unit, fit_result, conversion_factor):
            """Format curve fit plot"""
            # Formatting
            ax.tick_params(labelsize=14)
            ax.set_xlabel("Delay (s)", fontsize=12)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax.set_ylabel("Probability of measuring 0", fontsize=12)
            t2hahn = fit_result["popt"][1] / conversion_factor
            t2hahn_err = fit_result["popt_err"][1] / conversion_factor
            box_text = "$T_2Hahn$ = {:.2f} \u00B1 {:.2f} {}".format(t2hahn, t2hahn_err, unit)
            bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=1)
            ax.text(
                0.6,
                0.9,
                box_text,
                ha="center",
                va="center",
                size=12,
                bbox=bbox_props,
                transform=ax.transAxes,
            )
            return ax

        # implementation of  _run_analysis

        data = experiment_data.data()
        circ_metadata = data[0]["metadata"]
        unit = circ_metadata["unit"]
        conversion_factor = circ_metadata.get("dt_factor", None)
        if conversion_factor is None:
            conversion_factor = 1 if unit in ("s", "dt") else apply_prefix(1, unit)

        xdata, ydata, sigma = process_curve_data(data, lambda datum: level2_probability(datum, "0"))

        t2hahn_estimate = np.mean(xdata)  # Maybe need to change?
        p0, bounds = self._t2hahn_default_params(
            conversion_factor, user_p0, user_bounds, t2hahn_estimate
        )
        xdata *= conversion_factor
        fit_result = curve_fit(
            T2_fit_fun, xdata, ydata, p0=list(p0.values()), sigma=sigma, bounds=bounds
        )
        fit_result = dataclasses.asdict(fit_result)
        fit_result["circuit_unit"] = unit
        if unit == "dt":
            fit_result["dt"] = conversion_factor
        quality = self._fit_quality(
            fit_result["popt"], fit_result["popt_err"], fit_result["reduced_chisq"]
        )
        chisq = fit_result["reduced_chisq"]

        if plot:
            ax = plot_curve_fit(T2_fit_fun, fit_result, ax=ax)
            ax = plot_scatter(xdata, ydata, ax=ax)
            ax = plot_errorbar(xdata, ydata, sigma, ax=ax)
            _format_plot(ax, unit, fit_result, conversion_factor)
            figures = [ax.get_figure()]
        else:
            figures = None

        # Output unit is 'sec', regardless of the unit used in the input
        result_t2hahn = AnalysisResultData(
            "T2hahn",
            value=FitVal(fit_result["popt"][1], fit_result["popt_err"][1], "s"),
            quality=quality,
            chisq=chisq,
            extra=fit_result,
        )

        return [result_t2hahn], figures

    def _t2hahn_default_params(
        self,
        conversion_factor,
        user_p0=None,
        user_bounds=None,
        t2hahn_input=None,
    ) -> Tuple[Dict[str, Union[float, Any]], Union[List[List[Union[Union[float, int], Any]]], Any]]:
        """Default fit parameters for oscillation data.

        Note that :math:`T_2` unit is converted to 'sec' so the
         output will be given in 'sec'.
        """
        if user_p0 is None:
            a = 0.5
            t2hahn = t2hahn_input * conversion_factor
            b = 0.5
        else:
            a = user_p0["A"]
            t2hahn = user_p0["T2"] * conversion_factor
            b = user_p0["B"]
        p0 = {"a_guess": a, "T2": t2hahn, "b_guess": b}

        if user_bounds is None:
            a_bounds = [-0.5, 1.5]
            t2hahn_bounds = [0, np.inf]
            b_bounds = [-0.5, 1.5]
            bounds = (
                [a_bounds[0], t2hahn_bounds[0], b_bounds[0]],
                [a_bounds[1], t2hahn_bounds[1], b_bounds[1]],
            )
        else:
            bounds = user_bounds
        return (p0, bounds)

    @staticmethod
    def _fit_quality(fit_out, fit_err, reduced_chisq):
        # pylint: disable = too-many-boolean-expressions
        if (
            (reduced_chisq < 3)
            and (fit_err[0] is None or fit_err[0] < 0.1 * fit_out[0])
            and (fit_err[1] is None or fit_err[1] < 0.1 * fit_out[1])
            and (fit_err[2] is None or fit_err[2] < 0.1 * fit_out[2])
        ):
            return "good"
        else:
            return "bad"
