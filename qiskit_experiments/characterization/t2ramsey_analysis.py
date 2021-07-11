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
T2Ramsey Experiment class.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np

from qiskit.utils import apply_prefix
from qiskit.providers.options import Options
from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.analysis.curve_fitting import (
    curve_fit,
    process_curve_data,
)
from qiskit_experiments.analysis.data_processing import level2_probability
from qiskit_experiments.analysis import plotting
from qiskit_experiments.experiment_data import ExperimentData
from qiskit_experiments.database_service import DbAnalysisResultV1
from qiskit_experiments.database_service.device_component import Qubit


# pylint: disable = invalid-name
class T2RamseyAnalysis(BaseAnalysis):
    """T2Ramsey Experiment result analysis class."""

    @classmethod
    def _default_options(cls):
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
    ) -> Tuple[List[DbAnalysisResultV1], List["matplotlib.figure.Figure"]]:
        r"""Calculate T2Ramsey experiment.

        The probability of measuring `+` is assumed to be of the form
        :math:`f(t) = a\mathrm{e}^{-t / T_2^*}\cos(2\pi freq t + \phi) + b`
        for unknown parameters :math:`a, b, freq, \phi, T_2^*`.

        Args:
            experiment_data (ExperimentData): the experiment data to analyze
            user_p0: contains initial values given by the user, for the
            fit parameters :math:`(a, t_2ramsey, freq, \phi, b)`
            user_bounds: lower and upper bounds on the parameters in p0,
                         given by the user.
                         The first tuple is the lower bounds,
                         The second tuple is the upper bounds.
                         For both params, the order is :math:`a, t_2ramsey, freq, \phi, b`.
            plot: if True, create the plot, otherwise, do not create the plot.
            ax: the plot object
            **kwargs: additional parameters for curve fit.

        Returns:
            The analysis result with the estimated :math:`T_2Ramsey` and 'freq' (frequency)
            The graph of the function.
        """

        def osc_fit_fun(x, a, t2ramsey, freq, phi, c):
            """Decay cosine fit function"""
            return a * np.exp(-x / t2ramsey) * np.cos(2 * np.pi * freq * x + phi) + c

        def _format_plot(ax, unit, fit_result, conversion_factor):
            """Format curve fit plot"""
            # Formatting
            ax.tick_params(labelsize=14)
            ax.set_xlabel("Delay (s)", fontsize=12)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax.set_ylabel("Probability to measure |0>", fontsize=12)
            t2ramsey = fit_result["popt"][1] / conversion_factor
            t2_err = fit_result["popt_err"][1] / conversion_factor
            box_text = "$T_2Ramsey$ = {:.2f} \u00B1 {:.2f} {}".format(t2ramsey, t2_err, unit)
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

        t2ramsey_estimate = np.mean(xdata)
        p0, bounds = self._t2ramsey_default_params(
            conversion_factor, user_p0, user_bounds, t2ramsey_estimate
        )
        xdata *= conversion_factor
        fit_result = curve_fit(
            osc_fit_fun, xdata, ydata, p0=list(p0.values()), sigma=sigma, bounds=bounds
        )

        if plot and plotting.HAS_MATPLOTLIB:
            ax = plotting.plot_curve_fit(osc_fit_fun, fit_result, ax=ax)
            ax = plotting.plot_scatter(xdata, ydata, ax=ax)
            ax = plotting.plot_errorbar(xdata, ydata, sigma, ax=ax)
            _format_plot(ax, unit, fit_result, conversion_factor)
            figures = [ax.get_figure()]
        else:
            figures = None

        # Output unit is 'sec', regardless of the unit used in the input
        result_data = {
            "t2ramsey_value": fit_result["popt"][1],
            "frequency_value": fit_result["popt"][2],
            "stderr_t2": fit_result["popt_err"][1],
            "stderr_freq": fit_result["popt_err"][2],
            "unit": "s",
            "label": "T2Ramsey",
            "fit": fit_result,
            "quality": self._fit_quality(
                fit_result["popt"], fit_result["popt_err"], fit_result["reduced_chisq"]
            ),
        }

        result_data["fit"]["circuit_unit"] = unit
        if unit == "dt":
            result_data["fit"]["dt"] = conversion_factor

        analysis_result = DbAnalysisResultV1(
            result_data=result_data,
            result_type="T2Ramsey",
            device_components=[Qubit(circ_metadata["qubit"])],
            experiment_id=experiment_data.experiment_id,
            quality=result_data["quality"],
        )

        return [analysis_result], figures

    def _t2ramsey_default_params(
        self,
        conversion_factor,
        user_p0=None,
        user_bounds=None,
        t2ramsey_input=None,
    ) -> Tuple[List[float], Tuple[List[float]]]:
        """Default fit parameters for oscillation data.

        Note that :math:`T_2^*` and 'freq' units are converted to 'sec' and
        will be output in 'sec'.
        """
        if user_p0 is None:
            a = 0.5
            t2ramsey = t2ramsey_input * conversion_factor
            freq = 0.1 / conversion_factor
            phi = 0.0
            b = 0.5
        else:
            a = user_p0["A"]
            t2ramsey = user_p0["t2ramsey"] * conversion_factor
            freq = user_p0["f"] / conversion_factor
            phi = user_p0["phi"]
            b = user_p0["B"]
        p0 = {"a_guess": a, "t2ramsey": t2ramsey, "f_guess": freq, "phi_guess": phi, "b_guess": b}

        if user_bounds is None:
            a_bounds = [-0.5, 1.5]
            t2ramsey_bounds = [0, np.inf]
            f_bounds = [0.1 * freq, 10 * freq]
            phi_bounds = [-np.pi, np.pi]
            b_bounds = [-0.5, 1.5]
            bounds = [
                [a_bounds[i], t2ramsey_bounds[i], f_bounds[i], phi_bounds[i], b_bounds[i]]
                for i in range(2)
            ]
        else:
            bounds = user_bounds
        return p0, bounds

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
