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

from typing import List, Optional, Tuple, Dict, Union, Any
import numpy as np

from qiskit.utils import apply_prefix
from qiskit.providers.options import Options
from qiskit_experiments.base_analysis import BaseAnalysis, AnalysisResult
from qiskit_experiments.analysis.curve_fitting import curve_fit, process_curve_data
from qiskit_experiments.analysis.data_processing import level2_probability
from qiskit_experiments.analysis import plotting

from qiskit_experiments.analysis import (
    CurveAnalysis,
    CurveAnalysisResult,
    SeriesDef,
    fit_function,
    get_opt_value,
    get_opt_error,
)

from ..experiment_data import ExperimentData

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
    ) -> Tuple[List[AnalysisResult], List["matplotlib.figure.Figure"]]:
        r"""Calculate T2Ramsey experiment.

        The probability of measuring `+` is assumed to be of the form
        :math:`f(t) = a\mathrm{e}^{-t / T_2^*}\cos(2\pi freq t + \phi) + b`
        for unknown parameters :math:`a, b, freq, \phi, T_2^*`.

        Args:
            experiment_data (ExperimentData): the experiment data to analyze
            user_p0: contains initial values given by the user, for the
            fit parameters :math:`(a, T_2^*, freq, \phi, b)`
            user_bounds: lower and upper bounds on the parameters in p0,
                         given by the user.
                         The first tuple is the lower bounds,
                         The second tuple is the upper bounds.
                         For both params, the order is :math:`a, T_2^*, freq, \phi, b`.
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

        def _format_plot(ax, unit):
            """Format curve fit plot"""
            # Formatting
            ax.tick_params(labelsize=10)
            ax.set_xlabel("Delay (" + str(unit) + ")", fontsize=12)
            ax.set_ylabel("Probability to measure |0>", fontsize=12)

        # implementation of  _run_analysis
        data = experiment_data.data()
        metadata = data[0]["metadata"]
        unit = metadata["unit"]
        conversion_factor = metadata.get("dt_factor", None)
        if conversion_factor is None:
            conversion_factor = 1 if unit in ("s", "dt") else apply_prefix(1, unit)

        xdata, ydata, sigma = process_curve_data(data, lambda datum: level2_probability(datum, "0"))

        t2ramsey_estimate = np.mean(xdata)
        p0, bounds = self._t2ramsey_default_params(
            conversion_factor, user_p0, user_bounds, t2ramsey_estimate
        )
        si_xdata = xdata * conversion_factor
        fit_result = curve_fit(
            osc_fit_fun, si_xdata, ydata, p0=list(p0.values()), sigma=sigma, bounds=bounds
        )

        if plot and plotting.HAS_MATPLOTLIB:
            ax = plotting.plot_curve_fit(osc_fit_fun, fit_result, ax=ax)
            ax = plotting.plot_scatter(si_xdata, ydata, ax=ax)
            ax = plotting.plot_errorbar(si_xdata, ydata, sigma, ax=ax)
            _format_plot(ax, unit)
            figures = [ax.get_figure()]
        else:
            figures = None

        # Output unit is 'sec', regardless of the unit used in the input
        analysis_result = AnalysisResult(
            {
                "t2ramsey_value": fit_result["popt"][1],
                "frequency_value": fit_result["popt"][2],
                "stderr": fit_result["popt_err"][1],
                "unit": "s",
                "label": "T2Ramsey",
                "fit": fit_result,
                "quality": self._fit_quality(
                    fit_result["popt"], fit_result["popt_err"], fit_result["reduced_chisq"]
                ),
            }
        )

        analysis_result["fit"]["circuit_unit"] = unit
        if unit == "dt":
            analysis_result["fit"]["dt"] = conversion_factor
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
            return "computer_good"
        else:
            return "computer_bad"


class RamseyXYAnalysis(CurveAnalysis):
    """A class to analyze oscillation in complex plane.

    Overview
        This analysis takes two series for real and imaginary part oscillation to
        find oscillating frequency with sign.

    """
    __series__ = [
        SeriesDef(
            fit_func=lambda x, a, freq, b: fit_function.cos(
                x, amp=a, freq=freq, phase=0., baseline=b
            ),
            filter_kwargs={"post_pulse": "x"},
            name="sx-sx",
            plot_color="red",
            plot_symbol="o",
        ),
        SeriesDef(
            fit_func=lambda x, a, freq, b: fit_function.sin(
                x, amp=a, freq=freq, phase=np.pi, baseline=b
            ),
            filter_kwargs={"post_pulse": "y"},
            name="sx-sy",
            plot_color="blue",
            plot_symbol="^",
        )
    ]

    @classmethod
    def _default_options(cls):
        """Return default data processing options.

        See :meth:`~qiskit_experiment.analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """
        default_options = super()._default_options()
        default_options.p0 = {"a": None, "freq": None, "b": None}
        default_options.bounds = {"a": None, "freq": None, "b": None}
        default_options.fit_reports = {"freq": "frequency"}

        return default_options

    def _setup_fitting(self, **options) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Fitter options."""
        user_p0 = self._get_option("p0")
        user_bounds = self._get_option("bounds")

        # TODO write init guess generation code

        fit_option = {
            "p0": user_p0,
            "bounds": user_bounds,
        }
        fit_option.update(options)

        return fit_option
