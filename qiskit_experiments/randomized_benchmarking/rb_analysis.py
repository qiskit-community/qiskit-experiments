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
Standard RB analysis class.
"""

from typing import Optional, List

from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.analysis.curve_fitting import curve_fit, process_curve_data
from qiskit_experiments.analysis.data_processing import (
    level2_probability,
    mean_xy_data,
)
from qiskit_experiments.analysis.plotting import plot_curve_fit, plot_scatter, plot_errorbar

try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class RBAnalysis(BaseAnalysis):
    """RB Analysis class."""

    # pylint: disable = arguments-differ, invalid-name
    def _run_analysis(
        self,
        experiment_data,
        p0: Optional[List[float]] = None,
        plot: bool = True,
        ax: Optional["AxesSubplot"] = None,
    ):
        """Run analysis on circuit data.
        Args:
            experiment_data (ExperimentData): the experiment data to analyze.
            p0: Optional, initial parameter values for curve_fit.
            plot: If True generate a plot of fitted data.
            ax: Optional, matplotlib axis to add plot to.
        Returns:
            tuple: A pair ``(analysis_result, figures)`` where
                   ``analysis_results`` may be a single or list of
                   AnalysisResult objects, and ``figures`` may be
                   None, a single figure, or a list of figures.
        """
        num_qubits = len(experiment_data.data[0]["metadata"]["qubits"])

        # Process data
        def data_processor(datum):
            return level2_probability(datum, num_qubits * "0")

        # Raw data for each sample
        x_raw, y_raw, sigma_raw = process_curve_data(
            experiment_data.data, data_processor, x_key="xdata"
        )

        # Data averaged over samples
        xdata, ydata, ydata_sigma = mean_xy_data(x_raw, y_raw, sigma_raw, method="sample")

        # Perform fit
        def fit_fun(x, a, alpha, b):
            return a * alpha ** x + b

        p0 = self._p0(xdata, ydata, num_qubits)
        bounds = {"a": [0, 1], "alpha": [0, 1], "b": [0, 1]}
        analysis_result = curve_fit(fit_fun, xdata, ydata, p0, ydata_sigma, bounds=bounds)

        # Add EPC data
        popt = analysis_result["popt"]
        popt_err = analysis_result["popt_err"]
        scale = (2 ** num_qubits - 1) / (2 ** num_qubits)
        analysis_result["EPC"] = scale * (1 - popt[1])
        analysis_result["EPC_err"] = scale * popt_err[1] / popt[1]

        if plot:
            ax = plot_curve_fit(fit_fun, analysis_result, ax=ax)
            ax = plot_scatter(x_raw, y_raw, ax=ax)
            ax = plot_errorbar(xdata, ydata, ydata_sigma, ax=ax)
            self._format_plot(ax, analysis_result)
            analysis_result.plt = plt
        return analysis_result, None

    @staticmethod
    def _p0(xdata, ydata, num_qubits):
        """Initial guess for the fitting function"""
        fit_guess = {"a": 0.95, "alpha": 0.99, "b": 1 / 2 ** num_qubits}
        # Use the first two points to guess the decay param
        dcliff = xdata[1] - xdata[0]
        dy = (ydata[1] - fit_guess["b"]) / (ydata[0] - fit_guess["b"])
        alpha_guess = dy ** (1 / dcliff)
        if alpha_guess < 1.0:
            fit_guess["alpha"] = alpha_guess

        if ydata[0] > fit_guess["b"]:
            fit_guess["a"] = (ydata[0] - fit_guess["b"]) / fit_guess["alpha"] ** xdata[0]

        return fit_guess

    @classmethod
    def _format_plot(cls, ax, analysis_result, add_label=True):
        """Format curve fit plot"""
        # Formatting
        ax.tick_params(labelsize=14)
        ax.set_xlabel("Clifford Length", fontsize=16)
        ax.set_ylabel("Ground State Population", fontsize=16)
        ax.grid(True)

        if add_label:
            alpha = analysis_result["popt"][1]
            alpha_err = analysis_result["popt_err"][1]
            epc = analysis_result["EPC"]
            epc_err = analysis_result["EPC_err"]
            box_text = "\u03B1:{:.4f} \u00B1 {:.4f}".format(alpha, alpha_err)
            box_text += "\nEPC: {:.4f} \u00B1 {:.4f}".format(epc, epc_err)
            bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=1)
            ax.text(
                0.6,
                0.9,
                box_text,
                ha="center",
                va="center",
                size=14,
                bbox=bbox_props,
                transform=ax.transAxes,
            )
        return ax
