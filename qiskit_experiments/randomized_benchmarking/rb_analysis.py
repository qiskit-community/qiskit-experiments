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

import numpy as np
from qiskit_experiments.analysis.curve_fit_analysis import CurveFitAnalysis

try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class RBAnalysis(CurveFitAnalysis):
    """RB Analysis class."""

    # pylint: disable = arguments-differ
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
        # TODO: Get from experiment level metadata
        num_qubits = len(experiment_data.data[0]["metadata"]["qubits"])

        # Fit function
        def fit_fun(x, a, alpha, b):
            return a * alpha ** x + b

        # Initial guess function
        # NOTE: I don't think this is a good guess function
        # its just inserted as a place holder for one
        # pylint: disable = unused-argument
        def p0_func(xdata, ydata, sigma=None):
            xmin = np.min(xdata)
            y_mean_min = np.mean(ydata[xdata == xmin])

            xmax = np.max(xdata)
            y_mean_max = np.mean(ydata[xdata == xmax])

            b_guess = 1 / (2 ** num_qubits)
            a_guess = 1 - b_guess
            alpha_guess = np.exp(
                np.log((y_mean_min - b_guess) / (y_mean_max - b_guess)) / (xmin - xmax)
            )
            # Make sure alpha guess is feasible
            alpha_guess = max(min(alpha_guess, 1), 0)
            return [a_guess, alpha_guess, b_guess]

        # Run CurveFitAnalysis
        analysis_result, figs = super()._run_analysis(
            experiment_data,
            fit_fun,
            p0=p0,
            p0_func=p0_func,
            bounds=([0, 0, 0], [1, 1, 1]),
            fit_mean_data=True,
            plot=plot,
            ax=ax,
        )

        # Add EPC data
        popt = analysis_result["popt"]
        popt_err = analysis_result["popt_err"]
        scale = (2 ** num_qubits - 1) / (2 ** num_qubits)
        analysis_result["EPC"] = scale * (1 - popt[1])
        analysis_result["EPC_err"] = scale * popt_err[1] / popt[1]
        analysis_result["plabels"] = ["A", "alpha", "B"]

        # Format figure
        if figs is not None:
            self._format_plot(figs[0], analysis_result)
            # TODO: figure out what to do with plots
            plt.show()
        return analysis_result, figs

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