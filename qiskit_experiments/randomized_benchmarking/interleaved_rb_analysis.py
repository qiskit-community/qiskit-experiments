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
Interleaved RB analysis class.
"""
from typing import Optional, List
import numpy as np
from qiskit_experiments.analysis.curve_fitting import (
    process_multi_curve_data,
    multi_curve_fit,
)

from qiskit_experiments.analysis.plotting import plot_curve_fit, plot_scatter, plot_errorbar
from qiskit_experiments.analysis.data_processing import (
    level2_probability,
    multi_mean_xy_data,
)
from .rb_analysis import RBAnalysis

try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class InterleavedRBAnalysis(RBAnalysis):
    r"""Interleaved RB Analysis class.
    According to the paper: "Efficient measurement of quantum gate
    error by interleaved randomized benchmarking" (arXiv:1203.4550)

    The epc estimate is obtained using the equation
    :math:`r_{\mathcal{C}}^{\text{est}}=
    \frac{\left(d-1\right)\left(1-p_{\overline{\mathcal{C}}}/p\right)}{d}`

    The error bounds are given by
    :math:`E=\min\left\{ \begin{array}{c}
    \frac{\left(d-1\right)\left[\left|p-p_{\overline{\mathcal{C}}}/p\right|+\left(1-p\right)\right]}{d}\\
    \frac{2\left(d^{2}-1\right)\left(1-p\right)}{pd^{2}}+\frac{4\sqrt{1-p}\sqrt{d^{2}-1}}{p}
    \end{array}\right.`
    """

    # pylint: disable=invalid-name
    def _run_analysis(
        self,
        experiment_data,
        p0: Optional[List[float]] = None,
        plot: bool = True,
        ax: Optional["AxesSubplot"] = None,
    ):
        def data_processor(datum):
            return level2_probability(datum, datum["metadata"]["ylabel"])

        num_qubits = len(experiment_data.data[0]["metadata"]["qubits"])
        series, x, y, sigma = process_multi_curve_data(experiment_data.data, data_processor)
        series, xdata, ydata, ydata_sigma = multi_mean_xy_data(series, x, y, sigma)

        def fit_fun_standard(x, a, alpha_std, _, b):
            return a * alpha_std ** x + b

        def fit_fun_interleaved(x, a, _, alpha_int, b):
            return a * alpha_int ** x + b

        std_idx = series == 0
        std_xdata = xdata[std_idx]
        std_ydata = ydata[std_idx]
        std_ydata_sigma = ydata_sigma[std_idx]
        p0_std = self._p0(std_xdata, std_ydata, num_qubits)

        int_idx = series == 1
        int_xdata = xdata[int_idx]
        int_ydata = ydata[int_idx]
        int_ydata_sigma = ydata_sigma[int_idx]
        p0_int = self._p0(int_xdata, int_ydata, num_qubits)

        p0 = (
            np.mean([p0_std[0], p0_int[0]]),
            p0_std[1],
            p0_int[1],
            np.mean([p0_std[2], p0_int[2]]),
        )

        analysis_result = multi_curve_fit(
            [fit_fun_standard, fit_fun_interleaved],
            series,
            xdata,
            ydata,
            p0,
            ydata_sigma,
            bounds=([0, 0, 0, 0], [1, 1, 1, 1]),
        )

        # Add EPC data
        nrb = 2 ** num_qubits
        scale = (nrb - 1) / (2 ** nrb)
        _, alpha, alpha_c, _ = analysis_result["popt"]
        _, alpha_err, alpha_c_err, _ = analysis_result["popt_err"]

        # Calculate epc_est (=r_c^est) - Eq. (4):
        epc_est = scale * (1 - alpha_c / alpha)
        # Calculate the systematic error bounds - Eq. (5):
        systematic_err_1 = scale * (abs(alpha - alpha_c / alpha) + (1 - alpha))
        systematic_err_2 = (
            2 * (nrb * nrb - 1) * (1 - alpha) / (alpha * nrb * nrb)
            + 4 * (np.sqrt(1 - alpha)) * (np.sqrt(nrb * nrb - 1)) / alpha
        )
        systematic_err = min(systematic_err_1, systematic_err_2)
        systematic_err_l = epc_est - systematic_err
        systematic_err_r = epc_est + systematic_err

        alpha_err_sq = (alpha_err / alpha) ** 2
        alpha_c_err_sq = (alpha_c_err / alpha_c) ** 2
        epc_est_err = (
            ((nrb - 1) / nrb) * (alpha_c / alpha) * (np.sqrt(alpha_err_sq + alpha_c_err_sq))
        )

        analysis_result["EPC"] = epc_est
        analysis_result["EPC_err"] = epc_est_err
        analysis_result["systematic_err"] = systematic_err
        analysis_result["systematic_err_L"] = systematic_err_l
        analysis_result["systematic_err_R"] = systematic_err_r
        analysis_result["plabels"] = ["A", "alpha", "alpha_c", "B"]

        if plot:
            ax = plot_curve_fit(fit_fun_standard, analysis_result, ax=ax)
            ax = plot_curve_fit(fit_fun_interleaved, analysis_result, ax=ax)
            ax = plot_scatter(std_xdata, std_ydata, ax=ax)
            ax = plot_scatter(int_xdata, int_ydata, ax=ax)
            ax = plot_errorbar(std_xdata, std_ydata, std_ydata_sigma, ax=ax)
            ax = plot_errorbar(int_xdata, int_ydata, int_ydata_sigma, ax=ax)
            self._format_plot(ax, analysis_result)
            analysis_result.plt = plt

        return analysis_result, None

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
            alpha_c = analysis_result["popt"][2]
            alpha_err = analysis_result["popt_err"][1]
            alpha_c_err = analysis_result["popt_err"][2]
            epc = analysis_result["EPC"]
            epc_err = analysis_result["EPC_err"]
            box_text = "\u03B1:{:.4f} \u00B1 {:.4f}".format(alpha, alpha_err)
            box_text += "\n\u03B1_c:{:.4f} \u00B1 {:.4f}".format(alpha_c, alpha_c_err)
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
