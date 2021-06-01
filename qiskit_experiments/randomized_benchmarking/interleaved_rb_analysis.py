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
from qiskit_experiments.analysis.data_processing import (
    level2_probability,
    multi_mean_xy_data,
)
from qiskit_experiments.analysis import plotting

from .rb_analysis import RBAnalysis


class InterleavedRBAnalysis(RBAnalysis):
    r"""Interleaved RB Analysis class.
    According to the paper: "Efficient measurement of quantum gate
    error by interleaved randomized benchmarking" (arXiv:1203.4550)

    The epc estimate is obtained using the equation
    :math:`r_{\mathcal{C}}^{\text{est}}=
    \frac{\left(d-1\right)\left(1-p_{\overline{\mathcal{C}}}/p\right)}{d}`

    The error bounds are given by
    :math:`E=\min\left\{ \begin{array}{c}
    \frac{\left(d-1\right)\left[\left|p-p_{\overline{\mathcal{C}}}\right|+\left(1-p\right)\right]}{d}\\
    \frac{2\left(d^{2}-1\right)\left(1-p\right)}{pd^{2}}+\frac{4\sqrt{1-p}\sqrt{d^{2}-1}}{p}
    \end{array}\right.`
    """
    # pylint: disable=invalid-name
    def _run_analysis(
        self,
        experiment_data,
        p0: Optional[List[float]] = None,
        plot: bool = True,
        ax: Optional["matplotlib.axes.Axes"] = None,
    ):

        data = experiment_data.data()
        num_qubits = len(data[0]["metadata"]["qubits"])

        # Process data
        def data_processor(datum):
            return level2_probability(datum, num_qubits * "0")

        # Raw data for each sample
        series_raw, x_raw, y_raw, sigma_raw = process_multi_curve_data(data, data_processor)

        # Data averaged over samples
        series, xdata, ydata, ydata_sigma = multi_mean_xy_data(series_raw, x_raw, y_raw, sigma_raw)

        # pylint: disable = unused-argument
        def fit_fun_standard(x, a, alpha, alpha_c, b):
            return a * alpha ** x + b

        def fit_fun_interleaved(x, a, alpha, alpha_c, b):
            return a * (alpha * alpha_c) ** x + b

        p0 = self._p0_multi(series, xdata, ydata, num_qubits)
        bounds = {"a": [0, 1], "alpha": [0, 1], "alpha_c": [0, 1], "b": [0, 1]}

        analysis_result = multi_curve_fit(
            [fit_fun_standard, fit_fun_interleaved],
            series,
            xdata,
            ydata,
            p0,
            ydata_sigma,
            bounds=bounds,
        )

        # Add EPC data
        nrb = 2 ** num_qubits
        scale = (nrb - 1) / nrb
        _, alpha, alpha_c, _ = analysis_result["popt"]
        _, _, alpha_c_err, _ = analysis_result["popt_err"]

        # Calculate epc_est (=r_c^est) - Eq. (4):
        epc_est = scale * (1 - alpha_c)
        epc_est_err = scale * alpha_c_err
        analysis_result["EPC"] = epc_est
        analysis_result["EPC_err"] = epc_est_err

        # Calculate the systematic error bounds - Eq. (5):
        systematic_err_1 = scale * (abs(alpha - alpha_c) + (1 - alpha))
        systematic_err_2 = (
            2 * (nrb * nrb - 1) * (1 - alpha) / (alpha * nrb * nrb)
            + 4 * (np.sqrt(1 - alpha)) * (np.sqrt(nrb * nrb - 1)) / alpha
        )
        systematic_err = min(systematic_err_1, systematic_err_2)
        systematic_err_l = epc_est - systematic_err
        systematic_err_r = epc_est + systematic_err
        analysis_result["EPC_systematic_err"] = systematic_err
        analysis_result["EPC_systematic_bounds"] = [max(systematic_err_l, 0), systematic_err_r]

        if plot and plotting.HAS_MATPLOTLIB:
            ax = plotting.plot_curve_fit(fit_fun_standard, analysis_result, ax=ax, color="blue")
            ax = plotting.plot_curve_fit(
                fit_fun_interleaved,
                analysis_result,
                ax=ax,
                color="green",
            )
            ax = self._generate_multi_scatter_plot(series_raw, x_raw, y_raw, ax=ax)
            ax = self._generate_multi_errorbar_plot(series, xdata, ydata, ydata_sigma, ax=ax)
            self._format_plot(ax, analysis_result)
            ax.legend(loc="center right")
            figures = [ax.get_figure()]
        else:
            figures = None
        return analysis_result, figures

    @staticmethod
    def _generate_multi_scatter_plot(series, xdata, ydata, ax):
        """Generate scatter plot of raw data"""
        idx0 = series == 0
        idx1 = series == 1
        ax = plotting.plot_scatter(xdata[idx0], ydata[idx0], ax=ax)
        ax = plotting.plot_scatter(xdata[idx1], ydata[idx1], ax=ax, marker="+", c="darkslategrey")
        return ax

    @staticmethod
    def _generate_multi_errorbar_plot(series, xdata, ydata, sigma, ax):
        """Generate errorbar plot of average data"""
        idx0 = series == 0
        idx1 = series == 1
        ax = plotting.plot_errorbar(
            xdata[idx0],
            ydata[idx0],
            sigma[idx0],
            ax=ax,
            label="Standard",
            marker=".",
            color="red",
        )
        ax = plotting.plot_errorbar(
            xdata[idx1],
            ydata[idx1],
            sigma[idx1],
            ax=ax,
            label="Interleaved",
            marker="^",
            color="orange",
        )
        return ax

    @staticmethod
    def _p0_multi(series, xdata, ydata, num_qubits):
        """Initial guess for the fitting function"""
        std_idx = series == 0
        p0_std = RBAnalysis._p0(xdata[std_idx], ydata[std_idx], num_qubits)
        int_idx = series == 1
        p0_int = RBAnalysis._p0(xdata[int_idx], xdata[int_idx], num_qubits)
        return {
            "a": np.mean([p0_std["a"], p0_int["a"]]),
            "alpha": p0_std["alpha"],
            "alpha_c": min(p0_int["alpha"] / p0_std["alpha"], 1),
            "b": np.mean([p0_std["b"], p0_int["b"]]),
        }

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
