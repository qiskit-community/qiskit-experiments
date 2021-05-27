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

from typing import List, Tuple, Dict, Any, Optional

from qiskit.providers.options import Options
from qiskit_experiments.experiment_data import ExperimentData
from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.analysis.curve_fitting import curve_fit, process_curve_data
import numpy as np

from qiskit_experiments.analysis import CurveAnalysis, SeriesDef, FitOptions, exponential_decay
from qiskit_experiments.analysis.data_processing import mean_xy_data,
from qiskit_experiments.analysis import plotting
from qiskit_experiments.experiment_data import AnalysisResult


class RBAnalysis(CurveAnalysis):
    """RB Analysis class."""
    __series__ = [
        SeriesDef(
            name="RB curve",
            fit_func=lambda x, a, alpha, b: exponential_decay(
                x, amp=a, lamb=-1.0, base=alpha, baseline=b
            ),
            plot_color="blue",
        )
    ]

    __plot_labels__ = {"alpha": "\u03B1", "EPC": "EPC"}

    __plot_xlabel__ = "Clifford Length"

    __plot_ylabel__ = "P(0)"


    @classmethod
    def _default_options(cls):
        return Options(
            p0=None,
            plot=True,
            ax=None,
        )

    def _setup_fitting(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        y_sigmas: np.ndarray,
        series: np.ndarray,
        **options,
    ) -> List[FitOptions]:
        """Fitter options."""
        fit_guess = {"a": 0.95, "alpha": 0.99, "b": 1 / 2 ** self.num_qubits}

        # Use the first two points to guess the decay param
        dcliff = x_values[1] - x_values[0]
        dy = (y_values[1] - fit_guess["b"]) / (y_values[0] - fit_guess["b"])
        alpha_guess = dy ** (1 / dcliff)

        if alpha_guess < 1.0:
            fit_guess["alpha"] = alpha_guess

        if y_values[0] > fit_guess["b"]:
            fit_guess["a"] = (y_values[0] - fit_guess["b"]) / fit_guess["alpha"] ** x_values[0]

        fit_options = [
            FitOptions(p0=fit_guess, bounds={"a": [0.0, 1.0], "alpha": [0.0, 1.0], "b": [0.0, 1.0]})
        ]
        return fit_options

    def _data_processor_options(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Set outcome label."""
        return {"outcome": "0" * len(metadata["qubits"])}

    def _pre_processing(
        self, x_values: np.ndarray, y_values: np.ndarray, y_sigmas: np.ndarray, series: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """Average over the same x values."""
        xdata, ydata, sigma = mean_xy_data(x_values, y_values, y_sigmas, method="sample")
        return xdata, ydata, sigma, np.zeros(len(xdata))

    def _post_processing(self, analysis_result: AnalysisResult) -> AnalysisResult:
        """Calculate EPC."""
        alpha = analysis_result["popt"][1]
        alpha_err = analysis_result["popt_err"][1]

        scale = (2 ** self.num_qubits - 1) / (2 ** self.num_qubits)
        analysis_result["EPC"] = scale * (1 - alpha)
        analysis_result["EPC_err"] = scale * alpha_err / alpha

        return analysis_result
