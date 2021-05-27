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

from typing import List, Tuple, Dict, Any, Union

import numpy as np

from qiskit_experiments.analysis import CurveAnalysis, SeriesDef, exponential_decay
from qiskit_experiments.analysis.data_processing import mean_xy_data
from qiskit_experiments.experiment_data import AnalysisResult


class RBAnalysis(CurveAnalysis):
    """RB Analysis class."""

    __series__ = [
        SeriesDef(
            fit_func=lambda x, a, alpha, b: exponential_decay(
                x, amp=a, lamb=-1.0, base=alpha, baseline=b
            ),
            plot_color="blue",
        )
    ]

    @classmethod
    def _default_options(cls):
        default_options = super()._default_options()
        default_options.p0 = None
        default_options.xlabel = "Clifford Length"
        default_options.ylabel = "P(0)"
        default_options.fit_reports = {"alpha": "\u03B1", "EPC": "EPC"}

        return default_options

    def _setup_fitting(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        y_sigmas: np.ndarray,
        series: np.ndarray,
        **options,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Fitter options."""
        return {
            "p0": self._initial_guess(x_values, y_values, options["num_qubits"]),
            "bounds": {"a": [0.0, 1.0], "alpha": [0.0, 1.0], "b": [0.0, 1.0]},
        }

    @staticmethod
    def _initial_guess(
        x_values: np.ndarray, y_values: np.ndarray, num_qubits: int
    ) -> Dict[str, float]:
        """Create initial guess with experiment data."""
        fit_guess = {"a": 0.95, "alpha": 0.99, "b": 1 / 2 ** num_qubits}

        # Use the first two points to guess the decay param
        dcliff = x_values[1] - x_values[0]
        dy = (y_values[1] - fit_guess["b"]) / (y_values[0] - fit_guess["b"])
        alpha_guess = dy ** (1 / dcliff)

        if alpha_guess < 1.0:
            fit_guess["alpha"] = alpha_guess

        if y_values[0] > fit_guess["b"]:
            fit_guess["a"] = (y_values[0] - fit_guess["b"]) / fit_guess["alpha"] ** x_values[0]

        return fit_guess

    def _pre_processing(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        y_sigmas: np.ndarray,
        series: np.ndarray,
        **options,
    ) -> Tuple[np.ndarray, ...]:
        """Average over the same x values."""
        xdata, ydata, sigma = mean_xy_data(x_values, y_values, y_sigmas, method="sample")
        return xdata, ydata, sigma, np.zeros(len(xdata))

    def _post_processing(self, analysis_result: AnalysisResult, **options) -> AnalysisResult:
        """Calculate EPC."""
        alpha = analysis_result["popt"][1]
        alpha_err = analysis_result["popt_err"][1]

        scale = (2 ** options["num_qubits"] - 1) / (2 ** options["num_qubits"])
        analysis_result["EPC"] = scale * (1 - alpha)
        analysis_result["EPC_err"] = scale * alpha_err / alpha

        return analysis_result
