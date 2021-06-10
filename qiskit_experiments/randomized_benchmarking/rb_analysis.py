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

from qiskit_experiments.analysis import (
    CurveAnalysis,
    CurveAnalysisResult,
    SeriesDef,
    fit_function,
    get_opt_value,
    get_opt_error,
)

from .rb_utils import RBUtils
from qiskit_experiments.analysis.data_processing import multi_mean_xy_data
from qiskit_experiments.experiment_data import ExperimentData

class RBAnalysis(CurveAnalysis):
    r"""A class to analyze randomized benchmarking experiment.

    Overview
        This analysis takes only single series.
        This series is fit by the exponential decay function.
        From the fit :math:`\alpha` value this analysis estimates the error per Clifford (EPC).

    Fit Model
        The fit is based on the following decay function.

        .. math::

            F(x) = a \alpha^x + b

    Fit Parameters
        - :math:`a`: Height of decay curve.
        - :math:`b`:  Base line.
        - :math:`\alpha`: Depolarizing parameter. This is the fit parameter of main interest.

    Initial Guesses
        - :math:`a`: Determined by :math:`(y_0 - b) / \alpha^x_0`
          where :math:`b` and :math:`\alpha` are initial guesses.
        - :math:`b`: Determined by :math:`(1/2)^n` where :math:`n` is the number of qubit.
        - :math:`\alpha`: Determined by the slope of :math:`(y - b)^{-x}` of the first and the
          second data point.

    Bounds
        - :math:`a`: [0, 1]
        - :math:`b`: [0, 1]
        - :math:`\alpha`: [0, 1]

    """

    __series__ = [
        SeriesDef(
            fit_func=lambda x, a, alpha, b: fit_function.exponential_decay(
                x, amp=a, lamb=-1.0, base=alpha, baseline=b
            ),
            plot_color="blue",
        )
    ]

    @classmethod
    def _default_options(cls):
        """Return default data processing options.

        See :meth:`~qiskit_experiment.analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """

        default_options = super()._default_options()
        default_options.p0 = {"a": None, "alpha": None, "b": None}
        default_options.bounds = {"a": (0.0, 1.0), "alpha": (0.0, 1.0), "b": (0.0, 1.0)}
        default_options.xlabel = "Clifford Length"
        default_options.ylabel = "P(0)"
        default_options.fit_reports = {"alpha": "\u03B1", "EPC": "EPC"}

        return default_options

    def _setup_fitting(self, **options) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Fitter options."""
        user_p0 = self._get_option("p0")
        user_bounds = self._get_option("bounds")

        initial_guess = self._initial_guess(self._x_values, self._y_values, self._num_qubits)
        fit_option = {
            "p0": {
                "a": user_p0["a"] or initial_guess["a"],
                "alpha": user_p0["alpha"] or initial_guess["alpha"],
                "b": user_p0["b"] or initial_guess["b"],
            },
            "bounds": {
                "a": user_bounds["a"] or (0.0, 1.0),
                "alpha": user_bounds["alpha"] or (0.0, 1.0),
                "b": user_bounds["b"] or (0.0, 1.0),
            },
        }
        fit_option.update(options)

        return fit_option

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

    def _pre_processing(self) -> Tuple[np.ndarray, ...]:
        """Average over the same x values."""
        return multi_mean_xy_data(
            series=self._data_index,
            xdata=self._x_values,
            ydata=self._y_values,
            sigma=self._y_sigmas,
            method="sample",
        )

    def _post_processing(self,
                         analysis_result: CurveAnalysisResult,
                         experiment_data: ExperimentData
                         ) -> CurveAnalysisResult:
        """Calculate EPC."""
        alpha = get_opt_value(analysis_result, "alpha")
        alpha_err = get_opt_error(analysis_result, "alpha")

        scale = (2 ** self._num_qubits - 1) / (2 ** self._num_qubits)
        analysis_result["EPC"] = scale * (1 - alpha)
        analysis_result["EPC_err"] = scale * alpha_err / alpha
        
        # Add EPG data
        num_qubits = len(experiment_data.experiment.physical_qubits)
        if num_qubits in [1, 2]:
            count_ops = []
            for datum in experiment_data.data():
                count_ops += datum["metadata"]["ops_count"]
            gates_per_clifford = RBUtils.gates_per_clifford(count_ops)
            if num_qubits == 1:
                epg = RBUtils.calculate_1q_epg(
                    analysis_result["EPC"],
                    experiment_data.experiment.physical_qubits,
                    experiment_data.backend,
                    gates_per_clifford,
                )
            elif num_qubits == 2:
                epg = RBUtils.calculate_2q_epg(
                    analysis_result["EPC"],
                    experiment_data.experiment.physical_qubits,
                    experiment_data.backend,
                    gates_per_clifford,
                )
            analysis_result["EPG"] = epg


        return analysis_result
