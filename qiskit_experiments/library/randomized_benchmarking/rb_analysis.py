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

from typing import List, Dict, Any, Union

import numpy as np

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.data_processing import multi_mean_xy_data
from .rb_utils import RBUtils


class RBAnalysis(curve.CurveAnalysis):
    r"""Standard randomized benchmarking analysis.

    # section: overview
        This analysis takes only single series.
        This series is fit by the exponential decay function.
        From the fit :math:`\alpha` value this analysis estimates the error per Clifford (EPC).

    # section: fit_model
        .. math::

            F(x) = a \alpha^x + b

    # section: fit_parameters
        defpar a:
            desc: Height of decay curve.
            init_guess: Determined by :math:`(y - b) / \alpha^x`.
            bounds: [0, 1]
        defpar b:
            desc: Base line.
            init_guess: Determined by the average :math:`b` of the standard and interleaved RB.
                Usually equivalent to :math:`(1/2)^n` where :math:`n` is number of qubit.
            bounds: [0, 1]
        defpar \alpha:
            desc: Depolarizing parameter.
            init_guess: Determined by the slope of :math:`(y - b)^{-x}` of the first and the
                second data point.
            bounds: [0, 1]

    """

    __series__ = [
        curve.SeriesDef(
            fit_func=lambda x, a, alpha, b: curve.fit_function.exponential_decay(
                x, amp=a, lamb=-1.0, base=alpha, baseline=b
            ),
            plot_color="blue",
            plot_fit_uncertainty=True,
        )
    ]

    @classmethod
    def _default_options(cls):
        """Default analysis options.

        Analysis Options:
            error_dict (Dict[Tuple[Iterable[int], str], float]): Optional.
                Error estimates for gates from the backend properties.
            epg_1_qubit (Dict[int, Dict[str, float]]) : Optional.
                EPG data for the 1-qubit gate involved,
                assumed to have been obtained from previous experiments.
                This is used to estimate the 2-qubit EPG.
            gate_error_ratio (Dict[str, float]): An estimate for the ratios
                between errors on different gates.

        """
        default_options = super()._default_options()
        default_options.p0 = {"a": None, "alpha": None, "b": None}
        default_options.bounds = {"a": (0.0, 1.0), "alpha": (0.0, 1.0), "b": (0.0, 1.0)}
        default_options.xlabel = "Clifford Length"
        default_options.ylabel = "P(0)"
        default_options.fit_reports = {"alpha": "\u03B1", "EPC": "EPC"}
        default_options.error_dict = None
        default_options.epg_1_qubit = None
        default_options.gate_error_ratio = None

        return default_options

    def _setup_fitting(self, **options) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Fitter options."""
        user_p0 = self._get_option("p0")
        user_bounds = self._get_option("bounds")

        curve_data = self._data()
        initial_guess = self._initial_guess(curve_data.x, curve_data.y, self._num_qubits)
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

    def _format_data(self, data: curve.CurveData) -> curve.CurveData:
        """Take average over the same x values."""
        mean_data_index, mean_x, mean_y, mean_e = multi_mean_xy_data(
            series=data.data_index,
            xdata=data.x,
            ydata=data.y,
            sigma=data.y_err,
            method="sample",
        )
        return curve.CurveData(
            label="fit_ready",
            x=mean_x,
            y=mean_y,
            y_err=mean_e,
            data_index=mean_data_index,
        )

    def _run_analysis(self, experiment_data, **options):
        """Run analysis on circuit data."""
        error_dict = options["error_dict"]
        qubits = experiment_data.metadata()["physical_qubits"]
        if not error_dict:
            options["error_dict"] = RBUtils.get_error_dict_from_backend(
                experiment_data.backend, qubits
            )
        return super()._run_analysis(experiment_data, **options)

    def _post_analysis(
        self, result_data: curve.CurveAnalysisResultData
    ) -> curve.CurveAnalysisResultData:
        """Calculate EPC."""
        alpha = curve.get_opt_value(result_data, "alpha")
        alpha_err = curve.get_opt_error(result_data, "alpha")

        scale = (2 ** self._num_qubits - 1) / (2 ** self._num_qubits)
        result_data["EPC"] = scale * (1 - alpha)
        result_data["EPC_err"] = scale * alpha_err / alpha

        # Add EPG data
        gate_error_ratio = self._get_option("gate_error_ratio")
        if gate_error_ratio is None:
            # we attempt to get the ratio from the backend properties
            gate_error_ratio = self._get_option("error_dict")
        count_ops = []
        for meta in self._data(label="raw_data").metadata:
            count_ops += meta.get("count_ops", [])
        if len(count_ops) > 0 and gate_error_ratio is not None:
            gates_per_clifford = RBUtils.gates_per_clifford(count_ops)
            num_qubits = len(self._physical_qubits)
            if num_qubits in [1, 2]:
                if num_qubits == 1:
                    epg = RBUtils.calculate_1q_epg(
                        result_data["EPC"],
                        self._physical_qubits,
                        gate_error_ratio,
                        gates_per_clifford,
                    )
                elif self._num_qubits == 2:
                    epg_1_qubit = self._get_option("epg_1_qubit")
                    epg = RBUtils.calculate_2q_epg(
                        result_data["EPC"],
                        self._physical_qubits,
                        gate_error_ratio,
                        gates_per_clifford,
                        epg_1_qubit=epg_1_qubit,
                    )
                result_data["EPG"] = epg
        return result_data
