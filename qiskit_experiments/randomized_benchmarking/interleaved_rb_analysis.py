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
from typing import List, Dict, Any, Union

import numpy as np

from qiskit_experiments.analysis import (
    CurveAnalysisResult,
    SeriesDef,
    fit_function,
    get_opt_value,
    get_opt_error,
)
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

    __series__ = [
        SeriesDef(
            name="Standard",
            fit_func=lambda x, a, alpha, alpha_c, b: fit_function.exponential_decay(
                x, amp=a, lamb=-1.0, base=alpha, baseline=b
            ),
            filter_kwargs={"interleaved": False},
            plot_color="red",
            plot_symbol=".",
        ),
        SeriesDef(
            name="Interleaved",
            fit_func=lambda x, a, alpha, alpha_c, b: fit_function.exponential_decay(
                x, amp=a, lamb=-1.0, base=alpha * alpha_c, baseline=b
            ),
            filter_kwargs={"interleaved": True},
            plot_color="orange",
            plot_symbol="^",
        ),
    ]

    @classmethod
    def _default_options(cls):
        default_options = super()._default_options()
        default_options.p0 = {"a": None, "alpha": None, "alpha_c": None, "b": None}
        default_options.bounds = {
            "a": (0., 1.), "alpha": (0., 1.), "alpha_c": (0., 1.), "b": (0., 1.)
        }
        default_options.fit_reports = {"alpha": "\u03B1", "alpha_c": "\u03B1$_c$", "EPC": "EPC"}

        return default_options

    def _setup_fitting(self, **options) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Fitter options."""
        user_p0 = self._get_option("p0")
        user_bounds = self._get_option("bounds")

        std_xdata, std_ydata, _ = self._subset_data(
            name="Standard",
            data_index=self._data_index,
            x_values=self._x_values,
            y_values=self._y_values,
            y_sigmas=self._y_sigmas,
        )
        p0_std = self._initial_guess(std_xdata, std_ydata, self._num_qubits)

        int_xdata, int_ydata, _ = self._subset_data(
            name="Interleaved",
            data_index=self._data_index,
            x_values=self._x_values,
            y_values=self._y_values,
            y_sigmas=self._y_sigmas,
        )
        p0_int = self._initial_guess(int_xdata, int_ydata, self._num_qubits)

        fit_option = {
            "p0": {
                "a": user_p0["a"] or np.mean([p0_std["a"], p0_int["a"]]),
                "alpha": user_p0["alpha"] or p0_std["alpha"],
                "alpha_c": user_p0["alpha_c"] or min(p0_int["alpha"] / p0_std["alpha"], 1),
                "b": user_p0["b"] or np.mean([p0_std["b"], p0_int["b"]]),
            },
            "bounds": {
                "a": user_bounds["a"] or (0., 1.),
                "alpha": user_bounds["alpha"] or (0., 1.),
                "alpha_c": user_bounds["alpha_c"] or (0., 1.),
                "b": user_bounds["b"] or (0., 1.),
            }
        }
        fit_option.update(options)

        return fit_option

    def _post_processing(self, analysis_result: CurveAnalysisResult) -> CurveAnalysisResult:
        """Calculate EPC."""
        # Add EPC data
        nrb = 2 ** self._num_qubits
        scale = (nrb - 1) / nrb
        alpha = get_opt_value(analysis_result, "alpha")
        alpha_c = get_opt_value(analysis_result, "alpha_c")
        alpha_c_err = get_opt_error(analysis_result, "alpha_c")

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

        return analysis_result
