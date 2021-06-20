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
    r"""A class to analyze interleaved randomized benchmarking experiment.

    Overview
        This analysis takes only two series for standard and interleaved RB curve fitting.
        From the fit :math:`\alpha` and :math:`\alpha_c` value this analysis estimates
        the error per Clifford (EPC) of interleaved gate.

        The EPC estimate is obtained using the equation


        .. math::

            r_{\mathcal{C}}^{\text{est}} =
                \frac{\left(d-1\right)\left(1-\alpha_{\overline{\mathcal{C}}}/\alpha\right)}{d}

        The error bounds are given by

        .. math::

            E = \min\left\{
                \begin{array}{c}
                    \frac{\left(d-1\right)\left[\left|\alpha-\alpha_{\overline{\mathcal{C}}}\right|
                    +\left(1-\alpha\right)\right]}{d} \\
                    \frac{2\left(d^{2}-1\right)\left(1-\alpha\right)}
                    {\alpha d^{2}}+\frac{4\sqrt{1-\alpha}\sqrt{d^{2}-1}}{\alpha}
                \end{array}
            \right.

        See the reference[1] for more details.



    Fit Model
        The fit is based on the following decay functions.

        .. math::

            F_1(x_1) &= a \alpha^{x_1} + b  ... {\rm standard RB} \\
            F_2(x_2) &= a (\alpha_c \alpha)^{x_2} + b ... {\rm interleaved RB}

    Fit Parameters
        - :math:`a`: Height of decay curve.
        - :math:`b`: Base line.
        - :math:`\alpha`: Depolarizing parameter.
        - :math:`\alpha_c`: Ratio of the depolarizing parameter of
          interleaved RB to standard RB curve.

    Initial Guesses
        - :math:`a`: Determined by the average :math:`a` of the standard and interleaved RB.
        - :math:`b`: Determined by the average :math:`b` of the standard and interleaved RB.
          Usually equivalent to :math:`(1/2)**n` where :math:`n` is number of qubit.
        - :math:`\alpha`: Determined by the slope of :math:`(y_1 - b)**(-x_1)` of the first and the
          second data point of the standard RB.
        - :math:`\alpha_c`: Estimate :math:`\alpha' = \alpha_c * \alpha` from the
          interleaved RB curve, then divide this by the initial guess of :math:`\alpha`.

    Bounds
        - :math:`a`: [0, 1]
        - :math:`b`: [0, 1]
        - :math:`\alpha`: [0, 1]
        - :math:`\alpha_c`: [0, 1]

    References
        [1] "Efficient measurement of quantum gate error by interleaved randomized benchmarking"
            (arXiv:1203.4550).
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
        """Return default data processing options.

        See :meth:`~qiskit_experiment.analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.
        """
        default_options = super()._default_options()
        default_options.p0 = {"a": None, "alpha": None, "alpha_c": None, "b": None}
        default_options.bounds = {
            "a": (0.0, 1.0),
            "alpha": (0.0, 1.0),
            "alpha_c": (0.0, 1.0),
            "b": (0.0, 1.0),
        }
        default_options.fit_reports = {"alpha": "\u03B1", "alpha_c": "\u03B1$_c$", "EPC": "EPC"}

        return default_options

    def _setup_fitting(self, **options) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Fitter options."""
        user_p0 = self._get_option("p0")
        user_bounds = self._get_option("bounds")

        # for standard RB curve
        std_curve = self._data(series_name="Standard")
        p0_std = self._initial_guess(std_curve.x, std_curve.y, self._num_qubits)

        # for interleaved RB curve
        int_curve = self._data(series_name="Interleaved")
        p0_int = self._initial_guess(int_curve.x, int_curve.y, self._num_qubits)

        fit_option = {
            "p0": {
                "a": user_p0["a"] or np.mean([p0_std["a"], p0_int["a"]]),
                "alpha": user_p0["alpha"] or p0_std["alpha"],
                "alpha_c": user_p0["alpha_c"] or min(p0_int["alpha"] / p0_std["alpha"], 1),
                "b": user_p0["b"] or np.mean([p0_std["b"], p0_int["b"]]),
            },
            "bounds": {
                "a": user_bounds["a"] or (0.0, 1.0),
                "alpha": user_bounds["alpha"] or (0.0, 1.0),
                "alpha_c": user_bounds["alpha_c"] or (0.0, 1.0),
                "b": user_bounds["b"] or (0.0, 1.0),
            },
        }
        fit_option.update(options)

        return fit_option

    def _post_analysis(self, analysis_result: CurveAnalysisResult) -> CurveAnalysisResult:
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
