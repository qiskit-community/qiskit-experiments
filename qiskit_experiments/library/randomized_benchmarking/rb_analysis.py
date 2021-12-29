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

from typing import List, Union

import numpy as np

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.data_processing import multi_mean_xy_data, data_sort
from qiskit_experiments.database_service.device_component import Qubit
from qiskit_experiments.framework import AnalysisResultData, FitVal
from .rb_utils import RBUtils


class RBAnalysis(curve.CurveAnalysis):
    r"""A class to analyze randomized benchmarking experiments.

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
            model_description=r"a \alpha^x + b",
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
        default_options.xlabel = "Clifford Length"
        default_options.ylabel = "P(0)"
        default_options.result_parameters = ["alpha"]
        default_options.error_dict = None
        default_options.epg_1_qubit = None
        default_options.gate_error_ratio = None

        return default_options

    def _generate_fit_guesses(
        self, user_opt: curve.FitOptions
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Compute the initial guesses.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        curve_data = self._data()

        user_opt.bounds.set_if_empty(
            a=(0, 1),
            alpha=(0, 1),
            b=(0, 1),
        )

        return self._initial_guess(user_opt, curve_data.x, curve_data.y, self._num_qubits)

    @staticmethod
    def _initial_guess(
        opt: curve.FitOptions, x_values: np.ndarray, y_values: np.ndarray, num_qubits: int
    ) -> curve.FitOptions:
        """Create initial guess with experiment data."""
        opt.p0.set_if_empty(b=1 / 2 ** num_qubits)

        # Use the first two points to guess the decay param
        dcliff = x_values[1] - x_values[0]
        dy = (y_values[1] - opt.p0["b"]) / (y_values[0] - opt.p0["b"])
        alpha_guess = dy ** (1 / dcliff)

        opt.p0.set_if_empty(alpha=alpha_guess if alpha_guess < 1.0 else 0.99)

        if y_values[0] > opt.p0["b"]:
            opt.p0.set_if_empty(a=(y_values[0] - opt.p0["b"]) / (opt.p0["alpha"] ** x_values[0]))
        else:
            opt.p0.set_if_empty(a=0.95)

        return opt

    def _format_data(self, data: curve.CurveData) -> curve.CurveData:
        """Data format with averaging with sampling strategy."""
        # take average over the same x value by regenerating sigma from variance of y values
        series, xdata, ydata, sigma, shots = multi_mean_xy_data(
            series=data.data_index,
            xdata=data.x,
            ydata=data.y,
            sigma=data.y_err,
            shots=data.shots,
            method="sample",
        )

        # sort by x value in ascending order
        series, xdata, ydata, sigma, shots = data_sort(
            series=series,
            xdata=xdata,
            ydata=ydata,
            sigma=sigma,
            shots=shots,
        )

        return curve.CurveData(
            label="fit_ready",
            x=xdata,
            y=ydata,
            y_err=sigma,
            shots=shots,
            data_index=series,
        )

    def _extra_database_entry(self, fit_data: curve.FitData) -> List[AnalysisResultData]:
        """Calculate EPC."""
        extra_entries = []

        # Calculate EPC
        alpha = fit_data.fitval("alpha")
        scale = (2 ** self._num_qubits - 1) / (2 ** self._num_qubits)
        epc = FitVal(value=scale * (1 - alpha.value), stderr=scale * alpha.stderr)
        extra_entries.append(
            AnalysisResultData(
                name="EPC",
                value=epc,
                chisq=fit_data.reduced_chisq,
                quality=self._evaluate_quality(fit_data),
            )
        )

        # Calculate EPG
        if not self.options.gate_error_ratio:
            # we attempt to get the ratio from the backend properties
            if not self.options.error_dict:
                gate_error_ratio = RBUtils.get_error_dict_from_backend(
                    backend=self._backend, qubits=self._physical_qubits
                )
            else:
                gate_error_ratio = self.options.error_dict
        else:
            gate_error_ratio = self.options.gate_error_ratio

        count_ops = []
        for meta in self._data(label="raw_data").metadata:
            count_ops += meta.get("count_ops", [])

        if len(count_ops) > 0 and gate_error_ratio is not None:
            gates_per_clifford = RBUtils.gates_per_clifford(count_ops)
            num_qubits = len(self._physical_qubits)

            if num_qubits == 1:
                epg = RBUtils.calculate_1q_epg(
                    epc,
                    self._physical_qubits,
                    gate_error_ratio,
                    gates_per_clifford,
                )
            elif num_qubits == 2:
                epg_1_qubit = self.options.epg_1_qubit
                epg = RBUtils.calculate_2q_epg(
                    epc,
                    self._physical_qubits,
                    gate_error_ratio,
                    gates_per_clifford,
                    epg_1_qubit=epg_1_qubit,
                )
            else:
                # EPG calculation is not supported for more than 3 qubits RB
                epg = None
            if epg:
                for qubits, gate_dict in epg.items():
                    for gate, value in gate_dict.items():
                        extra_entries.append(
                            AnalysisResultData(
                                f"EPG_{gate}",
                                value,
                                chisq=fit_data.reduced_chisq,
                                quality=self._evaluate_quality(fit_data),
                                device_components=[Qubit(i) for i in qubits],
                            )
                        )
        return extra_entries
