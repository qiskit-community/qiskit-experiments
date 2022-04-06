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
import warnings
from typing import List, Tuple, Union

import numpy as np
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.data_processing import multi_mean_xy_data, data_sort
from qiskit_experiments.database_service.device_component import Qubit
from qiskit_experiments.framework import AnalysisResultData, ExperimentData

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
            model_description=r"a \alpha^x + b",
        )
    ]

    @classmethod
    def _default_options(cls):
        """Default analysis options.

        Analysis Options:
            epg_1_qubit (Dict[int, Dict[str, float]]) : Optional.
                EPG data for the 1-qubit gate involved,
                assumed to have been obtained from previous experiments.
                This is used to estimate the 2-qubit EPG.
            gate_error_ratio (Dict[str, float]): An estimate for the ratios
                between errors on different gates.
            gate_counts_per_clifford (Dict[Tuple[Sequence[int], str], float]): A dictionary
                of gate numbers constituting a single averaged Clifford operation
                on particular physical qubit.
        """
        default_options = super()._default_options()
        default_options.curve_plotter.set_options(
            xlabel="Clifford Length",
            ylabel="P(0)",
        )
        default_options.plot_raw_data = True
        default_options.result_parameters = ["alpha"]
        default_options.epg_1_qubit = None
        default_options.gate_error_ratio = None
        default_options.gate_counts_per_clifford = None

        return default_options

    def set_options(self, **fields):
        if "error_dict" in fields:
            warnings.warn(
                "Option 'error_dict' has been removed and merged into 'gate_error_ratio'.",
                DeprecationWarning,
            )
            fields["gate_error_ratio"] = fields.pop("error_dict")
        super().set_options(**fields)

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
        opt.p0.set_if_empty(b=1 / 2**num_qubits)

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
        scale = (2**self._num_qubits - 1) / (2**self._num_qubits)
        epc = scale * (1 - alpha)

        extra_entries.append(
            AnalysisResultData(
                name="EPC",
                value=epc,
                chisq=fit_data.reduced_chisq,
                quality=self._evaluate_quality(fit_data),
            )
        )

        # Calculate EPG

        if self.options.gate_counts_per_clifford is not None and self.options.gate_error_ratio:
            num_qubits = len(self._physical_qubits)

            if num_qubits == 1:
                epg_dict = RBUtils.calculate_1q_epg(
                    epc,
                    self._physical_qubits,
                    self.options.gate_error_ratio,
                    self.options.gate_counts_per_clifford,
                )
            elif num_qubits == 2:
                epg_1_qubit = self.options.epg_1_qubit
                epg_dict = RBUtils.calculate_2q_epg(
                    epc,
                    self._physical_qubits,
                    self.options.gate_error_ratio,
                    self.options.gate_counts_per_clifford,
                    epg_1_qubit=epg_1_qubit,
                )
            else:
                # EPG calculation is not supported for more than 3 qubits RB
                epg_dict = None

            if epg_dict:
                for qubits, gate_dict in epg_dict.items():
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

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List["pyplot.Figure"]]:

        if self.options.gate_error_ratio is None:
            gate_error_ratio = experiment_data.metadata.get("gate_error_ratio", None)

            try:
                gate_error_ratio = dict(gate_error_ratio)
            except TypeError:
                pass

            if gate_error_ratio is None:
                # For backward compatibility when loading old experiment data.
                # This could return errorneous error ratio.
                # Deprecation warning is triggered on RBUtils.
                gate_error_ratio = RBUtils.get_error_dict_from_backend(
                    backend=experiment_data.backend,
                    qubits=experiment_data.metadata["physical_qubits"],
                )

            self.set_options(gate_error_ratio=gate_error_ratio)

        if self.options.gate_counts_per_clifford is None:
            gpc = experiment_data.metadata.get("gate_counts_per_clifford", None)

            try:
                gpc = dict(gpc)
            except TypeError:
                pass

            if gpc is None and self.options.gate_error_ratio is not False:
                # Just for backward compatibility.
                # New framework assumes it is set to experiment metadata rather than in circuit metadata.
                # Deprecation warning is triggered on RBUtils.
                count_ops = []
                for circ_metadata in experiment_data.data().metadata:
                    count_ops += circ_metadata.get("count_ops", [])
                if len(count_ops) > 0 and self.options.gate_error_ratio is not None:
                    gpc = RBUtils.gates_per_clifford(count_ops)

            self.set_options(gate_counts_per_clifford=gpc)

        return super()._run_analysis(experiment_data)
