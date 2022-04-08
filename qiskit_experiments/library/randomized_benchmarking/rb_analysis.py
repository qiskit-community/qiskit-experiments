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
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.data_processing import multi_mean_xy_data, data_sort
from qiskit_experiments.framework import AnalysisResultData, ExperimentData
from qiskit_experiments.exceptions import AnalysisError

from .rb_utils import RBUtils, QubitGateTuple, calculate_epg, exclude_1q_error


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
            gate_error_ratio (Dict[str, float]): An estimate for the ratios
                between errors on different gates.
            gate_counts_per_clifford (Union[bool, Dict[str, float]]): A dictionary
                of gate numbers constituting a single averaged Clifford operation
                on particular physical qubit.
            qpg_1_qubit (List[DbAnalysisResultV1]): Analysis results from previous RB experiments
                for individual single qubit gates. If this is provided, EPC of
                2Q RB is corected to exclude the deporalization of underlying 1Q channels.
        """
        default_options = super()._default_options()
        default_options.curve_plotter.set_options(
            xlabel="Clifford Length",
            ylabel="P(0)",
        )
        default_options.plot_raw_data = True
        default_options.result_parameters = ["alpha"]
        default_options.gate_error_ratio = None
        default_options.gate_counts_per_clifford = None
        default_options.qpg_1_qubit = None

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
        # pylint: disable=assignment-from-none
        quality = self._evaluate_quality(fit_data)

        # Calculate EPC
        alpha = fit_data.fitval("alpha")
        scale = (2**self._num_qubits - 1) / (2**self._num_qubits)
        epc = scale * (1 - alpha)

        extra_entries.append(
            AnalysisResultData(
                name="EPC",
                value=epc,
                chisq=fit_data.reduced_chisq,
                quality=quality,
            )
        )

        # Correction for 1Q depolarizing channel if EPGs are provided
        if self.options.qpg_1_qubit and self._num_qubits == 2:
            epc = exclude_1q_error(
                epc=epc,
                qubits=self._physical_qubits,
                gate_counts_per_clifford=self.options.gate_counts_per_clifford,
                extra_analyses=self.options.qpg_1_qubit,
            )
            extra_entries.append(
                AnalysisResultData(
                    name="EPC_corrected",
                    value=epc,
                    chisq=fit_data.reduced_chisq,
                    quality=quality,
                )
            )

        # Calculate EPG
        if self.options.gate_counts_per_clifford is not None and self.options.gate_error_ratio:
            epg_dict = calculate_epg(
                epc=epc,
                qubits=self._physical_qubits,
                gate_error_ratio=self.options.gate_error_ratio,
                gate_counts_per_clifford=self.options.gate_counts_per_clifford,
            )
            if epg_dict:
                for gate, epg_val in epg_dict.items():
                    extra_entries.append(
                        AnalysisResultData(
                            name=f"EPG_{gate}",
                            value=epg_val,
                            chisq=fit_data.reduced_chisq,
                            quality=quality,
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
                gate_error_ratio = {}
                for q_gate_tup, ratio in RBUtils.get_error_dict_from_backend(
                    backend=experiment_data.backend,
                    qubits=experiment_data.metadata["physical_qubits"],
                ).items():
                    # Drop qubit information which is obvious
                    gate_error_ratio[q_gate_tup[1]] = ratio

            self.set_options(gate_error_ratio=gate_error_ratio)

        if self.options.gate_error_ratio and self.options.gate_counts_per_clifford is None:
            avg_gpc = defaultdict(float)
            n_circs = len(experiment_data.data())
            for circ_result in experiment_data.data():
                try:
                    count_ops = circ_result["metadata"]["count_ops"]
                except KeyError as ex:
                    raise AnalysisError(
                        "'count_ops' key is not found in the result metadata. "
                        "This analysis cannot compute error per gates. "
                        "Please disable this with 'gate_error_ratio=False'."
                    ) from ex
                nclif = circ_result["metadata"]["xval"]
                for (qubits, gate), count in count_ops:
                    key = QubitGateTuple(*qubits, gate=gate)
                    avg_gpc[key] += count / nclif / n_circs
            self.set_options(gate_counts_per_clifford=dict(avg_gpc))

        return super()._run_analysis(experiment_data)
