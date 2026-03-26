# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Purity RB analysis class.
"""

from typing import List, Dict, Union

from qiskit.result import sampled_expectation_value

from qiskit_experiments.curve_analysis import ScatterTable
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import AnalysisResultData
from .rb_analysis import RBAnalysis, _calculate_epg, _exclude_1q_error


class PurityRBAnalysis(RBAnalysis):
    r"""A class to analyze purity randomized benchmarking experiments.

    # section: overview
        This analysis takes only single series.
        This series is fit by the exponential decay function.
        From the fit :math:`\alpha` value this analysis estimates the error per Clifford (EPC).

        When analysis option ``gate_error_ratio`` is provided, this analysis also estimates
        errors of individual gates assembling a Clifford gate.
        In computation of two-qubit EPC, this analysis can also decompose
        the contribution from the underlying single qubit depolarizing channels when
        ``epg_1_qubit`` analysis option is provided [1].

    # section: fit_model
        .. math::

            F(x) = a \alpha^x + b

    # section: fit_parameters
        defpar a:
            desc: Height of decay curve.
            init_guess: Determined by :math:`1 - b`.
            bounds: [0, 1]
        defpar b:
            desc: Base line.
            init_guess: Determined by :math:`(1/2)^n` where :math:`n` is number of qubit.
            bounds: [0, 1]
        defpar \alpha:
            desc: Depolarizing parameter.
            init_guess: Determined by :func:`~.guess.rb_decay`.
            bounds: [0, 1]

    # section: reference
        .. ref_arxiv:: 1 1712.06550

    """

    def __init__(self):
        super().__init__()

    def _run_data_processing(
        self,
        raw_data: List[Dict],
        category: str = "raw",
    ) -> ScatterTable:
        """Perform data processing from the experiment result payload.

        For purity this converts the counts into Trace(rho^2) and then runs the
        rest of the standard RB fitters

        For now this does it by spoofing a new counts dictionary and then
        calling the super _run_data_processing

        Args:
            raw_data: Payload in the experiment data.
            category: Category string of the output dataset.

        Returns:
            Processed data that will be sent to the formatter method.

        Raises:
            DataProcessorError: When key for x values is not found in the metadata.
            ValueError: When data processor is not provided.
        """

        # figure out the number of qubits... has to be 1 or 2 for now
        if self.options.outcome == "0":
            nq = 1
        elif self.options.outcome == "00":
            nq = 2
        else:
            raise ValueError("Only supporting 1 or 2Q purity")

        ntrials = int(len(raw_data) / 3**nq)
        raw_data2 = []
        nshots = int(sum(raw_data[0]["counts"].values()))

        for i in range(ntrials):
            trial_raw = [d for d in raw_data if d["metadata"]["trial"] == i]

            raw_data2.append(trial_raw[0])

            purity = 1 / 2**nq
            if nq == 1:
                for ii in range(3):
                    purity += sampled_expectation_value(trial_raw[ii]["counts"], "Z") ** 2 / 2**nq
            else:
                for ii in range(9):
                    purity += sampled_expectation_value(trial_raw[ii]["counts"], "ZZ") ** 2 / 2**nq
                    purity += (
                        sampled_expectation_value(trial_raw[ii]["counts"], "IZ") ** 2
                        / 2**nq
                        / 3 ** (nq - 1)
                    )
                    purity += (
                        sampled_expectation_value(trial_raw[ii]["counts"], "ZI") ** 2
                        / 2**nq
                        / 3 ** (nq - 1)
                    )

            raw_data2[-1]["counts"] = {
                "0" * nq: int(purity * nshots * 10),
                "1" * nq: int((1 - purity) * nshots * 10),
            }

        return super()._run_data_processing(raw_data2, category)

    def _create_analysis_results(
        self,
        fit_data: curve.CurveFitResult,
        quality: str,
        **metadata,
    ) -> List[AnalysisResultData]:
        """Create analysis results for important fit parameters.

        Args:
            fit_data: Fit outcome.
            quality: Quality of fit outcome.

        Returns:
            List of analysis result data.
        """
        outcomes = curve.CurveAnalysis._create_analysis_results(self, fit_data, quality, **metadata)
        num_qubits = len(self._physical_qubits)

        # Calculate EPC
        # For purity we need to correct by
        alpha = fit_data.ufloat_params["alpha"] ** 0.5
        scale = (2**num_qubits - 1) / (2**num_qubits)
        epc = scale * (1 - alpha)

        outcomes.append(
            AnalysisResultData(
                name="EPC_pur",
                value=epc,
                chisq=fit_data.reduced_chisq,
                quality=quality,
                extra=metadata,
            )
        )

        # Correction for 1Q depolarizing channel if EPGs are provided
        if self.options.epg_1_qubit and num_qubits == 2:
            epc = _exclude_1q_error(
                epc=epc,
                qubits=self._physical_qubits,
                gate_counts_per_clifford=self._gate_counts_per_clifford,
                extra_analyses=self.options.epg_1_qubit,
            )
            outcomes.append(
                AnalysisResultData(
                    name="EPC_pur_corrected",
                    value=epc,
                    chisq=fit_data.reduced_chisq,
                    quality=quality,
                    extra=metadata,
                )
            )

        # Calculate EPG
        if self._gate_counts_per_clifford is not None and self.options.gate_error_ratio:
            epg_dict = _calculate_epg(
                epc=epc,
                qubits=self._physical_qubits,
                gate_error_ratio=self.options.gate_error_ratio,
                gate_counts_per_clifford=self._gate_counts_per_clifford,
            )
            if epg_dict:
                for gate, epg_val in epg_dict.items():
                    outcomes.append(
                        AnalysisResultData(
                            name=f"EPG_pur_{gate}",
                            value=epg_val,
                            chisq=fit_data.reduced_chisq,
                            quality=quality,
                            extra=metadata,
                        )
                    )

        return outcomes

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.ScatterTable,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Create algorithmic initial fit guess from analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        user_opt.bounds.set_if_empty(
            a=(0, 1),
            alpha=(0, 1),
            b=(0, 1),
        )

        b_guess = 1 / 2 ** len(self._physical_qubits)
        if len(curve_data.x) > 3:
            alpha_guess = curve.guess.rb_decay(curve_data.x[0:3], curve_data.y[0:3], b=b_guess)
        else:
            alpha_guess = curve.guess.rb_decay(curve_data.x, curve_data.y, b=b_guess)

        alpha_guess = alpha_guess**2

        if alpha_guess < 0.6:
            a_guess = curve_data.y[0] - b_guess
        else:
            a_guess = (curve_data.y[0] - b_guess) / (alpha_guess ** curve_data.x[0])

        user_opt.p0.set_if_empty(
            b=b_guess,
            a=a_guess,
            alpha=alpha_guess,
        )

        return user_opt
