# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Analysis classes for Layer Fidelity RB.
"""
from typing import List, Tuple, Union

import lmfit
import numpy as np

import qiskit_experiments.curve_analysis as curve
import qiskit_experiments.database_service.device_component as device
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import CompositeAnalysis, AnalysisResultData, ExperimentData


class _ProcessFidelityAnalysis(curve.CurveAnalysis):
    r"""A class to estimate process fidelity from one of 1Q/2Q simultaneous direct RB experiments

    # section: overview
        This analysis takes only single series.
        This series is fit by the exponential decay function.
        From the fit :math:`\alpha` value this analysis estimates the process fidelity.

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
        .. ref_arxiv:: 1 2311.05933
    """

    def __init__(self, physical_qubits):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="a * alpha ** x + b",
                    name="rb_decay",
                )
            ]
        )
        self._physical_qubits = physical_qubits
        self.set_options(outcome="0" * len(physical_qubits))
        self.plotter.set_figure_options(
            figure_title=f"Simultaneous Direct RB on Qubit{physical_qubits}",
        )

    @classmethod
    def _default_options(cls):
        """Default analysis options."""
        default_options = super()._default_options()
        default_options.plotter.set_figure_options(
            xlabel="Layers",
            ylabel="Ground State Population",
        )
        default_options.plot_raw_data = True
        default_options.result_parameters = ["alpha"]
        default_options.average_method = "sample"

        return default_options

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
        alpha_guess = curve.guess.rb_decay(curve_data.x, curve_data.y, b=b_guess)
        a_guess = (curve_data.y[0] - b_guess) / (alpha_guess ** curve_data.x[0])

        user_opt.p0.set_if_empty(
            b=b_guess,
            a=a_guess,
            alpha=alpha_guess,
        )

        return user_opt

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
        outcomes = super()._create_analysis_results(fit_data, quality, **metadata)
        num_qubits = len(self._physical_qubits)

        # Calculate process fidelity
        alpha = fit_data.ufloat_params["alpha"]
        pf = (1 + (2**num_qubits - 1) * alpha) / (2**num_qubits)

        metadata["qubits"] = self._physical_qubits
        outcomes.append(
            AnalysisResultData(
                name="ProcessFidelity",
                value=pf,
                chisq=fit_data.reduced_chisq,
                quality=quality,
                extra=metadata,
            )
        )
        return outcomes

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        r"""TODO

        Note: Empty analysis results will be returned when failing analysis.
        """
        try:
            return super()._run_analysis(experiment_data)
        except Exception:
            failed_result = AnalysisResultData(
                name="ProcessFidelity",
                value=None,
                quality="failed",
                extra={"qubits": self._physical_qubits},
            )
            return [failed_result], []

    def _get_experiment_components(self, experiment_data: ExperimentData):
        """Set physical qubits to the experiment components."""
        return [device.Qubit(qubit) for qubit in self._physical_qubits]


class _SingleLayerFidelityAnalysis(CompositeAnalysis):
    r"""A class to estimate a process fidelity per disjoint layer.

    Note: Empty analysis results will be returned when failing analysis.

    # section: reference
        .. ref_arxiv:: 1 2311.05933
    """

    def __init__(self, layer, analyses=None):
        if analyses:
            # TODO: Validation
            pass
        else:
            analyses = [_ProcessFidelityAnalysis(qubits) for qubits in layer]

        super().__init__(analyses, flatten_results=True)
        self._layer = layer

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        r"""TODO"""
        try:
            # Run composite analysis and extract sub-experiments results
            analysis_results, figures = super()._run_analysis(experiment_data)
            # Calculate single layer fidelity from process fidelities of subsystems
            pfs = [res.value for res in analysis_results if res.name == "ProcessFidelity"]
            slf = np.prod(pfs)
            quality_slf = "good" if all(sub.quality == "good" for sub in analysis_results) else "bad"
            slf_result = AnalysisResultData(
                name="SingleLF",
                value=slf,
                quality=quality_slf,
                extra={"qubits": [q for qubits in self._layer for q in qubits]},
            )
            # Return combined results
            analysis_results = [slf_result] + analysis_results
            return analysis_results, figures
        except Exception:
            failed_result = AnalysisResultData(
                name="SingleLF",
                value=None,
                quality="failed",
                extra={"qubits": [q for qubits in self._layer for q in qubits]},
            )
            return [failed_result] + analysis_results, figures

    def _get_experiment_components(self, experiment_data: ExperimentData):
        """Set physical qubits to the experiment components."""
        return [device.Qubit(q) for qubits in self._layer for q in qubits]


class LayerFidelityAnalysis(CompositeAnalysis):
    r"""A class to analyze layer fidelity experiments.

    # section: reference
        .. ref_arxiv:: 1 2311.05933
    """

    def __init__(self, layers, analyses=None):
        if analyses:
            # TODO: Validation
            pass
        else:
            analyses = [_SingleLayerFidelityAnalysis(a_layer) for a_layer in layers]

        super().__init__(analyses, flatten_results=True)
        self.num_layers = len(layers)
        self.num_2q_gates = sum(1 if len(qs) == 2 else 0 for lay in layers for qs in lay)

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        r"""Run analysis for Layer Fidelity experiment.
        It invokes CompositeAnalysis._run_analysis that will invoke
        _run_analysis for the sub-experiments (1Q/2Q simultaneous direct RBs).
        Based on the results, it computes the result for Layer Fidelity.
        """
        try:
            # Run composite analysis and extract sub-experiments results
            analysis_results, figures = super()._run_analysis(experiment_data)
            # Calculate full layer fidelity from single layer fidelities
            slfs = [res.value for res in analysis_results if res.name == "SingleLF"]
            lf = np.prod(slfs)
            quality_lf = "good" if all(sub.quality == "good" for sub in analysis_results) else "bad"
            lf_result = AnalysisResultData(
                name="LF",
                value=lf,
                quality=quality_lf,
            )
            eplg = 1 - (lf ** (1 / self.num_2q_gates))
            eplg_result = AnalysisResultData(
                name="EPLG",
                value=eplg,
                quality=quality_lf,
            )
            # Return combined results
            analysis_results = [lf_result, eplg_result] + analysis_results
            return analysis_results, figures
        except Exception:
            failed_results = [
                AnalysisResultData(
                    name="LF",
                    value=None,
                    quality="failed",
                ),
                AnalysisResultData(
                    name="EPLG",
                    value=None,
                    quality="failed",
                )
            ]
            return failed_results + analysis_results, figures
