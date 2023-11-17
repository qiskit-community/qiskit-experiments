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
Layer Fidelity RB analysis class.
"""
from typing import List, Tuple, Union

import lmfit

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import CompositeAnalysis, AnalysisResultData, ExperimentData


class _SubLayerFidelityAnalysis(curve.CurveAnalysis):
    r"""A class to analyze a sub-experiment for estimating layer fidelity,
    i.e. one of 1Q/2Q simultaneous direct RBs.

    # section: overview
        This analysis takes only single series.
        This series is fit by the exponential decay function.
        From the fit :math:`\alpha` value this analysis estimates the error per gate (EPG).

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

    @classmethod
    def _default_options(cls):
        """Default analysis options.
        """
        default_options = super()._default_options()
        default_options.plotter.set_figure_options(
            xlabel="Layer Length",
            ylabel="P(0)",
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

        # Calculate EPC
        alpha = fit_data.ufloat_params["alpha"]
        scale = (2**num_qubits - 1) / (2**num_qubits)
        epg = scale * (1 - alpha)

        outcomes.append(
            AnalysisResultData(
                name="EPG",
                value=epg,
                chisq=fit_data.reduced_chisq,
                quality=quality,
                extra=metadata,
            )
        )
        return outcomes



class LayerFidelityAnalysis(CompositeAnalysis):
    r"""A class to analyze layer fidelity experiments.

    # section: see_also
        * :py:class:`qiskit_experiments.library.characterization.analysis.SubLayerFidelityAnalysis`

    # section: reference
        .. ref_arxiv:: 1 2311.05933
    """

    def __init__(self, layers, analyses=None):
        if analyses:
            # TODO: Validation
            pass
        else:
            analyses = []
            for a_layer in layers:
                a_layer_analyses = [_SubLayerFidelityAnalysis(qubits) for qubits in a_layer]
                analyses.append(CompositeAnalysis(a_layer_analyses, flatten_results=True))

        super().__init__(analyses, flatten_results=True)

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        r"""Run analysis for Layer Fidelity experiment.
        It invokes CompositeAnalysis._run_analysis that will invoke
        _run_analysis for the sub-experiments (1Q/2Q simultaneous direct RBs).
        Based on the results, it computes the result for Layer Fidelity.
        """

        # Run composite analysis and extract sub-experiments results
        analysis_results, figures = super()._run_analysis(experiment_data)

        # Calculate Layer Fidelity from EPGs
        lf = None  # TODO
        quality_lf = (
            "good" if all(sub.quality == "good" for sub in analysis_results) else "bad"
        )
        lf_result = AnalysisResultData(
            name="LF",
            value=lf,
            chisq=None,
            quality=quality_lf,
            extra={},
        )

        # TODO: Plot LF by chain length for a full 2q-gate chain

        # Return combined results
        analysis_results = [lf_result] + analysis_results
        # figures = [lf_plot] + figures
        return analysis_results, figures
