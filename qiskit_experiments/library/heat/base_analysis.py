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
Analysis for HEAT experiments.
"""

from typing import List

import numpy as np

from qiskit_experiments.curve_analysis import ErrorAmplificationAnalysis
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import (
    CompositeAnalysis,
    ExperimentData,
    AnalysisResultData,
    Options,
    FitVal,
)


class HeatAnalysis(ErrorAmplificationAnalysis):
    """An analysis class for HEAT experiment to define the fixed parameters."""

    __fixed_parameters__ = ["angle_per_gate", "phase_offset", "amp"]

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options."""
        options = super()._default_options()
        options.angle_per_gate = np.pi
        options.phase_offset = np.pi / 2
        options.amp = 1.0

        return options


class CompositeHeatAnalysis(CompositeAnalysis):
    r"""A composite error amplification analysis to get unitary error coefficients.

    # section: fit_model

        This analysis takes a set of `d_theta` parameters from two error amplification
        analysis results. Each parameter is extracted from the HEAT experiment with
        different control qubit states. Namely,

        .. math::

            A_{I\beta} = \frac{{d\theta_{\beta 0}} + d\theta_{\beta 1}}}{2}

            A_{Z\beta} = \frac{{d\theta_{\beta 0}} - d\theta_{\beta 1}}}{2}

        where, :math:`\beta \in [X, Y, Z]` is one of single qubit Pauli terms,
        :math:`d\theta_{\beta k}` is `d_theta` parameter extracted from the HEAT experiment
        with the control qubit state :math:`|k\rangle \in [|0\rangle, |1\rangle]`.

    # section: see_also
        qiskit_experiments.curve_analysis.ErrorAmplificationAnalysis

    """

    def __init__(self, fit_params: List[str], out_params: List[str]):
        """Create new HEAT analysis.

        Args:
            fit_params: Name of error parameters for each amplification sequence.
            out_params: Name of Hamiltonian coefficients.
        """
        super(CompositeHeatAnalysis, self).__init__()

        if len(fit_params) != 2:
            raise AnalysisError(
                f"{self.__class__.__name__} assumes two fit parameters extracted from "
                "a set of experiments with different control qubit state input. "
                f"{len(fit_params)} input parameter names are specified."
            )
        self.fit_params = fit_params

        if len(out_params) != 2:
            raise AnalysisError(
                f"{self.__class__.__name__} assumes two output parameters computed with "
                "a set of experiment results with different control qubit state input. "
                f"{len(out_params)} output parameter names are specified."
            )
        self.out_params = out_params

    def _run_analysis(self, experiment_data: ExperimentData, **options):

        # Create analysis data of nested experiment and discard redundant entry.
        # Note that experiment_data is mutable.
        super()._run_analysis(experiment_data, **options)

        sub_analysis_results = []
        for i, pname in enumerate(fit_params):
            child_data = experiment_data.child_data(i)
            child_data._wait_for_callbacks()
            sub_analysis_results.append(child_data.analysis_results(pname))

        # Check data quality
        is_good_quality = all(r.quality == "good" for r in sub_analysis_results)

        # Compute unitary terms
        ib = (sub_analysis_results[0].value.value + sub_analysis_results[1].value.value) / 2
        zb = (sub_analysis_results[0].value.value - sub_analysis_results[1].value.value) / 2

        # Compute new variance
        sigma = np.sqrt(
            sub_analysis_results[0].value.stderr ** 2 + sub_analysis_results[1].value.stderr ** 2
        )

        estimate_ib = AnalysisResultData(
            name=out_params[0],
            value=FitVal(value=ib, stderr=sigma, unit="rad"),
            quality="good" if is_good_quality else "bad",
        )

        estimate_zb = AnalysisResultData(
            name=out_params[1],
            value=FitVal(value=zb, stderr=sigma, unit="rad"),
            quality="good" if is_good_quality else "bad",
        )

        composite_analysis_results = [estimate_ib, estimate_zb]

        return composite_analysis_results, None
