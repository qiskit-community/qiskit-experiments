# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Tphi Analysis class.
"""

from typing import List, Tuple
from uncertainties import ufloat

from qiskit_experiments.framework import (
    ExperimentData,
    AnalysisResultData,
    FitVal,
)
from qiskit_experiments.framework.composite.composite_analysis import CompositeAnalysis


class TphiAnalysis(CompositeAnalysis):

    r"""
    Tphi result analysis class.
    A class to analyze :math:`T_\phi` experiments.
    """

    def _run_analysis(
        self, experiment_data: ExperimentData, **options
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        r"""Run analysis for :math:`T_\phi` experiment.
        It invokes CompositeAnalysis._run_analysis that will invoke
        _run_analysis for the two sub-experiments.
        Based on the results, it computes the result for :math:`T_phi`.
        """
        _, _ = super()._run_analysis(experiment_data, **options)

        t1_result = experiment_data.child_data(0).analysis_results("T1")
        t2star_result = experiment_data.child_data(1).analysis_results("T2star")
        # we use the 'ucert' prefix to denote values that include
        # uncertainty using the `uncertainties` package
        uncert_t1_res = ufloat(t1_result.value.value, t1_result.value.stderr)
        uncert_t2star_res = ufloat(t2star_result.value.value, t2star_result.value.stderr)
        uncert_reciprocal = (1 / uncert_t2star_res) - (1 / (2 * uncert_t1_res))
        uncert_tphi = 1 / uncert_reciprocal

        quality_tphi = (
            "good" if (t1_result.quality == "good" and t2star_result.quality == "good") else "bad"
        )

        analysis_results = []
        analysis_results.append(
            AnalysisResultData(
                name="T_phi",
                value=FitVal(uncert_tphi.nominal_value, uncert_tphi.std_dev),
                chisq=None,
                quality=quality_tphi,
                extra={},
            )
        )
        return analysis_results, []
