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

from qiskit_experiments.framework import ExperimentData, AnalysisResultData
from qiskit_experiments.framework.composite.composite_analysis import CompositeAnalysis


class TphiAnalysis(CompositeAnalysis):

    r"""
    Tphi result analysis class.
    A class to analyze :math:`T_\phi` experiments.
    """

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        r"""Run analysis for :math:`T_\phi` experiment.
        It invokes CompositeAnalysis._run_analysis that will invoke
        _run_analysis for the two sub-experiments.
        Based on the results, it computes the result for :math:`T_phi`.
        """
        super()._run_analysis(experiment_data)

        t1_result = experiment_data.child_data(0).analysis_results("T1")
        t2star_result = experiment_data.child_data(1).analysis_results("T2star")
        tphi = 1 / (1 / t2star_result.value - 1 / (2 * t1_result.value))

        quality_tphi = (
            "good" if (t1_result.quality == "good" and t2star_result.quality == "good") else "bad"
        )

        analysis_results = [
            AnalysisResultData(
                name="T_phi",
                value=tphi,
                chisq=None,
                quality=quality_tphi,
                extra={"unit": "s"},
            )
        ]
        return analysis_results, []
