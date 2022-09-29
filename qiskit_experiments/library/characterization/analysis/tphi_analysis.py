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
from qiskit_experiments.library.characterization.analysis.t1_analysis import T1Analysis
from qiskit_experiments.library.characterization.analysis.t2ramsey_analysis import T2RamseyAnalysis
from qiskit_experiments.exceptions import QiskitError


class TphiAnalysis(CompositeAnalysis):

    r"""
    Tphi result analysis class.
    A class to analyze :math:`T_\phi` experiments.
    """

    def __init__(self, analyses=None):
        if analyses is None:
            analyses = [T1Analysis(), T2RamseyAnalysis()]

        # Validate analyses kwarg
        if (
            len(analyses) != 2
            or not isinstance(analyses[0], T1Analysis)
            or not isinstance(analyses[1], T2RamseyAnalysis)
        ):
            raise QiskitError(
                "Invalid component analyses for T2phi, analyses must be a pair of "
                "T1Analysis and T2RamseyAnalysis instances."
            )
        super().__init__(analyses, flatten_results=True)

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        r"""Run analysis for :math:`T_\phi` experiment.
        It invokes CompositeAnalysis._run_analysis that will invoke
        _run_analysis for the two sub-experiments.
        Based on the results, it computes the result for :math:`T_phi`.
        """
        # Run composite analysis and extract T1 and T2star results
        analysis_results, figures = super()._run_analysis(experiment_data)
        t1_result = next(filter(lambda res: res.name == "T1", analysis_results))
        t2star_result = next(filter(lambda res: res.name == "T2star", analysis_results))

        # Calculate Tphi from T1 and T2star
        tphi = 1 / (1 / t2star_result.value - 1 / (2 * t1_result.value))
        quality_tphi = (
            "good" if (t1_result.quality == "good" and t2star_result.quality == "good") else "bad"
        )
        tphi_result = AnalysisResultData(
            name="T_phi",
            value=tphi,
            chisq=None,
            quality=quality_tphi,
            extra={"unit": "s"},
        )

        # Return combined results
        analysis_results = [tphi_result] + analysis_results
        return analysis_results, figures
