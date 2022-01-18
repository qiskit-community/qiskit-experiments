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
T2Ramsey Experiment class.
"""

from typing import List, Optional, Tuple, Dict
import dataclasses
import numpy as np

from qiskit.utils import apply_prefix
from qiskit_experiments.framework import (
    BaseAnalysis,
    Options,
    ExperimentData,
    AnalysisResultData,
    FitVal,
)
from qiskit_experiments.curve_analysis import curve_fit, plot_curve_fit, plot_errorbar, plot_scatter
from qiskit_experiments.curve_analysis.curve_fit import process_curve_data
from qiskit_experiments.curve_analysis.data_processing import level2_probability
from qiskit_experiments.framework.composite.composite_analysis import CompositeAnalysis


class TphiAnalysis(CompositeAnalysis):

    r"""
    Tphi result analysis class.

    """

    def _run_analysis(self, experiment_data: ExperimentData, **options):
        # run CompositeAnalysis that will invoke _run_analysis for the
        # two sub-experiments
        _, _ = super()._run_analysis(experiment_data, **options)

        t1_result = experiment_data.child_data(0).analysis_results("T1")
        t2star_result = experiment_data.child_data(1).analysis_results("T2star")        
        t1 = t1_result.value.value
        t2star = t2star_result.value.value
        reciprocal = 1 / (2 * t1) + (1 / t2star)
        t_phi = 1 / reciprocal

        err_t1 =t1_result.value.stderr
        err_t2star =t2star_result.value.stderr
        err_tphi = t_phi * np.sqrt((err_t1 / t1)**2 + (err_t2star / t2star)**2)
        quality_tphi = "good" if (t1_result.quality=="good" and
                                  t2star_result.quality=="good") else bad

        analysis_results = []
        analysis_results.append(
            AnalysisResultData(
                name="T_phi",
                value=FitVal(t_phi, err_tphi),
                chisq=None,
                quality=quality_tphi,
                extra={}
                )
            )
        return analysis_results, []
