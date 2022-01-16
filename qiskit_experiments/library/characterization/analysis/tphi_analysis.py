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
        experiment_data, figures = super()._run_analysis(experiment_data, **options)

        res0 = experiment_data.child_data(0).analysis_results("T1").value.value
        print("res0 = " + str(res0))
        print(experiment_data.child_data(0).analysis_results("T1").value)

        res1 = experiment_data.child_data(1).analysis_results("T2star").value.value
        print("res1 = " + str(res1))
        reciprocal = 1 / (2 * res0) + (1 / res1)
        t_phi = 1 / reciprocal
        print("t_phi = " + str(t_phi))
        err_t1 =experiment_data.child_data(0).analysis_results("T1").value.stderr
        err_t2star =experiment_data.child_data(1).analysis_results("T2star").value.stderr
        print("errors = " + str(err_t1)  +" " + str(err_t2star))

        err_tphi = 2*err_t1 + err_t2star
        print("stderr = " + str(err_tphi))
        analysis_results = []

        analysis_results.append(
            AnalysisResultData(
                name="T_phi",
                value=FitVal(t_phi),
                stderr=err_phi,
                chisq=(1),
                quality="good",
                extra={}
                )
            )
        return analysis_results, []
