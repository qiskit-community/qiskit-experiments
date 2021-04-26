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
Interleaved RB analysis class.
"""
import numpy as np
from typing import Optional, List
from qiskit_experiments.analysis.curve_fitting import (
    process_multi_curve_data,
    multi_curve_fit,
)

from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.analysis.data_processing import (
    level2_probability,
    mean_xy_data,
    filter_data,
)
from .rb_analysis import RBAnalysis

class InterleavedRBAnalysis(RBAnalysis):
    """Interleaved RB Analysis class."""

    def _run_analysis(
            self,
            experiment_data,
            p0: Optional[List[float]] = None,
            plot: bool = True,
            ax: Optional["AxesSubplot"] = None,
    ):
        def data_processor(datum):
            return level2_probability(datum, datum['metadata']['ylabel'])

        self._num_qubits = len(experiment_data.data[0]["metadata"]["qubits"])
        series, x, y, sigma = process_multi_curve_data(experiment_data.data,
                                                       data_processor)
        xdata, ydata, ydata_sigma, series = mean_xy_data(x, y, sigma, series)

        def fit_fun_standard(x, a, alpha_std, alpha_int, b):
            return a * alpha_std ** x + b

        def fit_fun_interleaved(x, a, alpha_std, alpha_int, b):
            return a * alpha_int ** x + b

        std_idx = (series == 0)
        p0_std = self._p0(xdata[std_idx], ydata[std_idx])

        int_idx = (series == 1)
        p0_int = self._p0(xdata[int_idx], ydata[int_idx])

        p0 = (np.mean([p0_std[0], p0_int[0]]),
              p0_std[1], p0_int[1],
              np.mean([p0_std[2], p0_int[2]]))

        analysis_result = multi_curve_fit(
            [fit_fun_standard, fit_fun_interleaved],
            series,
            xdata,
            ydata,
            p0,
            ydata_sigma,
            bounds=([0, 0, 0, 0], [1, 1, 1, 1])
        )
        return analysis_result, None


