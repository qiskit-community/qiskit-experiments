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

        def fit_fun(x, a, alpha, b):
            return a * alpha ** x + b

        p0 = self._p0(xdata, ydata)
        analysis_result = multi_curve_fit(
            [fit_fun, fit_fun],
            series,
            xdata,
            ydata,
            p0,
            ydata_sigma,
            bounds=([0, 0, 0], [1, 1, 1])
        )
        return analysis_result


