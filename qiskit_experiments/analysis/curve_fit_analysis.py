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
Curve fitter analysis class
"""

from typing import Union, Optional, Callable, List, Tuple, Dict

import numpy as np
from qiskit_experiments.base_analysis import BaseAnalysis, AnalysisResult
from .curve_fit_data import curve_fit


class CurveFitAnalysis(BaseAnalysis):
    """Analysis class based on scipy.optimize.curve_fit"""

    # pylint: disable = arguments-differ
    def _run_analysis(
        self,
        experiment_data: "ExperimentData",
        func: Callable,
        p0: Optional[List[float]] = None,
        p0_func: Optional[Callable] = None,
        bounds: Optional[Tuple] = None,
        fit_mean_data: bool = False,
        plot: bool = True,
        ax: Optional["AxesSubplot"] = None,
        **kwargs
    ):
        """Run curve fit analysis on circuit data.

        Args:
            experiment_data (ExperimentData): the experiment data to analyze.
            func: fit function `f(x, *params)`.
            p0: Optional, initial parameter values for curve_fit.
            p0_func: Optional, function for calculating initial p0.
            bounds: Optional, parameter bounds for curve_fit.
            fit_mean_data: Optional, if True pass means of data points to curve_fit.
            plot: If True generate a plot of fitted data.
            ax: Optional, matplotlib axis to add plot to.
            kwargs: additional kwargs to pass to curve_fit.

        Returns:
            tuple: A pair ``(analysis_results, figures)`` where
                   ``analysis_results`` may be a single or list of
                   AnalysisResult objects, and ``figures`` may be
                   None, a single figure, or a list of figures.
        """
        analysis_result, ax = curve_fit(
            experiment_data.data, func=func, p0=p0, p0_func=p0_func,
            bound=bounds, fit_mean_data=fit_mean_data, plot=plotm ax=ax, **kwargs)
        return result, None
