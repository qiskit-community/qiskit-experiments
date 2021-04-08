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
# pylint: disable = invalid-name

from typing import Optional, Callable, List, Tuple, Dict

import numpy as np
from qiskit.exceptions import QiskitError
from qiskit_experiments.base_analysis import BaseAnalysis
from .curve_fitting import curve_fit
from .curve_fitting_data import curve_fit_data, mean_xy_data
from .plotting import HAS_MATPLOTLIB, plot_curve_fit, plot_errorbar, plot_scatter


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
        **kwargs,
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
        return self._run_curve_fit(
            experiment_data.data,
            func,
            p0=p0,
            p0_func=p0_func,
            bounds=bounds,
            fit_mean_data=fit_mean_data,
            plot=plot,
            ax=ax,
            **kwargs,
        )

    def _run_curve_fit(
        self,
        data: List[Dict[str, any]],
        func: Callable,
        p0: Optional[List[float]] = None,
        p0_func: Optional[Callable] = None,
        bounds: Optional[Tuple] = None,
        fit_mean_data: bool = False,
        plot: bool = True,
        ax: Optional["AxesSubplot"] = None,
        **kwargs,
    ):
        """Run curve fit analysis on circuit data.

        Args:
            data (ExperimentData): the experiment data to analyze.
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
        # Compute curve fit data
        if fit_mean_data:
            xraw, yraw, _ = curve_fit_data(data)
            xdata, ydata, sigma = average_curve_fit_data(xraw, yraw)
        else:
            xdata, ydata, sigma = curve_fit_data(data)

        if p0_func is not None and p0 is None:
            p0 = p0_func(xdata, ydata, sigma=sigma)

        # Run curve fit
        result = curve_fit(func, xdata, ydata, p0, sigma=sigma, bounds=bounds, **kwargs)

        if plot and HAS_MATPLOTLIB:
            ax = plot_curve_fit(func, result)
            if fit_mean_data:
                ax = plot_scatter(xraw, yraw, ax=ax)
                ax = plot_errorbar(xdata, ydata, sigma, ax=ax)
            else:
                ax = plot_scatter(xdata, ydata)
            figs = [ax]
        else:
            figs = None

        return result, figs
