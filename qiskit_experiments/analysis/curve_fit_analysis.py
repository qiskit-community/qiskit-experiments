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
from scipy.optimize import curve_fit
from qiskit_experiments.base_analysis import BaseAnalysis, AnalysisResult

try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


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
        plot: bool = True,
        ax: Optional["AxesSubplot"] = None,
    ):
        """Run curve fit analysis on circuit data.

        Args:
            experiment_data (ExperimentData): the experiment data to analyze.
            func: fit function `f(x, *params)`.
            p0: Optional, initial parameter values for curve_fit.
            p0_func: Optional, function for calculating initial p0.
            bounds: Optional, parameter bounds for curve_fit.
            plot: If True generate a plot of fitted data.
            ax: Optional, matplotlib axis to add plot to.

        Returns:
            tuple: A pair ``(analysis_results, figures)`` where
                   ``analysis_results`` may be a single or list of
                   AnalysisResult objects, and ``figures`` may be
                   None, a single figure, or a list of figures.
        """
        # Initial guess
        xdata, ydata, ystderr = self._curve_fit_data(experiment_data.data)

        if p0_func is not None and p0 is None:
            p0 = p0_func(xdata, ydata, sigma=ystderr)

        # Fit ydata
        popt, pcov = curve_fit(func, xdata, ydata, sigma=ystderr, p0=p0, bounds=bounds)
        popt_err = np.sqrt(np.diag(pcov))
        chisq = self._chisq(xdata, ydata, ystderr, func, popt)

        result = AnalysisResult(
            {
                "popt": popt,
                "popt_err": popt_err,
                "pcov": pcov,
                "chisq": chisq,
            }
        )

        if plot and HAS_MATPLOTLIB:
            ax = self._curve_fit_plot(xdata, ydata, func, popt, popt_err, ax=ax)
            # TODO: figure out what to do with plots
            return result, [ax]

        return result, None

    @staticmethod
    def _counts_probability(counts: "Counts", key: Union[str, int]) -> Tuple[float]:
        """Return the specified outcome probability mean and variance"""
        shots = sum(counts.values())
        p_mean = counts.get(key, 0.0) / shots
        p_var = shots * p_mean * (1 - p_mean)
        return p_mean, p_var

    @staticmethod
    def _chisq(
        xdata: np.ndarray, ydata: np.ndarray, ystderr: np.ndarray, func: Callable, popt: np.ndarray
    ) -> float:
        """Return the chi-squared of fit"""
        yfits = func(xdata, *popt)
        residuals = ((yfits - ydata) / ystderr) ** 2
        return np.sum(residuals)

    @classmethod
    def _curve_fit_data(cls, data: List[Dict[str, any]]) -> Tuple[np.ndarray]:
        """Return input data for curve_fit function.

        This requires count data and metadata with `"xdata"`, `"ylabel"`
        keys containing the numeric x-value for fitting, and the outcome
        bitstring for computing y-value probability from counts.
        """
        size = len(data)
        xdata = np.zeros(size, dtype=int)
        ydata = np.zeros(size, dtype=float)
        ydata_var = np.zeros(size, dtype=float)

        for i, datum in enumerate(data):
            metadata = datum["metadata"]
            xdata[i] = metadata["xdata"]
            y_mean, y_var = cls._counts_probability(datum["counts"], metadata["ylabel"])
            ydata[i] = y_mean
            ydata_var[i] = y_var

        return xdata, ydata, np.sqrt(ydata_var)

    @classmethod
    def _curve_fit_plot(
        cls,
        xdata: np.ndarray,
        ydata: np.ndarray,
        func: Callable,
        popt: np.ndarray,
        popt_err: np.ndarray,
        num_fit_points: int = 100,
        ax: Optional["AxesSubplot"] = None,
    ):

        if not HAS_MATPLOTLIB:
            raise ImportError(
                "{} requires matplotlib to generate curve fit plot."
                ' Run "pip install matplotlib" before.'.format(cls.__name__)
            )

        if ax is None:
            plt.figure()
            ax = plt.gca()

        # Plot raw data
        ax.scatter(xdata, ydata, c="red", marker="x")

        # Plot fit data
        xs = np.linspace(np.min(xdata), np.max(xdata), num_fit_points)
        ys_fit = func(xs, *popt)
        ax.plot(xs, ys_fit, color="blue", linestyle="-", linewidth=2)

        # Plot standard error interval
        ys_upper = func(xs, *(popt + popt_err))
        ys_lower = func(xs, *(popt - popt_err))
        ax.fill_between(xs, ys_lower, ys_upper, color="blue", alpha=0.1)

        # Formatting
        ax.tick_params(labelsize=14)
        ax.grid(True)
        return ax
