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
Plotting functions for experiment analysis
"""
from typing import Callable, List, Tuple, Optional
import numpy as np
from uncertainties import unumpy as unp

from qiskit_experiments.curve_analysis.curve_data import FitData
from qiskit_experiments.framework.matplotlib import get_non_gui_ax


def plot_curve_fit(
    func: Callable,
    result: FitData,
    ax=None,
    num_fit_points: int = 100,
    labelsize: int = 14,
    grid: bool = True,
    fit_uncertainty: List[Tuple[float, float]] = None,
    **kwargs,
):
    """Generate plot of a curve fit analysis result.

    Wraps :func:`matplotlib.pyplot.plot`.

    Args:
        func: the fit function for curve_fit.
        result: a fitting data set.
        ax (matplotlib.axes.Axes): Optional, a matplotlib axes to add the plot to.
        num_fit_points: the number of points to plot for xrange.
        labelsize: label size for plot
        grid: Show grid on plot.
        fit_uncertainty: a list of sigma values to plot confidence interval of fit line.
        **kwargs: Additional options for matplotlib.pyplot.plot

    Returns:
        matplotlib.axes.Axes: the matplotlib axes containing the plot.

    Raises:
        ImportError: if matplotlib is not installed.
    """
    if ax is None:
        ax = get_non_gui_ax()

    if fit_uncertainty is None:
        fit_uncertainty = list()
    elif isinstance(fit_uncertainty, tuple):
        fit_uncertainty = [fit_uncertainty]

    # Default plot options
    plot_opts = kwargs.copy()
    if "color" not in plot_opts:
        plot_opts["color"] = "blue"
    if "linestyle" not in plot_opts:
        plot_opts["linestyle"] = "-"
    if "linewidth" not in plot_opts:
        plot_opts["linewidth"] = 2

    xmin, xmax = result.x_range

    # Plot fit data
    xs = np.linspace(xmin, xmax, num_fit_points)
    ys_fit_with_error = func(xs, **dict(zip(result.popt_keys, result.popt)))

    # Line
    ax.plot(xs, unp.nominal_values(ys_fit_with_error), **plot_opts)

    # Confidence interval of N sigma values
    stdev_arr = unp.std_devs(ys_fit_with_error)
    if np.isfinite(stdev_arr).all():
        for sigma, alpha in fit_uncertainty:
            ax.fill_between(
                xs,
                y1=unp.nominal_values(ys_fit_with_error) - sigma * stdev_arr,
                y2=unp.nominal_values(ys_fit_with_error) + sigma * stdev_arr,
                alpha=alpha,
                color=plot_opts["color"],
            )

    # Formatting
    ax.tick_params(labelsize=labelsize)
    ax.grid(grid)
    return ax


def plot_scatter(
    xdata: np.ndarray,
    ydata: np.ndarray,
    ax=None,
    labelsize: int = 14,
    grid: bool = True,
    **kwargs,
):
    """Generate a scatter plot of xy data.

    Wraps :func:`matplotlib.pyplot.scatter`.

    Args:
        xdata: xdata used for fitting
        ydata: ydata used for fitting
        ax (matplotlib.axes.Axes): Optional, a matplotlib axes to add the plot to.
        labelsize: label size for plot
        grid: Show grid on plot.
        **kwargs: Additional options for :func:`matplotlib.pyplot.scatter`

    Returns:
        matplotlib.axes.Axes: the matplotlib axes containing the plot.
    """
    if ax is None:
        ax = get_non_gui_ax()

    # Default plot options
    plot_opts = kwargs.copy()
    if "c" not in plot_opts:
        plot_opts["c"] = "grey"
    if "marker" not in plot_opts:
        plot_opts["marker"] = "x"
    if "alpha" not in plot_opts:
        plot_opts["alpha"] = 0.8

    # Plot data
    ax.scatter(xdata, unp.nominal_values(ydata), **plot_opts)

    # Formatting
    ax.tick_params(labelsize=labelsize)
    ax.grid(grid)
    return ax


def plot_errorbar(
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    ax=None,
    labelsize: int = 14,
    grid: bool = True,
    **kwargs,
):
    """Generate an errorbar plot of xy data.

    Wraps :func:`matplotlib.pyplot.errorbar`

    Args:
        xdata: xdata used for fitting
        ydata: ydata used for fitting
        sigma: Optional, standard deviation of ydata
        ax (matplotlib.axes.Axes): Optional, a matplotlib axes to add the plot to.
        labelsize: label size for plot
        grid: Show grid on plot.
        **kwargs: Additional options for :func:`matplotlib.pyplot.errorbar`

    Returns:
        matplotlib.axes.Axes: the matplotlib axes containing the plot.
    """
    if ax is None:
        ax = get_non_gui_ax()

    # Default plot options
    plot_opts = kwargs.copy()
    if "color" not in plot_opts:
        plot_opts["color"] = "red"
    if "marker" not in plot_opts:
        plot_opts["marker"] = "."
    if "markersize" not in plot_opts:
        plot_opts["markersize"] = 9
    if "linestyle" not in plot_opts:
        plot_opts["linestyle"] = "None"

    # Plot data
    ax.errorbar(xdata, ydata, yerr=sigma, **plot_opts)

    # Formatting
    ax.tick_params(labelsize=labelsize)
    ax.grid(grid)
    return ax
