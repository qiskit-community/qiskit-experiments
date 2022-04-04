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
Visualization functions
"""

from enum import Enum

from .base_drawer import BaseCurveDrawer
from .mpl_drawer import MplCurveDrawer

from . import fit_result_plotters
from .curves import plot_scatter, plot_errorbar, plot_curve_fit
from .style import PlotterStyle


# pylint: disable=invalid-name
class FitResultPlotters(Enum):
    """Map the plotter name to the plotters."""

    mpl_single_canvas = fit_result_plotters.MplDrawSingleCanvas
    mpl_multiv_canvas = fit_result_plotters.MplDrawMultiCanvasVstack
