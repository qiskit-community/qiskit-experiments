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
Deprecated Visualization Functions.

.. note::
    This module is deprecated and replaced by :mod:`qiskit_experiments.visualization`. The new
    visualization module contains classes to manage drawing to a figure canvas and plotting data
    obtained from an experiment or analysis.
"""

from . import fit_result_plotters
from .base_drawer import BaseCurveDrawer
from .curves import plot_curve_fit, plot_errorbar, plot_scatter
from .fit_result_plotters import FitResultPlotters
from .mpl_drawer import MplCurveDrawer
from .style import PlotterStyle
