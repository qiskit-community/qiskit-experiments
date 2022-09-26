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
r"""
=========================================================
Visualization (:mod:`qiskit_experiments.visualization`)
=========================================================

.. currentmodule:: qiskit_experiments.visualization

Visualization provides plotting functionality for creating figures from experiment and analysis results.
This includes plotter and drawer classes to plot data in :py:class:`CurveAnalysis` and its subclasses.

Plotter Library
==============

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/class.rst

    BasePlotter
    CurvePlotter

Drawer Library
==============

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/class.rst

    BaseDrawer
    MplDrawer

Plotting Style
==============

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/class.rst

    PlotStyle
"""

# PlotStyle is imported by .drawers and .plotters. Skip PlotStyle import for isort to prevent circular
# import.
from .style import PlotStyle  # isort:skip
from .drawers import BaseDrawer, MplDrawer
from .plotters import BasePlotter, CurvePlotter
