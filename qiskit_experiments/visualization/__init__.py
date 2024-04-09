# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022, 2023.
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

The visualization module provides plotting functionality for creating figures from
experiment and analysis results. This includes plotter and drawer classes to plot data
in :class:`.CurveAnalysis` and its subclasses. Plotters inherit from
:class:`BasePlotter` and define a type of figure that may be generated from experiment
or analysis data. For example, the results from :class:`.CurveAnalysis`---or any other
experiment where results are plotted against a single parameter (i.e., :math:`x`)---can
be plotted using the :class:`CurvePlotter` class, which plots X-Y-like values.

These plotter classes act as a bridge (from the common bridge pattern in software
development) between analysis classes (or even users) and plotting backends such as
Matplotlib. Drawers are the backends, with a common interface defined in
:class:`BaseDrawer`. Though Matplotlib is the only officially supported plotting backend
in Qiskit Experiments (i.e., through :class:`MplDrawer`), custom drawers can be
implemented by users to use alternative backends. As long as the backend is a subclass
of :class:`BaseDrawer`, and implements all the necessary functionality, all plotters
should be able to generate figures with the alternative backend.

To collate style parameters together, plotters and drawers store instances of the
:class:`PlotStyle` class. These instances can be merged and updated, so that default
styles can have their values overwritten.

Plotter Library
===============

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/plotter.rst

    BasePlotter
    CurvePlotter
    IQPlotter

Drawer Library
==============

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/drawer.rst

    BaseDrawer
    MplDrawer

Plotting Style
==============

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/class.rst

    PlotStyle
"""

from .drawers import BaseDrawer, MplDrawer
from .plotters import BasePlotter, CurvePlotter, IQPlotter
from .style import PlotStyle
