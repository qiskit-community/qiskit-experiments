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
=========================================================
Curve Analysis (:mod:`qiskit_experiments.curve_analysis`)
=========================================================

.. currentmodule:: qiskit_experiments.curve_analysis

Classes
=======

These are the base class and internal data structures to implement a curve analysis.

.. autosummary::
    :toctree: ../stubs/

    CurveAnalysis
    SeriesDef
    CurveData
    FitData
    ParameterRepr
    FitOptions

Standard Analysis
=================

These classes provide typical analysis functionality.
These are expected to be reused in multiple experiments.
By overriding default options from the class method :meth:`_default_analysis_options` of
your experiment class, you can still tailor the standard analysis classes to your experiment.

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/analysis.rst

    DecayAnalysis
    DumpedOscillationAnalysis
    OscillationAnalysis
    ResonanceAnalysis
    ErrorAmplificationAnalysis

Functions
=========

These are the helper functions to realize a part of curve fitting functionality.

Curve Fitting
*************

.. autosummary::
    :toctree: ../stubs/

    curve_fit
    multi_curve_fit

Fit Functions
*************
.. autosummary::
    :toctree: ../stubs/

    fit_function.cos
    fit_function.cos_decay
    fit_function.exponential_decay
    fit_function.gaussian
    fit_function.sin
    fit_function.sin_decay
    fit_function.bloch_oscillation_x
    fit_function.bloch_oscillation_y
    fit_function.bloch_oscillation_z

Initial Guess
*************
.. autosummary::
    :toctree: ../stubs/

    guess.constant_sinusoidal_offset
    guess.constant_spectral_offset
    guess.exp_decay
    guess.full_width_half_max
    guess.frequency
    guess.max_height
    guess.min_height
    guess.oscillation_exp_decay

Visualization
*************
.. autosummary::
    :toctree: ../stubs/

    plot_curve_fit
    plot_errorbar
    plot_scatter
"""
from .curve_analysis import CurveAnalysis
from .curve_data import CurveData, SeriesDef, FitData, ParameterRepr, FitOptions
from .curve_fit import (
    curve_fit,
    multi_curve_fit,
    process_curve_data,
    process_multi_curve_data,
)
from .visualization import plot_curve_fit, plot_errorbar, plot_scatter, FitResultPlotters
from . import guess
from . import fit_function

# standard analysis
from .standard_analysis import (
    DecayAnalysis,
    DumpedOscillationAnalysis,
    OscillationAnalysis,
    ResonanceAnalysis,
    ErrorAmplificationAnalysis,
)
