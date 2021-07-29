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
.. autosummary::
    :toctree: ../stubs/

    CurveAnalysis
    SeriesDef
    CurveData
    FitData

Functions
=========

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
    fit_function.exponential_decay
    fit_function.gaussian
    fit_function.sin

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

Utility
*******
.. autosummary::
    :toctree: ../stubs/

    get_fitval
"""
from .curve_analysis import CurveAnalysis
from .curve_data import CurveData, SeriesDef, FitData
from .curve_fit import (
    curve_fit,
    multi_curve_fit,
    process_curve_data,
    process_multi_curve_data,
)
from .visualization import plot_curve_fit, plot_errorbar, plot_scatter
from .utils import get_fitval
from . import guess
from . import fit_function
