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
Curve Analysis (:mod:`qiskit_experiments.curve_analysis`)
=========================================================

.. currentmodule:: qiskit_experiments.curve_analysis

Curve analysis provides the analysis base class for a variety of experiments with
a single experimental parameter sweep. This analysis subclasses can override
several class attributes to customize the behavior from data processing to post-processing,
including providing systematic initial guess for parameters tailored to the experiment.


Base Classes
============

.. autosummary::
    :toctree: ../stubs/

    BaseCurveAnalysis
    CurveAnalysis
    CompositeCurveAnalysis

Data Classes
============

.. autosummary::
    :toctree: ../stubs/

    ScatterTable
    CurveFitResult
    ParameterRepr
    FitOptions

Standard Analysis Library
=========================

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/analysis.rst

    BlochTrajectoryAnalysis
    DecayAnalysis
    DampedOscillationAnalysis
    OscillationAnalysis
    GaussianAnalysis
    ErrorAmplificationAnalysis

Fit Functions
=============
.. autosummary::
    :toctree: ../stubs/

    fit_function.cos
    fit_function.cos_decay
    fit_function.exponential_decay
    fit_function.gaussian
    fit_function.sqrt_lorentzian
    fit_function.sin
    fit_function.sin_decay

Initial Guess Estimators
========================
.. autosummary::
    :toctree: ../stubs/

    guess.constant_sinusoidal_offset
    guess.constant_spectral_offset
    guess.exp_decay
    guess.rb_decay
    guess.full_width_half_max
    guess.frequency
    guess.max_height
    guess.min_height
    guess.oscillation_exp_decay

Utilities
=========
.. autosummary::
    :toctree: ../stubs/

    utils.is_error_not_significant
    utils.analysis_result_to_repr
    utils.convert_lmfit_result
    utils.eval_with_uncertainties
    utils.filter_data
    utils.mean_xy_data
    utils.multi_mean_xy_data
    utils.data_sort
    utils.level2_probability
    utils.probability

"""
from .base_curve_analysis import BaseCurveAnalysis
from .curve_analysis import CurveAnalysis
from .composite_curve_analysis import CompositeCurveAnalysis
from .scatter_table import ScatterTable
from .curve_data import (
    CurveFitResult,
    FitOptions,
    ParameterRepr,
)
from .curve_fit import (
    process_curve_data,
    process_multi_curve_data,
)
from . import guess
from . import fit_function
from . import utils

# standard analysis
from .standard_analysis import (
    DecayAnalysis,
    DampedOscillationAnalysis,
    OscillationAnalysis,
    GaussianAnalysis,
    ErrorAmplificationAnalysis,
    BlochTrajectoryAnalysis,
)
