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
=============================================================
Tomography Experiments (:mod:`qiskit_experiments.tomography`)
=============================================================

.. currentmodule:: qiskit_experiments.tomography


Experiments
===========
.. autosummary::
    :toctree: ../stubs/

    StateTomographyExperiment
    ProcessTomographyExperiment


Analysis
========

.. autosummary::
    :toctree: ../stubs/

    TomographyAnalysis

Fitter Functions
================

.. autosummary::
    :toctree: ../stubs/

    fitters.linear_inversion
    fitters.scipy_gaussian_lstsq
    fitters.scipy_linear_lstsq
    fitters.cvxpy_gaussian_lstsq

Bases Classes
=============

.. autosummary::
    :toctree: ../stubs/

    basis.TomographyBasis
    basis.CircuitBasis
    basis.FitterBasis
    basis.PauliPreparationBasis
    basis.Pauli6PreparationBasis
    basis.PauliMeasurementBasis
"""

# Experiment Classes
from .qst_experiment import StateTomographyExperiment
from .qpt_experiment import ProcessTomographyExperiment
from .tomography_analysis import TomographyAnalysis

# Basis Classes
from . import basis
from . import fitters
