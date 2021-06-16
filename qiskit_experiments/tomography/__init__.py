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

    StateTomography
    ProcessTomography


Analysis
========

.. autosummary::
    :toctree: ../stubs/

    StateTomographyAnalysis
    ProcessTomographyAnalysis


Tomography Fitters
==================

Fitter methods for tomography analysis

.. autosummary::
    :toctree: ../stubs/

    fitters.linear_inversion
    fitters.scipy_gaussian_lstsq
    fitters.scipy_linear_lstsq
    fitters.cvxpy_gaussian_lstsq
    fitters.cvxpy_linear_lstsq

.. warning::
    The API for tomography fitters is still under development so may change
    in a future release.


Basis Classes
=============

Tensor product basis classes

.. autosummary::
    :toctree: ../stubs/

    basis.TomographyMeasurementBasis
    basis.TomographyPreparationBasis
    basis.FitterMeasurementBasis
    basis.FitterPreparationBasis
    basis.PauliPreparationBasis
    basis.Pauli6PreparationBasis
    basis.PauliMeasurementBasis

Abstract base classes

.. autosummary::
    :toctree: ../stubs/

    basis.BaseTomographyMeasurementBasis
    basis.BaseTomographyPreparationBasis
    basis.BaseFitterMeasurementBasis
    basis.BaseFitterPreparationBasis
"""

# Experiment Classes
from .qst_experiment import StateTomography, StateTomographyAnalysis
from .qpt_experiment import ProcessTomography, ProcessTomographyAnalysis

# Basis Classes
from . import basis
from . import fitters
