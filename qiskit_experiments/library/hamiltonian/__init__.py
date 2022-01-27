# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
===============================================================================================
Hamiltonian Characterization Experiments (:mod:`qiskit_experiments.library.hamiltonian`)
===============================================================================================

.. currentmodule:: qiskit_experiments.library.hamiltonian

This module provides a set of experiments to characterize qubit Hamiltonians.

HEAT Experiments
================

HEAT stands for `Hamiltonian Error Amplifying Tomography` which amplifies the
dynamics of entangler along the interrogated axis on the target qubit with
the conventional error amplification (ping-pong) technique.

These are the base experiment classes for developer to write own experiments.

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    HeatElement
    BatchHeatHelper

HEAT for ZX Hamiltonian
-----------------------

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    ZXHeat
    ZX90HeatXError
    ZX90HeatYError
    ZX90HeatZError

HEAT Analysis
-------------

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/analysis.rst

    HeatElementAnalysis
    HeatAnalysis

"""

from .heat_base import HeatElement, BatchHeatHelper
from .heat_zx import ZXHeat, ZX90HeatXError, ZX90HeatYError, ZX90HeatZError
from .heat_analysis import HeatElementAnalysis, HeatAnalysis
