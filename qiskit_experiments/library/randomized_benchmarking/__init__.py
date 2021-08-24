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
===============================================================================================
Randomized Benchmarking Experiments (:mod:`qiskit_experiments.library.randomized_benchmarking`)
===============================================================================================

.. currentmodule:: qiskit_experiments.library.randomized_benchmarking

Experiments
===========
.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    StandardRB
    InterleavedRB


Analysis
========

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/analysis.rst

    RBAnalysis
    InterleavedRBAnalysis

.. autosummary::
    :toctree: ../stubs/

    RBUtils
"""
from .rb_experiment import StandardRB
from .interleaved_rb_experiment import InterleavedRB
from .rb_analysis import RBAnalysis
from .interleaved_rb_analysis import InterleavedRBAnalysis
from .rb_utils import RBUtils
from .clifford_utils import CliffordUtils
