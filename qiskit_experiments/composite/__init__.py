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
===========================================================
Composite Experiments (:mod:`qiskit_experiments.composite`)
===========================================================

.. currentmodule:: qiskit_experiments.composite

Experiments
===========
.. autosummary::
    :toctree: ../stubs/

    ParallelExperiment
    BatchExperiment


Analysis
========

.. autosummary::
    :toctree: ../stubs/

    CompositeAnalysis


Experiment Data
===============

.. autosummary::
    :toctree: ../stubs/

    CompositeExperimentData
"""

# Base classes
from .composite_experiment_data import CompositeExperimentData
from .composite_analysis import CompositeAnalysis

# Composite experiment classes
from .parallel_experiment import ParallelExperiment
from .batch_experiment import BatchExperiment
