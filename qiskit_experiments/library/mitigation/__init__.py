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
Readout Mitigation Experiments (:mod:`qiskit_experiments.library.mitigation`)
===============================================================================================

.. currentmodule:: qiskit_experiments.library.mitigation

Experiment
===========
.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    ReadoutMitigationExperiment


Analysis
========

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/analysis.rst

    CorrelatedMitigationAnalysis
    LocalMitigationAnalysis
"""
from .mitigation_experiment import ReadoutMitigationExperiment
from .mitigation_analysis import CorrelatedMitigationAnalysis
from .mitigation_analysis import LocalMitigationAnalysis
