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
=======================================================================
Example Documentation (:mod:`qiskit_experiments.documentation.example`)
=======================================================================

.. currentmodule:: qiskit_experiments.documentation.example


.. warning::

    This module is just an example for documentation. Do not import.

.. note::

    Under the autosummary directive you need to set template to trigger custom documentation.


Experiments
===========
.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    DocumentedExperiment

Analysis
========
.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/analysis.rst

    DocumentedCurveAnalysis

"""

from .example_experiment import DocumentedExperiment, DocumentedCurveAnalysis
