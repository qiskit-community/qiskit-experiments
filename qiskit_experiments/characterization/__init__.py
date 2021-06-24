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
=========================================================================
Characterization Experiments (:mod:`qiskit_experiments.characterization`)
=========================================================================

.. currentmodule:: qiskit_experiments.characterization

Experiments
===========
.. autosummary::
    :toctree: ../stubs/

    T1
    T2Ramsey
    QubitSpectroscopy


Analysis
========

.. autosummary::
    :toctree: ../stubs/

    T1Analysis
    T2RamseyAnalysis
    SpectroscopyAnalysis
"""
from .t1 import T1
from .t1_analysis import T1Analysis
from .qubit_spectroscopy import QubitSpectroscopy, SpectroscopyAnalysis
from .ef_spectroscopy import EFSpectroscopy
from .t2ramsey import T2Ramsey
from .t2ramsey_analysis import T2RamseyAnalysis
