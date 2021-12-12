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
=================================================================================
Characterization Experiments (:mod:`qiskit_experiments.library.characterization`)
=================================================================================

.. currentmodule:: qiskit_experiments.library.characterization

Experiments
===========
.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    T1
    T2Ramsey
    QubitSpectroscopy
    CrossResonanceHamiltonian
    EchoedCrossResonanceHamiltonian
    Rabi
    EFRabi
    HalfAngle
    FineAmplitude
    FineXAmplitude
    FineSXAmplitude
    RamseyXY
    FineFrequency
    RoughDrag
    ReadoutAngle
    FineDrag
    FineXDrag
    FineSXDrag


Analysis
========

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/analysis.rst

    T1Analysis
    T2RamseyAnalysis
    CrossResonanceHamiltonianAnalysis
    DragCalAnalysis
    FineHalfAngleAnalysis
    FineDragAnalysis
    FineAmplitudeAnalysis
    RamseyXYAnalysis
    ReadoutAngleAnalysis
"""

from .analysis import (
    DragCalAnalysis,
    FineHalfAngleAnalysis,
    FineDragAnalysis,
    FineAmplitudeAnalysis,
    RamseyXYAnalysis,
    T2RamseyAnalysis,
    T1Analysis,
    CrossResonanceHamiltonianAnalysis,
    ReadoutAngleAnalysis,
)

from .t1 import T1
from .qubit_spectroscopy import QubitSpectroscopy
from .ef_spectroscopy import EFSpectroscopy
from .t2ramsey import T2Ramsey
from .cr_hamiltonian import CrossResonanceHamiltonian, EchoedCrossResonanceHamiltonian
from .rabi import Rabi, EFRabi
from .half_angle import HalfAngle
from .fine_amplitude import FineAmplitude, FineXAmplitude, FineSXAmplitude
from .ramsey_xy import RamseyXY
from .fine_frequency import FineFrequency
from .drag import RoughDrag
from .readout_angle import ReadoutAngle
from .fine_drag import FineDrag, FineXDrag, FineSXDrag
