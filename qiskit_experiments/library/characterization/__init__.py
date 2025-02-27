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
    T2Hahn
    Tphi
    HalfAngle
    FineAmplitude
    FineXAmplitude
    FineSXAmplitude
    FineZXAmplitude
    RamseyXY
    FineFrequency
    ReadoutAngle
    FineDrag
    FineXDrag
    FineSXDrag
    LocalReadoutError
    CorrelatedReadoutError
    MultiStateDiscrimination
    ZZRamsey


Analysis
========

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/analysis.rst

    T1Analysis
    T1KerneledAnalysis
    T2RamseyAnalysis
    T2HahnAnalysis
    TphiAnalysis
    FineAmplitudeAnalysis
    RamseyXYAnalysis
    ReadoutAngleAnalysis
    LocalReadoutErrorAnalysis
    CorrelatedReadoutErrorAnalysis
    ZZRamseyAnalysis
    MultiStateDiscriminationAnalysis

"""

from .analysis import (
    FineAmplitudeAnalysis,
    RamseyXYAnalysis,
    T2RamseyAnalysis,
    T1Analysis,
    T1KerneledAnalysis,
    T2HahnAnalysis,
    TphiAnalysis,
    ReadoutAngleAnalysis,
    LocalReadoutErrorAnalysis,
    CorrelatedReadoutErrorAnalysis,
    ZZRamseyAnalysis,
    MultiStateDiscriminationAnalysis,
)

from .t1 import T1
from .t2ramsey import T2Ramsey
from .t2hahn import T2Hahn
from .tphi import Tphi
from .half_angle import HalfAngle
from .fine_amplitude import FineAmplitude, FineXAmplitude, FineSXAmplitude, FineZXAmplitude
from .ramsey_xy import RamseyXY
from .fine_frequency import FineFrequency
from .readout_angle import ReadoutAngle
from .fine_drag import FineDrag, FineXDrag, FineSXDrag
from .local_readout_error import LocalReadoutError
from .correlated_readout_error import CorrelatedReadoutError
from .zz_ramsey import ZZRamsey
from .multi_state_discrimination import MultiStateDiscrimination
