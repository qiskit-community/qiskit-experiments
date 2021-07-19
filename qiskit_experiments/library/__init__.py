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
======================================================
Experiment Library (:mod:`qiskit_experiments.library`)
======================================================

.. currentmodule:: qiskit_experiments.library

Verification Experiments
========================

.. autosummary::
    :toctree: ../stubs/

    StandardRB
    InterleavedRB
    StateTomography
    ProcessTomography
    QuantumVolume

Characterization Experiments
============================

.. autosummary::
    :toctree: ../stubs/

    T1
    T2Ramsey
    QubitSpectroscopy
    EFSpectroscopy

Calibration Experiments
=======================

.. autosummary::
    :toctree: ../stubs/

    DragCal
    Rabi
    EFRabi
    QubitSpectroscopy
    EFSpectroscopy

Composite Experiments
=====================

.. autosummary::
    :toctree: ../stubs/

    ParallelExperiment
    BatchExperiment
"""

from qiskit_experiments.quantum_volume import QuantumVolume
from qiskit_experiments.characterization import T1, T2Ramsey, QubitSpectroscopy, EFSpectroscopy
from qiskit_experiments.calibration import DragCal, Rabi, EFRabi
from qiskit_experiments.composite import ParallelExperiment, BatchExperiment
from .randomized_benchmarking import StandardRB, InterleavedRB
from .tomography import StateTomography, ProcessTomography
