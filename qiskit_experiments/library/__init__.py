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

A library of quantum characterization, calibration and verification
experiments for calibrating and benchmarking quantum devices. See
:mod:`qiskit_experiments.framework` for general information on the framework
for running experiments.


.. _verification:

Verification Experiments
========================

Experiments for verification and validation of quantum devices.

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    ~randomized_benchmarking.StandardRB
    ~randomized_benchmarking.InterleavedRB
    ~randomized_benchmarking.LayerFidelity
    ~tomography.TomographyExperiment
    ~tomography.StateTomography
    ~tomography.ProcessTomography
    ~tomography.MitigatedStateTomography
    ~tomography.MitigatedProcessTomography
    ~quantum_volume.QuantumVolume

.. _characterization single qubit:

Characterization Experiments: Single Qubit
==========================================

Experiments for characterization of properties of individual qubits.

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    ~characterization.T1
    ~characterization.T2Hahn
    ~characterization.T2Ramsey
    ~characterization.Tphi
    ~characterization.HalfAngle
    ~characterization.FineAmplitude
    ~characterization.FineXAmplitude
    ~characterization.FineSXAmplitude
    ~characterization.RamseyXY
    ~characterization.FineFrequency
    ~characterization.ReadoutAngle
    ~characterization.FineDrag
    ~characterization.FineXDrag
    ~characterization.FineSXDrag
    ~characterization.MultiStateDiscrimination

.. _characterization two qubits:

Characterization Experiments: Two Qubits
========================================

Experiments for characterization of properties of two qubit interactions.

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    ~characterization.ZZRamsey
    ~characterization.FineZXAmplitude

.. _characterization-mitigation:

Characterization Experiments: Mitigation
========================================

Experiments for characterizing and mitigating readout error.

.. autosummary::
    :toctree: ../stubs/
    :template: autosummary/experiment.rst

    ~characterization.LocalReadoutError
    ~characterization.CorrelatedReadoutError

"""
from .characterization import (
    T1,
    T2Hahn,
    T2Ramsey,
    Tphi,
    FineDrag,
    FineXDrag,
    FineSXDrag,
    HalfAngle,
    FineAmplitude,
    FineXAmplitude,
    FineSXAmplitude,
    FineZXAmplitude,
    RamseyXY,
    FineFrequency,
    ReadoutAngle,
    LocalReadoutError,
    CorrelatedReadoutError,
    ZZRamsey,
    MultiStateDiscrimination,
)
from .randomized_benchmarking import StandardRB, InterleavedRB
from .tomography import (
    TomographyExperiment,
    StateTomography,
    ProcessTomography,
    MitigatedStateTomography,
    MitigatedProcessTomography,
)
from .quantum_volume import QuantumVolume

# Experiment Sub-modules
from . import characterization
from . import randomized_benchmarking
from . import tomography
from . import quantum_volume
