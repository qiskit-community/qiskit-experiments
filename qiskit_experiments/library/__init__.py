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

This provides a set of standard experiments in three groups, namely characterization,
calibration, and verification.
Though Qiskit Experiments is a framework that is agnostic to the underlying hardware architecture,
our collection may lean towards superconducting quantum processors which IBM develops.

We are still actively developing this library. If you cannot find experiment you need
and if you find that is quite useful, please feel free to write a feature request in our
`Github <https://github.com/Qiskit/qiskit-experiments/issues>`_.


.. _verification:

Verification Experiments
========================

.. epigraph::

    "Verification involves verifying that a control operation implements a
    desired ideal operation to within a specified precision.

    Validation is demonstrating that a quantum information processor can solve specific problems."

    -- Joel Wallman, Steven Flammia and Ian Hincks

This group provides a set of experiments for verification and validation of quantum processor.

.. autosummary::
    :toctree: ../stubs/

    ~randomized_benchmarking.StandardRB
    ~randomized_benchmarking.InterleavedRB
    ~tomography.StateTomography
    ~tomography.ProcessTomography
    ~quantum_volume.QuantumVolume

.. _characterization:

Characterization Experiments
============================

.. epigraph::

    "Characterization means determining the effect of control operations on a quantum system,
    and the nature of external noise acting on the quantum system."

    -- Joel Wallman, Steven Flammia and Ian Hincks

This group provides a set of experiments for characterizing a quantum processor.
Some experiments may be used for the calibration as well.

.. autosummary::
    :toctree: ../stubs/

    ~characterization.T1
    ~characterization.T2Ramsey
    ~characterization.QubitSpectroscopy
    ~characterization.EFSpectroscopy

.. _calibration:

Calibration Experiments
=======================

This group provides a set of experiments for creating a quantum gate.
These experiments are usually run with a
:py:class:`~qiskit_experiments.calibration_management.Calibrations`
class instance to manage parameters and pulse schedules.
See :doc:`/tutorials/calibrating_armonk` for example.

.. autosummary::
    :toctree: ../stubs/

    ~calibration.DragCal
    ~calibration.Rabi
    ~calibration.EFRabi
    ~calibration.FineAmplitude
    ~calibration.FineXAmplitude
    ~calibration.FineSXAmplitude

"""
from .calibration import DragCal, Rabi, EFRabi, FineAmplitude, FineXAmplitude, FineSXAmplitude
from .characterization import T1, T2Ramsey, QubitSpectroscopy, EFSpectroscopy
from .randomized_benchmarking import StandardRB, InterleavedRB
from .tomography import StateTomography, ProcessTomography
from .quantum_volume import QuantumVolume

# Experiment Sub-modules
from . import calibration
from . import characterization
from . import randomized_benchmarking
from . import tomography
from . import quantum_volume
