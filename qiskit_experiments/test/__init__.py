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
===============================================
Test Utilities (:mod:`qiskit_experiments.test`)
===============================================

.. currentmodule:: qiskit_experiments.test

This module contains classes and functions that are used to enable testing
of Qiskit Experiments. It's primarily composed of mock backends that
simulate real backends.

.. _backends:

Fake Backends
=============

Mock backends for running simulated jobs.

.. autosummary::
    :toctree: ../stubs/

    MockIQBackend
    MockIQParallelBackend
    T2HahnBackend
    NoisyDelayAerBackend

Helpers
=======

Helper classes for supporting test functionality.

.. autosummary::
    :toctree: ../stubs/

    MockIQExperimentHelper
    MockIQParallelExperimentHelper

"""

from .utils import FakeJob
from .mock_iq_backend import MockIQBackend, MockIQParallelBackend
from .mock_iq_helpers import MockIQExperimentHelper, MockIQParallelExperimentHelper
from .noisy_delay_aer_simulator import NoisyDelayAerBackend
from .t2hahn_backend import T2HahnBackend
from .fake_service import FakeService
