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
=================================================================
Qiskit Experiments Test Utilties (:mod:`qiskit_experiments.test`)
=================================================================

.. currentmodule:: qiskit_experiments.test

This module contains classes and functions that are used to enable testing
of Qiskit Experiments. It's primarily composed of fake and mock backends that
act like a normal :class:`~qiskit.providers.BackendV1` for a real device but
instead call a simulator internally.

.. autosummary::
    :toctree: ../stubs/

    MockIQBackend
    T1Backend
    T2RamseyBackend
    FakeJob
    FakeService

"""

from .utils import FakeJob
from .mock_iq_backend import MockIQBackend
from .t1_backend import T1Backend
from .t2ramsey_backend import T2RamseyBackend
from .fake_service import FakeService
