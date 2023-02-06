# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for BackendData."""

from test.base import QiskitExperimentsTestCase

from ddt import data, ddt, unpack
from qiskit import QiskitError
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target

from qiskit_experiments.framework import BackendData


class MinimalBackend(BackendV2):
    """Class for testing a backend with minimal data"""

    def __init__(self, target: Target):
        super().__init__()
        self._target = target

    @property
    def target(self) -> Target:
        """Target set by the user"""
        return self._target

    @property
    def max_circuits(self):
        """Maximum circuits to run at once"""
        return 100

    @classmethod
    def _default_options(cls):
        return Options()

    def run(self, run_input, **options):
        """Empty method to satisfy abstract base class"""
        pass


class TestBackendData(QiskitExperimentsTestCase):
    """Test BackendTiming"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Creating a complete fake backend is difficult so we use one from
        # terra. Just to be safe, we check that the properties we care about
        # for these tests are never changed from what the tests assume.
        backend = FakeNairobiV2()
        target = backend.target
        acquire_alignment = getattr(target, "acquire_alignment", target.aquire_alignment)
        assumptions = (
            (abs(target.dt * 4.5e9 - 1) < 1e-6)
            and acquire_alignment == 16
            and target.pulse_alignment == 1
            and target.min_length == 64
            and target.granularity == 16
        )
        if not assumptions:  # pragma: no cover
            raise ValueError("FakeNairobiV2 properties have changed!")

        cls.acquire_alignment = acquire_alignment
        cls.dt = target.dt
        cls.granularity = target.granularity
        cls.min_length = target.min_length
        cls.pulse_alignment = target.pulse_alignment

    def test_empty_target(self):
        """Test BackendData gives no error for an empty target"""
        backend = MinimalBackend(Target())
