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

"""Tests for BackendTiming."""

from test.base import QiskitExperimentsTestCase

from ddt import data, ddt, unpack
from qiskit import QiskitError
from qiskit.providers.fake_provider import FakeNairobiV2

from qiskit_experiments.framework import BackendTiming


@ddt
class TestBackendTiming(QiskitExperimentsTestCase):
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

    @data((True, "s"), (False, "dt"))
    @unpack
    def test_delay_unit(self, null_dt, result):
        """Test delay unit matches dt"""
        backend = FakeNairobiV2()
        if null_dt:
            backend.target.dt = None
        timing = BackendTiming(backend)
        self.assertEqual(timing.delay_unit, result)

    @data([14, 16], [16, 16], [18, 16], [64.5, 64])
    @unpack
    def test_circuit_delay(self, samples_in, samples_out):
        """Test circuit delay calculation"""
        time = self.dt * samples_in

        backend = FakeNairobiV2()
        timing = BackendTiming(backend)
        self.assertEqual(timing.circuit_delay(time), samples_out)

    def test_circuit_delay_no_dt(self):
        """Test circuit delay when dt is None"""
        time = self.dt * 16

        backend = FakeNairobiV2()
        backend.target.dt = None
        timing = BackendTiming(backend)
        self.assertEqual(timing.circuit_delay(time), time)

    @data([14, 16], [16, 16], [18, 16], [64.5, 64])
    @unpack
    def test_schedule_delay(self, samples_in, samples_out):
        """Test schedule delay calculation"""
        time = self.dt * samples_in

        backend = FakeNairobiV2()
        timing = BackendTiming(backend)
        self.assertEqual(timing.schedule_delay(time), samples_out)

    @data([12, 64], [65, 64], [79, 80], [83, 80])
    @unpack
    def test_pulse_samples(self, samples_in, samples_out):
        """Test pulse samples calculation"""
        time = self.dt * samples_in

        backend = FakeNairobiV2()
        timing = BackendTiming(backend)
        self.assertEqual(timing.pulse_samples(time), samples_out)

    @data([12, 64], [65, 64], [79, 80], [83, 80], [80.5, 80])
    @unpack
    def test_round_pulse_samples(self, samples_in, samples_out):
        """Test round_pulse_samples calculation"""
        backend = FakeNairobiV2()
        timing = BackendTiming(backend)
        self.assertEqual(timing.round_pulse_samples(samples_in), samples_out)

    def test_delay_time(self):
        """Test delay_time calculation"""
        time_in = self.dt * 16.1
        time_out = self.dt * 16

        backend = FakeNairobiV2()
        timing = BackendTiming(backend)
        self.assertAlmostEqual(timing.delay_time(time_in), time_out, delta=1e-6 * self.dt)

    def test_delay_time_no_dt(self):
        """Test delay time calculation when dt is None"""
        time_in = self.dt * 16.1
        time_out = time_in

        backend = FakeNairobiV2()
        backend.target.dt = None
        timing = BackendTiming(backend)
        self.assertAlmostEqual(timing.delay_time(time_in), time_out, delta=1e-6 * self.dt)

    def test_pulse_time(self):
        """Test pulse_time calculation"""
        time_in = self.dt * 85.1
        time_out = self.dt * 80

        backend = FakeNairobiV2()
        timing = BackendTiming(backend)
        self.assertAlmostEqual(timing.pulse_time(time_in), time_out, delta=1e-6 * self.dt)

    def test_no_dt_errors(self):
        """Test methods that don't work when dt is None raise exceptions"""
        backend = FakeNairobiV2()
        backend.target.dt = None
        timing = BackendTiming(backend)

        time = self.dt * 81

        with self.assertRaises(QiskitError):
            timing.schedule_delay(time)
        with self.assertRaises(QiskitError):
            timing.pulse_samples(time)
        with self.assertRaises(QiskitError):
            timing.pulse_time(time)

    def test_unexpected_pulse_alignment(self):
        """Test that a weird pulse_alignment parameter is caught"""
        backend = FakeNairobiV2()
        backend.target.pulse_alignment = 33
        timing = BackendTiming(backend)
        with self.assertRaises(QiskitError):
            timing.round_pulse_samples(81)

    def test_unexpected_acquire_alignment(self):
        """Test that a weird acquire_alignment parameter is caught"""
        backend = FakeNairobiV2()
        try:
            backend.target.acquire_alignment = 33
        except AttributeError:
            backend.target.aquire_alignment = 33
        timing = BackendTiming(backend)
        with self.assertRaises(QiskitError):
            timing.round_pulse_samples(81)
