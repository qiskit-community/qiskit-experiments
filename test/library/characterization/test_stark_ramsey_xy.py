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

"""Test Stark Ramsey XY experiments."""

from test.base import QiskitExperimentsTestCase

import numpy as np
from qiskit import pulse
from qiskit.providers.fake_provider import FakeHanoiV2

from qiskit_experiments.library import StarkRamseyXY


class TestStarkRamseyXY(QiskitExperimentsTestCase):
    """Test case for the Stark Ramsey XY experiment.

    Because the analysis is identical to the RamseyXY, integration test and
    test for analysis class is skipped here. These must be covered by
    test/library/calibration/test_ramsey_xy.py
    """

    def test_calibration_with_positive_amp(self):
        """Test Stark frequency shift chooses the proper sign with positive amplitude.

        * Frequency shift must be negative.
        * Stark tone amplitude shift must be positive.
        """
        backend = FakeHanoiV2()
        exp = StarkRamseyXY(
            physical_qubits=[0],
            stark_amp=0.1,  # positive amplitude
            backend=backend,
            stark_sigma=15e-9,
            stark_risefall=2,
            stark_freq_offset=80e6,
        )
        param_ram_x, _ = exp.parameterized_circuits()
        freq = backend.qubit_properties(0).frequency - 80e6  # negative frequency shift
        granularity = backend.target.granularity
        dt = backend.dt
        duration = granularity * int(round(4 * 15e-9 / dt / granularity))
        sigma = duration / 4

        with pulse.build() as ref_schedule:
            pulse.set_frequency(freq, pulse.DriveChannel(0))
            pulse.play(
                pulse.Gaussian(duration=duration, amp=0.1, sigma=sigma), pulse.DriveChannel(0)
            )

        test_schedule = param_ram_x.calibrations["StarkV"][(0,), ()]
        self.assertEqual(test_schedule, ref_schedule)

    def test_calibration_with_negative_amp(self):
        """Test Stark frequency shift chooses the proper sign with negative amplitude.

        * Frequency shift must be positive.
        * Stark tone amplitude shift must be positive.
        """
        backend = FakeHanoiV2()
        exp = StarkRamseyXY(
            physical_qubits=[0],
            stark_amp=-0.1,  # negative amplitude
            backend=backend,
            stark_sigma=15e-9,
            stark_risefall=2,
            stark_freq_offset=80e6,
        )
        param_ram_x, _ = exp.parameterized_circuits()
        freq = backend.qubit_properties(0).frequency + 80e6  # positive frequency shift
        granularity = backend.target.granularity
        dt = backend.dt
        duration = granularity * int(round(4 * 15e-9 / dt / granularity))
        sigma = duration / 4

        with pulse.build() as ref_schedule:
            pulse.set_frequency(freq, pulse.DriveChannel(0))
            pulse.play(
                pulse.Gaussian(duration=duration, amp=0.1, sigma=sigma), pulse.DriveChannel(0)
            )

        test_schedule = param_ram_x.calibrations["StarkV"][(0,), ()]
        self.assertEqual(test_schedule, ref_schedule)

    def test_gen_delays(self):
        """Test generating delays with experiment options."""
        min_freq = 1e6
        max_freq = 50e6
        exp = StarkRamseyXY(
            physical_qubits=[0],
            stark_amp=0.1,
            min_freq=min_freq,
            max_freq=max_freq,
        )
        test_delays = exp.delays()
        ref_delays = np.arange(0, 1 / min_freq, 1 / max_freq / 2)
        np.testing.assert_array_equal(test_delays, ref_delays)

    def test_circuit_valid_delays(self):
        """Test Stark tone durations are valid."""
        backend = FakeHanoiV2()
        dt = backend.dt
        exp = StarkRamseyXY(
            physical_qubits=[0],
            stark_amp=0.1,
            backend=backend,
            delays=np.linspace(0, 10e-6, 5),
        )
        circs = exp.circuits()

        for circ in circs:
            stark_v = next(iter(circ.calibrations["StarkV"].values()))
            self.assertEqual(stark_v.duration % backend.target.granularity, 0)
            stark_u = next(iter(circ.calibrations["StarkU"].values()))
            self.assertEqual(stark_u.duration % backend.target.granularity, 0)

            stark_u_dur = stark_u.duration
            stark_v_dur = stark_v.duration
            delay_dt = stark_u_dur - stark_v_dur
            self.assertAlmostEqual(circ.metadata["xval"], delay_dt * dt)

    def test_stark_offset_always_positive(self):
        """Test raise error by definition when the offset is negative."""
        exp = StarkRamseyXY(physical_qubits=[0], stark_amp=0.1)
        with self.assertRaises(ValueError):
            exp.set_experiment_options(stark_freq_offset=-10e6)

    def test_circuit_roundtrip_serializable(self):
        """Test circuit round trip JSON serialization"""
        # backend=None due to bug in serialization of backend obj.
        backend = FakeHanoiV2()
        exp = StarkRamseyXY(
            physical_qubits=[0],
            stark_amp=0.1,
            backend=backend,
            delays=np.linspace(0, 10e-6, 5),
        )
        self.assertRoundTripSerializable(exp.circuits())
