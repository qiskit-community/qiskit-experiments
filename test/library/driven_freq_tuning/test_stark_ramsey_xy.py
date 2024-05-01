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

"""Test Stark Ramsey XY experiments."""

from test.base import QiskitExperimentsTestCase

import numpy as np
from ddt import ddt, unpack, named_data

from qiskit import pulse
from qiskit_ibm_runtime.fake_provider import FakeHanoiV2

from qiskit_experiments.library import StarkRamseyXY, StarkRamseyXYAmpScan
from qiskit_experiments.library.driven_freq_tuning.ramsey_amp_scan_analysis import (
    StarkRamseyXYAmpScanAnalysis,
)
from qiskit_experiments.library.driven_freq_tuning.coefficients import StarkCoefficients
from qiskit_experiments.framework import ExperimentData


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
        test_delays = exp.parameters()
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
        self.assertRoundTripSerializable(exp._transpiled_circuits())


@ddt
class TestStarkRamseyXYAmpScan(QiskitExperimentsTestCase):
    """Test case for StarkRamseyXYAmpScan experiment."""

    def test_frequency_shift_direction(self):
        """Check frequency shift direction reflects abstracted amplitude policy.

        When amplitude is positive (negative), it must induce positive (negative) Stark shift
        for simplicity. To achieve this, the spectral location of the tone must be
        lower (higher) than the qubit frequency f01. Tone amplitude is always positive.
        """
        backend = FakeHanoiV2()
        exp = StarkRamseyXYAmpScan(
            physical_qubits=[0],
            backend=backend,
            stark_amps=[-0.1, 0.1],
        )
        f01 = backend.qubit_properties(0).frequency

        circs = exp.circuits()

        # Check circuit metadata
        circs0_x_starkv = next(iter(circs[0].calibrations["StarkV"].values()))
        self.assertDictEqual(circs[0].metadata, {"xval": -0.1, "series": "X", "direction": "neg"})
        circs0_y_starkv = next(iter(circs[1].calibrations["StarkV"].values()))
        self.assertDictEqual(circs[1].metadata, {"xval": -0.1, "series": "Y", "direction": "neg"})
        circs1_x_starkv = next(iter(circs[2].calibrations["StarkV"].values()))
        self.assertDictEqual(circs[2].metadata, {"xval": 0.1, "series": "X", "direction": "pos"})
        circs1_y_starkv = next(iter(circs[3].calibrations["StarkV"].values()))
        self.assertDictEqual(circs[3].metadata, {"xval": 0.1, "series": "Y", "direction": "pos"})

        # Check frequency shift
        circ0_x_set_freq = circs0_x_starkv.blocks[0].frequency
        circ0_y_set_freq = circs0_y_starkv.blocks[0].frequency
        # This must induce negative frequency shift.
        # This tone frequency must be greater than f01.
        self.assertGreater(circ0_x_set_freq, f01)
        self.assertGreater(circ0_y_set_freq, f01)

        circ1_x_set_freq = circs1_x_starkv.blocks[0].frequency
        circ1_y_set_freq = circs1_y_starkv.blocks[0].frequency
        # This must induce positive frequency shift.
        # This tone frequency must be lower than f01.
        self.assertLess(circ1_x_set_freq, f01)
        self.assertLess(circ1_y_set_freq, f01)

        # Check amplitude is always positive
        circ0_x_set_amp = circs0_x_starkv.blocks[1].pulse.parameters["amp"]
        circ0_y_set_amp = circs0_y_starkv.blocks[1].pulse.parameters["amp"]
        self.assertEqual(circ0_x_set_amp, 0.1)
        self.assertEqual(circ0_y_set_amp, 0.1)
        circ1_x_set_amp = circs1_x_starkv.blocks[1].pulse.parameters["amp"]
        circ1_y_set_amp = circs1_y_starkv.blocks[1].pulse.parameters["amp"]
        self.assertEqual(circ1_x_set_amp, 0.1)
        self.assertEqual(circ1_y_set_amp, 0.1)

    def test_circuit_valid_delays(self):
        """Test Stark tone durations are valid."""
        backend = FakeHanoiV2()
        exp = StarkRamseyXYAmpScan(
            physical_qubits=[0],
            backend=backend,
            stark_amps=[0.1],
        )
        circs = exp.circuits()

        for circ in circs:
            stark_v = next(iter(circ.calibrations["StarkV"].values()))
            self.assertEqual(stark_v.duration % backend.target.granularity, 0)
            stark_u = next(iter(circ.calibrations["StarkU"].values()))
            self.assertEqual(stark_u.duration % backend.target.granularity, 0)

    @named_data(
        ["ideal_quadratic", 0.0, 30e6, 0.0, 0.0, -30e6, 0.0, 0.0],
        ["with_all_terms", 15e6, 200e6, -100e6, 15e6, -200e6, -100e6, 300e3],
        ["asymmetric_shift", -20e6, 200e6, -100e6, -15e6, -180e6, -90e6, 200e3],
        ["large_cubic_term", 10e6, 15e6, 30e6, 5e6, -10e6, 40e6, 0.0],
    )
    @unpack
    def test_ramsey_fast_analysis(self, c1p, c2p, c3p, c1n, c2n, c3n, ferr):
        """End-to-end test for Ramsey fast analysis with artificial data."""
        amp = 0.5
        off = 0.5
        rng = np.random.default_rng(seed=123)
        shots = 1000

        xvals = np.linspace(-1.0, 1.0, 101)
        const = 2 * np.pi * 50e-9
        exp_data = ExperimentData()
        exp_data.metadata.update({"stark_length": 50e-9})

        ref_coeffs = StarkCoefficients(
            pos_coef_o1=c1p,
            pos_coef_o2=c2p,
            pos_coef_o3=c3p,
            neg_coef_o1=c1n,
            neg_coef_o2=c2n,
            neg_coef_o3=c3n,
            offset=ferr,
        )
        yvals = ref_coeffs.convert_amp_to_freq(xvals)

        # Generate fake data based on fit model.
        for x, y in zip(xvals, yvals):
            if x >= 0.0:
                direction = "pos"
            else:
                direction = "neg"

            # Add some sampling error
            ramx_count = rng.binomial(shots, amp * np.cos(const * y) + off)
            exp_data.add_data(
                {
                    "counts": {"0": shots - ramx_count, "1": ramx_count},
                    "metadata": {"xval": x, "series": "X", "direction": direction},
                }
            )
            ramy_count = rng.binomial(shots, amp * np.sin(const * y) + off)
            exp_data.add_data(
                {
                    "counts": {"0": shots - ramy_count, "1": ramy_count},
                    "metadata": {"xval": x, "series": "Y", "direction": direction},
                }
            )

        analysis = StarkRamseyXYAmpScanAnalysis()
        analysis.run(exp_data, replace_results=True)
        self.assertExperimentDone(exp_data)

        # Check the fitted parameter can approximate the same polynominal.
        # Note that coefficient values don't need to exactly match as long as
        # frequency shift is predictable.
        # Since the fit model is just an empirical polynomial,
        # comparing coefficients don't physically sound.
        # Curves must be agreed within the tolerance of 1.5 * 1 MHz.
        fit_coeffs = exp_data.analysis_results("stark_coefficients").value
        fit_yvals = fit_coeffs.convert_amp_to_freq(xvals)

        np.testing.assert_array_almost_equal(
            yvals,
            fit_yvals,
            decimal=-6,
            err_msg="Reconstructed phase polynominal doesn't match with the actual phase shift.",
        )
