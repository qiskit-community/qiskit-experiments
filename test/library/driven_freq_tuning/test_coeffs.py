# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for Stark coefficients utility."""

from test.base import QiskitExperimentsTestCase

from ddt import ddt, named_data, data, unpack
import numpy as np

from qiskit_experiments.library.driven_freq_tuning import coefficients as util
from qiskit_experiments.test import FakeService


@ddt
class TestStarkUtil(QiskitExperimentsTestCase):
    """Test cases for Stark coefficient utilities."""

    def test_coefficients(self):
        """Test getting group of coefficients."""
        coeffs = util.StarkCoefficients(
            pos_coef_o1=1e6,
            pos_coef_o2=2e6,
            pos_coef_o3=3e6,
            neg_coef_o1=-1e6,
            neg_coef_o2=-2e6,
            neg_coef_o3=-3e6,
            offset=0,
        )
        self.assertListEqual(coeffs.positive_coeffs(), [3e6, 2e6, 1e6])
        self.assertListEqual(coeffs.negative_coeffs(), [-3e6, -2e6, -1e6])

    def test_roundtrip_coefficients(self):
        """Test serializing and deserializing the coefficient object."""
        coeffs = util.StarkCoefficients(
            pos_coef_o1=1e6,
            pos_coef_o2=2e6,
            pos_coef_o3=3e6,
            neg_coef_o1=-1e6,
            neg_coef_o2=-2e6,
            neg_coef_o3=-3e6,
            offset=0,
        )
        self.assertRoundTripSerializable(coeffs)

    @named_data(
        ["ordinary", 5e6, 200e6, -50e6, 5e6, -180e6, -40e6, 100e3],
        ["asymmetric_inflection_1st_ord", 10e6, 200e6, -20e6, -50e6, -180e6, -20e6, -10e6],
        ["inflection_3st_ord", 10e6, 200e6, -80e6, 80e6, -180e6, -200e6, 100e3],
    )
    @unpack
    def test_roundtrip_convert_freq_amp(
        self,
        pos_o1: float,
        pos_o2: float,
        pos_o3: float,
        neg_o1: float,
        neg_o2: float,
        neg_o3: float,
        offset: float,
    ):
        """Test round-trip conversion between frequency shift and Stark amplitude."""
        coeffs = util.StarkCoefficients(
            pos_coef_o1=pos_o1,
            pos_coef_o2=pos_o2,
            pos_coef_o3=pos_o3,
            neg_coef_o1=neg_o1,
            neg_coef_o2=neg_o2,
            neg_coef_o3=neg_o3,
            offset=offset,
        )
        target_freqs = np.linspace(-70e6, 70e6, 11)
        test_amps = coeffs.convert_freq_to_amp(target_freqs)
        test_freqs = coeffs.convert_amp_to_freq(test_amps)

        np.testing.assert_array_almost_equal(test_freqs, target_freqs, decimal=2)

    @data(
        [-0.5, 0.5],
        [-0.9, 0.9],
        [0.25, 1.0],
    )
    @unpack
    def test_calculate_min_max_shift(self, min_amp, max_amp):
        """Test estimating maximum frequency shift within given Stark amplitude budget."""

        # These coefficients induce inflection points around ±0.75, for testing
        coeffs = util.StarkCoefficients(
            pos_coef_o1=10e6,
            pos_coef_o2=100e6,
            pos_coef_o3=-90e6,
            neg_coef_o1=80e6,
            neg_coef_o2=-180e6,
            neg_coef_o3=-200e6,
            offset=100e3,
        )
        # This numerical solution is correct up to amp resolution of 0.001
        nop = int((max_amp - min_amp) / 0.001)
        amps = np.linspace(min_amp, max_amp, nop)
        freqs = coeffs.convert_amp_to_freq(amps)

        # This finds strict solution, unless it has a bug
        min_freq, max_freq = coeffs.find_min_max_frequency(
            min_amp=min_amp,
            max_amp=max_amp,
        )

        # Allow 1kHz tolerance because ref is approximate value
        self.assertAlmostEqual(min_freq, np.min(freqs), delta=1e3)
        self.assertAlmostEqual(max_freq, np.max(freqs), delta=1e3)

    def test_get_coeffs_from_service(self):
        """Test retrieve the saved Stark coefficients from the experiment service."""
        mock_experiment_id = "6453f3d1-04ef-4e3b-82c6-1a92e3e066eb"
        mock_result_id = "d067ae34-96db-4e8e-adc8-030305d3d404"
        mock_backend = "mock_backend"

        ref_coeffs = util.StarkCoefficients(
            pos_coef_o1=1e6,
            pos_coef_o2=2e6,
            pos_coef_o3=3e6,
            neg_coef_o1=-1e6,
            neg_coef_o2=-2e6,
            neg_coef_o3=-3e6,
            offset=0,
        )

        service = FakeService()
        service.create_experiment(
            experiment_type="StarkRamseyXYAmpScan",
            backend_name=mock_backend,
            experiment_id=mock_experiment_id,
        )
        service.create_analysis_result(
            experiment_id=mock_experiment_id,
            result_data={"value": ref_coeffs},
            result_type="stark_coefficients",
            device_components=["Q0"],
            tags=[],
            quality="Good",
            verified=False,
            result_id=mock_result_id,
        )

        retrieved = util.retrieve_coefficients_from_service(
            service=service,
            backend_name=mock_backend,
            qubit=0,
        )

        self.assertEqual(retrieved, ref_coeffs)

    def test_get_coeffs_no_data(self):
        """Test raises when Stark coefficients don't exist in the result database."""
        mock_backend = "mock_backend"

        service = FakeService()

        with self.assertRaises(RuntimeError):
            util.retrieve_coefficients_from_service(
                service=service,
                backend_name=mock_backend,
                qubit=0,
            )
