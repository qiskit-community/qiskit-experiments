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

"""Spectroscopy tests."""
from test.base import QiskitExperimentsTestCase
from typing import Callable, Tuple, Dict, List, Any
import numpy as np

from qiskit import QuantumCircuit
from qiskit.qobj.utils import MeasLevel

from qiskit_experiments.library import QubitSpectroscopy, EFSpectroscopy
from qiskit_experiments.test.mock_iq_backend import MockIQBackend


def compute_prob_qubit_spectroscopy(
    circuits: List[QuantumCircuit], calc_parameters_list: List[Dict[str, Any]]
) -> List[Dict[str, float]]:
    """Returns the probability based on the parameters provided."""
    freq_offset = calc_parameters_list[0].get("freq_offset", 0.0)
    line_width = calc_parameters_list[0].get("line_width", 2e6)
    output_dict_list = []
    for circuit in circuits:
        probability_output_dict = {}
        freq_shift = next(iter(circuit.calibrations["Spec"]))[1][0]
        delta_freq = freq_shift - freq_offset

        probability_output_dict["1"] = np.abs(1 / (1 + 2.0j * delta_freq / line_width))
        probability_output_dict["0"] = 1 - probability_output_dict["1"]
        output_dict_list.append(probability_output_dict)
    return output_dict_list


class SpectroscopyBackend(MockIQBackend):
    """A simple and primitive backend to test spectroscopy experiments."""

    def __init__(
        self,
        compute_probabilities: Callable[[List[QuantumCircuit], ...], List[Dict[str, float]]],
        line_width: float = 2e6,
        freq_offset: float = 0.0,
        iq_cluster_centers: List[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        iq_cluster_width: List[float] = None,
        calculation_parameters: List[Dict[str, Any]] = None,
    ):
        """Initialize the spectroscopy backend."""

        self._iq_cluster_centers = iq_cluster_centers or [((-1.0, -1.0), (1.0, 1.0))]
        self._iq_cluster_width = iq_cluster_width or [0.2]
        self._freq_offset = freq_offset
        self._linewidth = line_width
        super().__init__(
            iq_cluster_centers=self._iq_cluster_centers,
            iq_cluster_width=self._iq_cluster_width,
            compute_probabilities=compute_probabilities,
            calculation_parameters=calculation_parameters,
        )

        self._configuration.basis_gates = ["x"]
        self._configuration.timing_constraints = {"granularity": 16}


class TestQubitSpectroscopy(QiskitExperimentsTestCase):
    """Test spectroscopy experiment."""

    def test_spectroscopy_end2end_classified(self):
        """End to end test of the spectroscopy experiment."""

        calc_parameters = {"line_width": 2e6}
        backend = SpectroscopyBackend(
            compute_probabilities=compute_prob_qubit_spectroscopy,
            calculation_parameters=[calc_parameters],
        )
        qubit = 1
        freq01 = backend.defaults().qubit_freq_est[qubit]
        frequencies = np.linspace(freq01 - 10.0e6, freq01 + 10.0e6, 21)

        spec = QubitSpectroscopy(qubit, frequencies)
        spec.set_run_options(meas_level=MeasLevel.CLASSIFIED)
        expdata = spec.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results(1)
        self.assertRoundTripSerializable(result.value, check_func=self.ufloat_equiv)

        self.assertTrue(4.999e9 < result.value.n < 5.001e9)
        self.assertEqual(result.quality, "good")
        self.assertEqual(str(result.device_components[0]), f"Q{qubit}")

        # Test if we find still find the peak when it is shifted by 5 MHz.
        calc_parameters = {"line_width": 2e6, "freq_offset": 5.0e6}
        backend = SpectroscopyBackend(
            compute_probabilities=compute_prob_qubit_spectroscopy,
            calculation_parameters=[calc_parameters],
        )

        spec = QubitSpectroscopy(qubit, frequencies)
        spec.set_run_options(meas_level=MeasLevel.CLASSIFIED)
        expdata = spec.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results(1)
        self.assertRoundTripSerializable(result.value, check_func=self.ufloat_equiv)

        self.assertTrue(5.0049e9 < result.value.n < 5.0051e9)
        self.assertEqual(result.quality, "good")

    def test_spectroscopy_end2end_kerneled(self):
        """End to end test of the spectroscopy experiment on IQ data."""

        calc_parameters = {"line_width": 2e6}
        backend = SpectroscopyBackend(
            iq_cluster_centers=[((1.0, 1.0), (-1.0, -1.0))],
            compute_probabilities=compute_prob_qubit_spectroscopy,
            calculation_parameters=[calc_parameters],
        )
        qubit = 0
        freq01 = backend.defaults().qubit_freq_est[qubit]
        frequencies = np.linspace(freq01 - 10.0e6, freq01 + 10.0e6, 21)

        spec = QubitSpectroscopy(qubit, frequencies)
        expdata = spec.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results(1)
        self.assertRoundTripSerializable(result.value, check_func=self.ufloat_equiv)

        self.assertTrue(freq01 - 2e6 < result.value.n < freq01 + 2e6)
        self.assertEqual(result.quality, "good")

        # Test if we find still find the peak when it is shifted by 5 MHz.
        calc_parameters = {"line_width": 2e6, "freq_offset": 5.0e6}
        backend = SpectroscopyBackend(
            iq_cluster_centers=[((1.0, 1.0), (-1.0, -1.0))],
            compute_probabilities=compute_prob_qubit_spectroscopy,
            calculation_parameters=[calc_parameters],
        )

        spec = QubitSpectroscopy(qubit, frequencies)
        expdata = spec.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results(1)
        self.assertRoundTripSerializable(result.value, check_func=self.ufloat_equiv)

        self.assertTrue(freq01 + 3e6 < result.value.n < freq01 + 8e6)
        self.assertEqual(result.quality, "good")

        spec.set_run_options(meas_return="avg")
        expdata = spec.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results(1)
        self.assertRoundTripSerializable(result.value, check_func=self.ufloat_equiv)

        self.assertTrue(freq01 + 3e6 < result.value.n < freq01 + 8e6)
        self.assertEqual(result.quality, "good")

    def test_spectroscopy12_end2end_classified(self):
        """End to end test of the spectroscopy experiment with an x pulse."""

        calc_parameters = {"line_width": 2e6}
        backend = SpectroscopyBackend(
            compute_probabilities=compute_prob_qubit_spectroscopy,
            calculation_parameters=[calc_parameters],
        )
        qubit = 0
        freq01 = backend.defaults().qubit_freq_est[qubit]
        frequencies = np.linspace(freq01 - 10.0e6, freq01 + 10.0e6, 21)

        # Note that the backend is not sophisticated enough to simulate an e-f
        # transition so we run the test with g-e.
        spec = EFSpectroscopy(qubit, frequencies)
        spec.backend = backend
        spec.set_run_options(meas_level=MeasLevel.CLASSIFIED)
        expdata = spec.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results(1)
        self.assertRoundTripSerializable(result.value, check_func=self.ufloat_equiv)

        self.assertTrue(freq01 - 2e6 < result.value.n < freq01 + 2e6)
        self.assertEqual(result.quality, "good")

        # Test the circuits
        circ = spec.circuits()[0]
        self.assertEqual(circ.data[0][0].name, "x")
        self.assertEqual(circ.data[1][0].name, "Spec")

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = QubitSpectroscopy(1, np.linspace(100, 150, 20) * 1e6)
        loaded_exp = QubitSpectroscopy.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = QubitSpectroscopy(1, np.linspace(int(100e6), int(150e6), int(20e6)))
        self.assertRoundTripSerializable(exp, self.json_equiv)
