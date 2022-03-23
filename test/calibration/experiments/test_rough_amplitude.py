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

"""Test rough amplitude calibration experiment classes."""

from test.base import QiskitExperimentsTestCase
import unittest
from typing import Dict, List, Any
import numpy as np

from qiskit import QuantumCircuit, transpile
import qiskit.pulse as pulse
from qiskit.circuit import Parameter
from qiskit.test.mock import FakeArmonk

from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.library import EFRoughXSXAmplitudeCal, RoughXSXAmplitudeCal
from qiskit_experiments.test.mock_iq_backend import MockIQBackend


def rabi_compute_probabilities(
    circuits: List[QuantumCircuit], calc_parameters_list: List[Dict[str, Any]]
) -> List[Dict[str, float]]:
    """Returns the probability based on the rotation angle and amplitude_to_angle."""
    amplitude_to_angle = calc_parameters_list[0].get("amplitude_to_angle", np.pi)
    output_dict_list = []
    for circuit in circuits:
        probability_output_dict = {}
        amp = next(iter(circuit.calibrations["Rabi"].keys()))[1][0]

        # Dictionary of output string vectors and their probability
        probability_output_dict["1"] = np.sin(amplitude_to_angle * amp) ** 2
        probability_output_dict["0"] = 1 - probability_output_dict["1"]
        output_dict_list.append(probability_output_dict)
    return output_dict_list


class TestRoughAmpCal(QiskitExperimentsTestCase):
    """A class to test the rough amplitude calibration experiments."""

    def setUp(self):
        """Setup the tests."""
        super().setUp()
        library = FixedFrequencyTransmon()

        self.backend = FakeArmonk()
        self.cals = Calibrations.from_backend(self.backend, library)

    def test_circuits(self):
        """Test the quantum circuits."""
        test_amps = [-0.5, 0, 0.5]
        rabi = RoughXSXAmplitudeCal(0, self.cals, amplitudes=test_amps)

        circs = transpile(rabi.circuits(), self.backend, inst_map=rabi.transpile_options.inst_map)

        for circ, amp in zip(circs, test_amps):
            self.assertEqual(circ.count_ops()["Rabi"], 1)

            d0 = pulse.DriveChannel(0)
            with pulse.build(name="x") as expected_x:
                pulse.play(pulse.Drag(160, amp, 40, 0), d0)

            self.assertEqual(circ.calibrations["Rabi"][((0,), (amp,))], expected_x)

    def test_update(self):
        """Test that the calibrations update properly."""

        self.assertTrue(np.allclose(self.cals.get_parameter_value("amp", 0, "x"), 0.5))
        self.assertTrue(np.allclose(self.cals.get_parameter_value("amp", 0, "sx"), 0.25))

        rabi_ef = RoughXSXAmplitudeCal(0, self.cals)
        rabi_calc_parameters_list = {"amplitude_to_angle": np.pi * 1.5}
        expdata = rabi_ef.run(MockIQBackend(
            compute_probabilities=rabi_compute_probabilities,
            calculation_parameters=[rabi_calc_parameters_list],
        ))
        self.assertExperimentDone(expdata)

        tol = 0.002
        self.assertTrue(abs(self.cals.get_parameter_value("amp", 0, "x") - 0.333) < tol)
        self.assertTrue(abs(self.cals.get_parameter_value("amp", 0, "sx") - 0.333 / 2) < tol)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = RoughXSXAmplitudeCal(0, self.cals)
        config = exp.config()
        loaded_exp = RoughXSXAmplitudeCal.from_config(config)
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqual(config, loaded_exp.config())


class TestSpecializations(QiskitExperimentsTestCase):
    """Test the specialized versions of the calibration."""

    def setUp(self):
        """Setup the tests"""
        super().setUp()

        library = FixedFrequencyTransmon()

        self.backend = FakeArmonk()
        self.cals = Calibrations.from_backend(self.backend, library)

        # Add some pulses on the 1-2 transition.
        d0 = pulse.DriveChannel(0)
        with pulse.build(name="x12") as x12:
            with pulse.frequency_offset(-300e6, d0):
                pulse.play(pulse.Drag(160, Parameter("amp"), 40, 0.0), d0)

        with pulse.build(name="sx12") as sx12:
            with pulse.frequency_offset(-300e6, d0):
                pulse.play(pulse.Drag(160, Parameter("amp"), 40, 0.0), d0)

        self.cals.add_schedule(x12, 0)
        self.cals.add_schedule(sx12, 0)
        self.cals.add_parameter_value(0.4, "amp", 0, "x12")
        self.cals.add_parameter_value(0.2, "amp", 0, "sx12")

    def test_ef_circuits(self):
        """Test that we get the expected circuits with calibrations for the EF experiment."""

        test_amps = [-0.5, 0, 0.5]
        rabi_ef = EFRoughXSXAmplitudeCal(0, self.cals, amplitudes=test_amps)

        circs = transpile(
            rabi_ef.circuits(), self.backend, inst_map=rabi_ef.transpile_options.inst_map
        )

        for circ, amp in zip(circs, test_amps):

            self.assertEqual(circ.count_ops()["x"], 1)
            self.assertEqual(circ.count_ops()["Rabi"], 1)

            d0 = pulse.DriveChannel(0)
            with pulse.build(name="x") as expected_x:
                pulse.play(pulse.Drag(160, 0.5, 40, 0), d0)

            with pulse.build(name="x12") as expected_x12:
                with pulse.frequency_offset(-300e6, d0):
                    pulse.play(pulse.Drag(160, amp, 40, 0), d0)

            self.assertEqual(circ.calibrations["x"][((0,), ())], expected_x)
            self.assertEqual(circ.calibrations["Rabi"][((0,), (amp,))], expected_x12)

    def test_ef_update(self):
        """Tes that we properly update the pulses on the 1<->2 transition."""

        self.assertTrue(np.allclose(self.cals.get_parameter_value("amp", 0, "x12"), 0.4))
        self.assertTrue(np.allclose(self.cals.get_parameter_value("amp", 0, "sx12"), 0.2))

        rabi_ef = EFRoughXSXAmplitudeCal(0, self.cals)
        rabi_calc_parameters_list = {"amplitude_to_angle": np.pi * 1.5}
        expdata = rabi_ef.run(MockIQBackend(
            compute_probabilities=rabi_compute_probabilities,
            calculation_parameters=[rabi_calc_parameters_list],
        ))
        self.assertExperimentDone(expdata)

        tol = 0.002
        self.assertTrue(abs(self.cals.get_parameter_value("amp", 0, "x12") - 0.333) < tol)
        self.assertTrue(abs(self.cals.get_parameter_value("amp", 0, "sx12") - 0.333 / 2) < tol)


if __name__ == "__main__":
    unittest.main()
