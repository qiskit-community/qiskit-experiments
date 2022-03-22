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

"""Test drag calibration experiment."""

from test.base import QiskitExperimentsTestCase
import unittest
from typing import Dict, List, Any
from ddt import ddt, data, unpack
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.exceptions import QiskitError
from qiskit.pulse import DriveChannel, Drag
import qiskit.pulse as pulse
from qiskit.qobj.utils import MeasLevel
from qiskit import transpile

from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.library import RoughDrag, RoughDragCal
from qiskit_experiments.test.mock_iq_backend import MockIQBackend
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.calibration_management import Calibrations


def set_default_calc_parameters_list(calc_parameters_list: List[Dict[str, Any]]):
    """
    Set default values to the calculation parameters list if they are not defined.
    Args:
        calc_parameters_list(List[Dict[str, any]]): A list of dictionaries that contain parameters for the probability
        calculation for the corresponding quantum circuit.
    """
    if "gate_name" not in calc_parameters_list[0].keys():
        calc_parameters_list[0]["gate_name"] = "Rp"
    if "error" not in calc_parameters_list[0].keys():
        calc_parameters_list[0]["error"] = 0.03
    if "ideal_beta" not in calc_parameters_list[0].keys():
        calc_parameters_list[0]["ideal_beta"] = 2.0
    if "freq" not in calc_parameters_list[0].keys():
        calc_parameters_list[0]["freq"] = 0.02
    if "max_prob" not in calc_parameters_list[0].keys():
        calc_parameters_list[0]["max_prob"] = 1.0
    if "offset_prob" not in calc_parameters_list[0].keys():
        calc_parameters_list[0]["offset_prob"] = 0.0

    if calc_parameters_list[0]["max_prob"] + calc_parameters_list[0]["offset_prob"] > 1:
        raise ValueError("Probabilities need to be between 0 and 1.")


def compute_probability(
    circuits: List[QuantumCircuit], calc_parameters_list: List[Dict[str, Any]]
) -> List[Dict[str, float]]:
    """Returns the probability based on the beta, number of gates, and leakage."""
    set_default_calc_parameters_list(calc_parameters_list)

    gate_name = calc_parameters_list[0]["gate_name"]
    error = calc_parameters_list[0]["error"]
    ideal_beta = calc_parameters_list[0]["ideal_beta"]
    freq = calc_parameters_list[0]["freq"]
    max_prob = calc_parameters_list[0]["max_prob"]
    offset_prob = calc_parameters_list[0]["offset_prob"]

    output_dict_list = []
    for circuit in circuits:
        probability_output_dict = {}
        # Need to change that the output will be dict. Need to see what the circuit do.
        n_gates = circuit.count_ops()[gate_name]
        beta = next(iter(circuit.calibrations[gate_name].keys()))[1][0]

        # Dictionary of output string vectors and their probability
        prob = np.sin(2 * np.pi * n_gates * freq * (beta - ideal_beta) / 4) ** 2
        probability_output_dict["1"] = max_prob * prob + offset_prob
        probability_output_dict["0"] = 1 - probability_output_dict["1"]
        output_dict_list.append(probability_output_dict)
    return output_dict_list


@ddt
class TestDragEndToEnd(QiskitExperimentsTestCase):
    """Test the drag experiment."""

    def setUp(self):
        """Setup some schedules."""
        super().setUp()

        beta = Parameter("β")

        with pulse.build(name="xp") as xp:
            pulse.play(Drag(duration=160, amp=0.208519, sigma=40, beta=beta), DriveChannel(0))

        self.x_plus = xp
        self.test_tol = 0.1

    def test_reps(self):
        """Test that setting reps raises and error if reps is not of length three."""

        drag = RoughDrag(0, self.x_plus)

        with self.assertRaises(CalibrationError):
            drag.set_experiment_options(reps=[1, 2, 3, 4])

    def test_end_to_end(self):
        """Test the drag experiment end to end."""

        calc_parameters = {"gate_name": "Drag(xp)", "ideal_beta": 2.0, "error": 0.03}
        backend = MockIQBackend(
            compute_probabilities=compute_probability, calculation_parameters=[calc_parameters]
        )

        drag = RoughDrag(1, self.x_plus)

        expdata = drag.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results(1)

        self.assertTrue(abs(result.value.n - calc_parameters["ideal_beta"]) < self.test_tol)
        self.assertEqual(result.quality, "good")

        # Small leakage will make the curves very flat, in this case one should
        # rather increase beta.
        calc_parameters["error"] = 0.0051
        calc_parameters["freq"] = 0.0044
        calc_parameters["gate_name"] = "Drag(xp)"
        backend = MockIQBackend(
            compute_probabilities=compute_probability, calculation_parameters=[calc_parameters]
        )

        drag = RoughDrag(0, self.x_plus)
        exp_data = drag.run(backend)
        self.assertExperimentDone(exp_data)
        result = exp_data.analysis_results(1)

        self.assertTrue(abs(result.value.n - calc_parameters["ideal_beta"]) < self.test_tol)
        self.assertEqual(result.quality, "good")

        # Large leakage will make the curves oscillate quickly.
        calc_parameters["error"] = 0.05
        calc_parameters["freq"] = 0.04
        backend = MockIQBackend(
            compute_probabilities=compute_probability, calculation_parameters=[calc_parameters]
        )

        drag = RoughDrag(1, self.x_plus, betas=np.linspace(-4, 4, 31))
        # pylint: disable=no-member
        drag.set_run_options(shots=200)
        drag.analysis.set_options(p0={"beta": 1.8, "freq": 0.08})
        exp_data = drag.run(backend)
        self.assertExperimentDone(exp_data)
        result = exp_data.analysis_results(1)

        meas_level = exp_data.metadata["job_metadata"][-1]["run_options"]["meas_level"]

        self.assertEqual(meas_level, MeasLevel.CLASSIFIED)
        self.assertTrue(abs(result.value.n - calc_parameters["ideal_beta"]) < self.test_tol)
        self.assertEqual(result.quality, "good")

    @data(
        (0.0040, 1.0, 0.00, [1, 3, 5], None, 0.1),  # partial oscillation.
        (0.0020, 0.5, 0.00, [1, 3, 5], None, 0.5),  # even slower oscillation with amp < 1
        (0.0040, 0.8, 0.05, [3, 5, 7], None, 0.1),  # constant offset, i.e. lower SNR.
        (0.0800, 0.9, 0.05, [1, 3, 5], np.linspace(-1, 1, 51), 0.1),  # Beta not in range
        (0.2000, 0.5, 0.10, [1, 3, 5], np.linspace(-2.5, 2.5, 51), 0.1),  # Max closer to zero
    )
    @unpack
    def test_nasty_data(self, freq, amp, offset, reps, betas, tol):
        """A set of tests for non-ideal data."""
        calc_parameters = {"gate_name": "Drag(xp)", "ideal_beta": 2.0, "error": 0.03, "freq": freq,
                           "max_prob": amp, "offset_prob": offset}
        backend = MockIQBackend(
            compute_probabilities=compute_probability, calculation_parameters=[calc_parameters]
        )

        drag = RoughDrag(0, self.x_plus, betas=betas)
        drag.set_experiment_options(reps=reps)

        exp_data = drag.run(backend)
        self.assertExperimentDone(exp_data)
        result = exp_data.analysis_results("beta")
        self.assertTrue(abs(result.value.n - calc_parameters["ideal_beta"]) < tol)
        # self.assertTrue(abs(result.value.n - backend.ideal_beta) < tol)
        self.assertEqual(result.quality, "good")


class TestDragCircuits(QiskitExperimentsTestCase):
    """Test the circuits of the drag calibration."""

    def setUp(self):
        """Setup some schedules."""
        super().setUp()

        beta = Parameter("β")

        with pulse.build(name="xp") as xp:
            pulse.play(Drag(duration=160, amp=0.208519, sigma=40, beta=beta), DriveChannel(0))

        self.x_plus = xp

    def test_default_circuits(self):
        """Test the default circuit."""

        calc_parameters = {"gate_name": "Drag(xp)", "ideal_beta": 2.0, "error": 0.03}
        calc_parameters["freq"] = 0.005
        backend = MockIQBackend(
            compute_probabilities=compute_probability, calculation_parameters=[calc_parameters]
        )
        calc_parameters = {"gate_name": "Drag(xp)", "ideal_beta": 2.0, "error": 0.03}
        drag = RoughDrag(0, self.x_plus)
        drag.set_experiment_options(reps=[2, 4, 8])
        drag.backend = MockIQBackend(
            compute_probabilities=compute_probability, calculation_parameters=[calc_parameters]
        )
        circuits = drag.circuits()

        for idx, expected in enumerate([4, 8, 16]):
            ops = transpile(circuits[idx * 51], backend).count_ops()
            self.assertEqual(ops["Drag(xp)"], expected)

    def test_raise_multiple_parameter(self):
        """Check that the experiment raises with unassigned parameters."""

        beta = Parameter("β")
        amp = Parameter("amp")

        with pulse.build(name="xp") as xp:
            pulse.play(Drag(duration=160, amp=amp, sigma=40, beta=beta), DriveChannel(0))

        with self.assertRaises(QiskitError):
            RoughDrag(1, xp, betas=np.linspace(-3, 3, 21))


class TestRoughDragCalUpdate(QiskitExperimentsTestCase):
    """Test that a Drag calibration experiment properly updates the calibrations."""

    def setUp(self):
        """Setup the tests"""
        super().setUp()

        library = FixedFrequencyTransmon()

        self.calc_parameters = {"gate_name": "Drag(x)", "ideal_beta": 2.0, "error": 0.03}
        self.backend = MockIQBackend(
            compute_probabilities=compute_probability, calculation_parameters=[self.calc_parameters]
        )
        self.cals = Calibrations.from_backend(self.backend, library)
        self.test_tol = 0.05

    def test_update(self):
        """Test that running RoughDragCal updates the calibrations."""

        qubit = 0
        prev_beta = self.cals.get_parameter_value("β", (0,), "x")
        self.assertEqual(prev_beta, 0)

        expdata = RoughDragCal(qubit, self.cals, backend=self.backend).run()
        self.assertExperimentDone(expdata)

        new_beta = self.cals.get_parameter_value("β", (0,), "x")
        self.assertTrue(abs(new_beta - self.calc_parameters["ideal_beta"]) < self.test_tol)
        self.assertTrue(abs(new_beta) > self.test_tol)

    def test_dragcal_experiment_config(self):
        """Test RoughDragCal config can round trip"""
        exp = RoughDragCal(0, self.cals, backend=self.backend)
        loaded_exp = RoughDragCal.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    @unittest.skip("Calibration experiments are not yet JSON serializable")
    def test_dragcal_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = RoughDragCal(0, self.cals)
        self.assertRoundTripSerializable(exp, self.json_equiv)

    def test_drag_experiment_config(self):
        """Test RoughDrag config can roundtrip"""
        with pulse.build(name="xp") as sched:
            pulse.play(pulse.Drag(160, 0.5, 40, Parameter("β")), pulse.DriveChannel(0))
        exp = RoughDrag(0, backend=self.backend, schedule=sched)
        loaded_exp = RoughDrag.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    @unittest.skip("Schedules are not yet JSON serializable")
    def test_drag_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        with pulse.build(name="xp") as sched:
            pulse.play(pulse.Drag(160, 0.5, 40, Parameter("β")), pulse.DriveChannel(0))
        exp = RoughDrag(0, backend=self.backend, schedule=sched)
        self.assertRoundTripSerializable(exp, self.json_equiv)
