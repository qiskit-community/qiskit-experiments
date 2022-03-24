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

"""Test the fine frequency characterization and calibration experiments."""

from test.base import QiskitExperimentsTestCase
from typing import Dict, List, Any
import numpy as np
from ddt import ddt, data

from qiskit import QuantumCircuit
from qiskit.test.mock import FakeArmonk
from qiskit.providers.aer import AerSimulator
import qiskit.pulse as pulse

from qiskit_experiments.library import (
    FineFrequency,
    FineFrequencyCal,
)
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.test.mock_iq_backend import MockIQBackend


def fine_freq_compute_probabilities(
    circuits: List[QuantumCircuit], calc_parameters: List[Dict[str, Any]]
) -> List[Dict[str, float]]:
    """Return the probability of being in the excited state."""
    simulator = AerSimulator(method="automatic")
    sx_duration = calc_parameters[0].get("sx_duration", 160)
    freq_shift = calc_parameters[0].get("freq_shift", 0)
    dt = calc_parameters[0].get("dt", 1e-9)
    output_dict_list = []
    for circuit in circuits:
        probability_output_dict = {}
        delay = None
        for instruction in circuit.data:
            if instruction[0].name == "delay":
                delay = instruction[0].duration

        if delay is None:
            probability_output_dict["1"] = 1
            probability_output_dict["0"] = 1 - probability_output_dict["1"]
        else:
            reps = delay // sx_duration

            qc = QuantumCircuit(1)
            qc.sx(0)
            qc.rz(np.pi * reps / 2 + 2 * np.pi * freq_shift * delay * dt, 0)
            qc.sx(0)
            qc.measure_all()

            counts = simulator.run(qc, seed_simulator=1).result().get_counts(0)
            probability_output_dict["1"] = counts.get("1", 0) / sum(counts.values())
            probability_output_dict["0"] = 1 - probability_output_dict["1"]
        output_dict_list.append(probability_output_dict)

    return output_dict_list


@ddt
class TestFineFreqEndToEnd(QiskitExperimentsTestCase):
    """Test the fine freq experiment."""

    def setUp(self):
        """Setup for the test."""
        super().setUp()
        self.inst_map = pulse.InstructionScheduleMap()

        self.sx_duration = 160

        with pulse.build(name="sx") as sx_sched:
            pulse.play(pulse.Gaussian(self.sx_duration, 0.5, 40), pulse.DriveChannel(0))

        self.inst_map.add("sx", 0, sx_sched)

        self.cals = Calibrations.from_backend(FakeArmonk(), FixedFrequencyTransmon())

    @data(-0.5e6, -0.1e6, 0.1e6, 0.5e6)
    def test_end_to_end(self, freq_shift):
        """Test the experiment end to end."""
        calc_parameters = {"freq_shift": freq_shift, "sx_duration": self.sx_duration}
        backend = MockIQBackend(compute_probabilities=fine_freq_compute_probabilities)
        calc_parameters["dt"] = backend.configuration().dt
        backend.set_calculation_parameters([calc_parameters])

        freq_exp = FineFrequency(0, 160, backend)
        freq_exp.set_transpile_options(inst_map=self.inst_map)

        expdata = freq_exp.run(shots=100)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results(1)
        d_theta = result.value.n
        dt = backend.configuration().dt
        d_freq = d_theta / (2 * np.pi * self.sx_duration * dt)

        tol = 0.01e6

        self.assertAlmostEqual(d_freq, freq_shift, delta=tol)
        self.assertEqual(result.quality, "good")

    def test_calibration_version(self):
        """Test the calibration version of the experiment."""

        freq_shift = 0.1e6
        calc_parameters = {"freq_shift": freq_shift, "sx_duration": self.sx_duration}
        backend = MockIQBackend(compute_probabilities=fine_freq_compute_probabilities)
        calc_parameters["dt"] = backend.configuration().dt
        backend.set_calculation_parameters([calc_parameters])

        fine_freq = FineFrequencyCal(0, self.cals, backend)
        armonk_freq = FakeArmonk().defaults().qubit_freq_est[0]

        freq_before = self.cals.get_parameter_value(self.cals.__drive_freq_parameter__, 0)

        self.assertAlmostEqual(freq_before, armonk_freq)

        expdata = fine_freq.run()
        self.assertExperimentDone(expdata)

        freq_after = self.cals.get_parameter_value(self.cals.__drive_freq_parameter__, 0)

        # Test equality up to 10kHz on a 100 kHz shift
        self.assertAlmostEqual(freq_after, armonk_freq + freq_shift, delta=1e4)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = FineFrequency(0, 160)
        loaded_exp = FineFrequency.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = FineFrequency(0, 160)
        self.assertRoundTripSerializable(exp, self.json_equiv)
