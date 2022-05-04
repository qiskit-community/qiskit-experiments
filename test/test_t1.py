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
Test T1 experiment
"""

from test.base import QiskitExperimentsTestCase
import numpy as np
from qiskit_experiments.framework import ExperimentData, ParallelExperiment
from qiskit_experiments.library import T1
from qiskit_experiments.library.characterization import T1Analysis
from qiskit_experiments.test.t1_backend import T1Backend
from qiskit_experiments.test.fake_service import FakeService


class TestT1(QiskitExperimentsTestCase):
    """
    Test measurement of T1
    """

    def test_t1_end2end(self):
        """
        Test T1 experiment using a simulator.
        """
        t1 = 25e-6
        backend = T1Backend(
            [t1],
            initial_prob1=[0.02],
            readout0to1=[0.02],
            readout1to0=[0.02],
        )

        delays = np.arange(1e-6, 40e-6, 3e-6)

        exp = T1(0, delays)
        exp.analysis.set_options(p0={"amp": 1, "tau": t1, "base": 0})
        exp_data = exp.run(backend, shots=10000)
        self.assertExperimentDone(exp_data)
        self.assertRoundTripSerializable(exp_data, check_func=self.experiment_data_equiv)
        self.assertRoundTripPickle(exp_data, check_func=self.experiment_data_equiv)
        res = exp_data.analysis_results("T1")
        self.assertEqual(res.quality, "good")
        self.assertAlmostEqual(res.value.n, t1, delta=3)
        self.assertEqual(res.extra["unit"], "s")

        exp_data.service = FakeService()
        exp_data.save()
        loaded_data = ExperimentData.load(exp_data.experiment_id, exp_data.service)
        exp_res = exp_data.analysis_results()
        load_res = loaded_data.analysis_results()
        repr1 = sorted([repr(res) for res in exp_res])
        repr2 = sorted([repr(res) for res in load_res])
        self.assertEqual(repr1, repr2)

    def test_t1_parallel(self):
        """
        Test parallel experiments of T1 using a simulator.
        """

        t1 = [25, 15]
        delays = list(range(1, 40, 3))

        exp0 = T1(0, delays)
        exp2 = T1(2, delays)
        par_exp = ParallelExperiment([exp0, exp2])
        res = par_exp.run(T1Backend([t1[0], None, t1[1]]))
        self.assertExperimentDone(res)

        for i in range(2):
            sub_res = res.child_data(i).analysis_results("T1")
            self.assertEqual(sub_res.quality, "good")
            self.assertAlmostEqual(sub_res.value.n, t1[i], delta=3)

        res.service = FakeService()
        res.save()
        loaded_data = ExperimentData.load(res.experiment_id, res.service)

        for i in range(2):
            sub_res = res.child_data(i).analysis_results("T1")
            sub_loaded = loaded_data.child_data(i).analysis_results("T1")
            self.assertEqual(repr(sub_res), repr(sub_loaded))

    def test_t1_parallel_different_analysis_options(self):
        """
        Test parallel experiments of T1 using a simulator, for the case where
        the sub-experiments have different analysis options
        """

        t1 = 25
        delays = list(range(1, 40, 3))

        exp0 = T1(0, delays)
        exp0.analysis.set_options(p0={"tau": 30})
        exp1 = T1(1, delays)
        exp1.analysis.set_options(p0={"tau": 1000000})

        par_exp = ParallelExperiment([exp0, exp1])
        res = par_exp.run(T1Backend([t1, t1]))
        self.assertExperimentDone(res)

        sub_res = []
        for i in range(2):
            sub_res.append(res.child_data(i).analysis_results("T1"))

        self.assertEqual(sub_res[0].quality, "good")
        self.assertAlmostEqual(sub_res[0].value.n, t1, delta=3)
        self.assertEqual(sub_res[1].quality, "bad")

    def test_t1_analysis(self):
        """
        Test T1Analysis
        """

        data = ExperimentData()
        data._metadata = {"meas_level": 2}

        numbers = [750, 1800, 2750, 3550, 4250, 4850, 5450, 5900, 6400, 6800, 7000, 7350, 7700]

        for i, count0 in enumerate(numbers):
            data.add_data(
                {
                    "counts": {"0": count0, "1": 10000 - count0},
                    "metadata": {
                        "xval": (3 * i + 1) * 1e-9,
                        "experiment_type": "T1",
                        "qubit": 0,
                        "unit": "s",
                    },
                }
            )

        res, _ = T1Analysis()._run_analysis(data)
        result = res[1]
        self.assertEqual(result.quality, "good")
        self.assertAlmostEqual(result.value.nominal_value, 25e-9, delta=3)

    def test_t1_metadata(self):
        """
        Test the circuits metadata
        """

        delays = np.arange(1e-3, 40e-3, 3e-3)
        exp = T1(0, delays)
        circs = exp.circuits()

        self.assertEqual(len(circs), len(delays))

        for delay, circ in zip(delays, circs):
            xval = circ.metadata.pop("xval")
            self.assertAlmostEqual(xval, delay)
            self.assertEqual(
                circ.metadata,
                {
                    "experiment_type": "T1",
                    "qubit": 0,
                    "unit": "s",
                },
            )

    def test_t1_low_quality(self):
        """
        A test where the fit's quality will be low
        """

        data = ExperimentData()
        data._metadata = {"meas_level": 2}

        for i in range(10):
            data.add_data(
                {
                    "counts": {"0": 10, "1": 10},
                    "metadata": {
                        "xval": i * 1e-9,
                        "experiment_type": "T1",
                        "qubit": 0,
                        "unit": "s",
                    },
                }
            )

        res, _ = T1Analysis()._run_analysis(data)
        result = res[1]
        self.assertEqual(result.quality, "bad")

    def test_t1_parallel_exp_transpile(self):
        """Test parallel transpile options for T1 experiment"""
        num_qubits = 5
        instruction_durations = []
        for i in range(num_qubits):
            instruction_durations += [
                ("rx", [i], (i + 1) * 10, "ns"),
                ("measure", [i], (i + 1) * 1000, "ns"),
            ]
        coupling_map = [[i - 1, i] for i in range(1, num_qubits)]
        basis_gates = ["rx", "delay"]

        exp1 = T1(1, delays=[50e-9, 100e-9, 160e-9])
        exp2 = T1(3, delays=[40e-9, 80e-9, 190e-9])
        parexp = ParallelExperiment([exp1, exp2])
        parexp.set_transpile_options(
            basis_gates=basis_gates,
            instruction_durations=instruction_durations,
            coupling_map=coupling_map,
            scheduling_method="alap",
        )

        circs = parexp.circuits()
        for circ in circs:
            self.assertEqual(circ.num_qubits, 2)
            op_counts = circ.count_ops()
            self.assertEqual(op_counts.get("rx"), 2)
            self.assertEqual(op_counts.get("delay"), 2)

        tcircs = parexp._transpiled_circuits()
        for circ in tcircs:
            self.assertEqual(circ.num_qubits, num_qubits)
            op_counts = circ.count_ops()
            self.assertEqual(op_counts.get("rx"), 2)
            self.assertGreater(op_counts.get("delay"), num_qubits - 1)

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = T1(0, [1, 2, 3, 4, 5])
        loaded_exp = T1.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = T1(0, [1, 2, 3, 4, 5])
        self.assertRoundTripSerializable(exp, self.json_equiv)

    def test_analysis_config(self):
        """ "Test converting analysis to and from config works"""
        analysis = T1Analysis()
        loaded = T1Analysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())
