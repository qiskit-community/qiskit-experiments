# -*- coding: utf-8 -*-

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

import unittest
import numpy as np
from qiskit.providers import BaseBackend
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit_experiments import ExperimentData
from qiskit_experiments.composite import ParallelExperiment
from qiskit_experiments.characterization import T1Experiment, T1Analysis


class T1Backend(BaseBackend):
    """
    A simple and primitive backend, to be run by the T1 tests
    """

    def __init__(self, t1, initial_prob1=None, readout0to1=None, readout1to0=None, dt_factor=None):
        """
        Initialize the T1 backend
        """

        dt_factor_in_ns = dt_factor * 1e9 if dt_factor is not None else None

        configuration = QasmBackendConfiguration(
            backend_name="t1_simulator",
            backend_version="0",
            n_qubits=int(1e6),
            basis_gates=["barrier", "x", "delay", "measure"],
            gates=[],
            local=True,
            simulator=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=int(1e6),
            coupling_map=None,
            dt=dt_factor_in_ns,
        )

        self._t1 = t1
        self._initial_prob1 = initial_prob1
        self._readout0to1 = readout0to1
        self._readout1to0 = readout1to0
        self._dt_factor = dt_factor
        super().__init__(configuration)

    # pylint: disable = arguments-differ
    def run(self, qobj):
        """
        Run the T1 backend
        """

        shots = qobj.config.shots

        result = {
            "backend_name": "T1 backend",
            "backend_version": "0",
            "qobj_id": 0,
            "job_id": 0,
            "success": True,
            "results": [],
        }

        for circ in qobj.experiments:
            nqubits = circ.config.n_qubits
            counts = dict()

            if self._readout0to1 is None:
                ro01 = np.zeros(nqubits)
            else:
                ro01 = self._readout0to1

            if self._readout1to0 is None:
                ro10 = np.zeros(nqubits)
            else:
                ro10 = self._readout1to0

            for _ in range(shots):
                if self._initial_prob1 is None:
                    prob1 = np.zeros(nqubits)
                else:
                    prob1 = self._initial_prob1.copy()

                clbits = np.zeros(circ.config.memory_slots, dtype=int)

                for op in circ.instructions:
                    qubit = op.qubits[0]
                    if op.name == "x":
                        prob1[qubit] = 1 - prob1[qubit]
                    elif op.name == "delay":
                        delay = op.params[0]
                        prob1[qubit] = prob1[qubit] * np.exp(-delay / self._t1[qubit])
                    elif op.name == "measure":
                        meas_res = np.random.binomial(
                            1, prob1[qubit] * (1 - ro10[qubit]) + (1 - prob1[qubit]) * ro01[qubit]
                        )
                        clbits[op.memory[0]] = meas_res
                        prob1[qubit] = meas_res

                clstr = ""
                for clbit in clbits[::-1]:
                    clstr = clstr + str(clbit)

                if clstr in counts:
                    counts[clstr] += 1
                else:
                    counts[clstr] = 1

            result["results"].append(
                {
                    "shots": shots,
                    "success": True,
                    "header": {"metadata": circ.header.metadata},
                    "data": {"counts": counts},
                }
            )

        return Result.from_dict(result)


class TestT1(unittest.TestCase):
    """
    Test measurement of T1
    """

    def test_t1_end2end(self):
        """
        Test T1 experiment using a simulator.
        """

        dt_factor = 2e-7

        t1 = 25e-6
        backend = T1Backend(
            [t1 / dt_factor],
            initial_prob1=[0.02],
            readout0to1=[0.02],
            readout1to0=[0.02],
            dt_factor=dt_factor,
        )

        delays = list(
            range(
                int(1e-6 / dt_factor),
                int(40e-6 / dt_factor),
                int(3e-6 / dt_factor),
            )
        )

        # dummy numbers to avoid exception triggerring
        instruction_durations = [
            ("measure", [0], 3, "dt"),
            ("x", [0], 3, "dt"),
        ]

        exp = T1Experiment(0, delays, unit="dt")
        res = exp.run(
            backend,
            amplitude_guess=1,
            t1_guess=t1 / dt_factor,
            offset_guess=0,
            instruction_durations=instruction_durations,
            shots=10000,
        ).analysis_result(0)

        self.assertEqual(res["quality"], "computer_good")
        self.assertAlmostEqual(res["value"], t1, delta=3)

    def test_t1_parallel(self):
        """
        Test parallel experiments of T1 using a simulator.
        """

        t1 = [25, 15]
        delays = list(range(1, 40, 3))

        exp0 = T1Experiment(0, delays)
        exp2 = T1Experiment(2, delays)
        par_exp = ParallelExperiment([exp0, exp2])
        res = par_exp.run(
            T1Backend([t1[0], None, t1[1]]),
            shots=10000,
        )

        for i in range(2):
            sub_res = res.component_experiment_data(i).analysis_result(0)
            self.assertTrue(sub_res["quality"], "computer_good")
            self.assertAlmostEqual(sub_res["value"], t1[i], delta=3)

    def test_t1_analysis(self):
        """
        Test T1Analysis
        """

        data = ExperimentData(None)
        numbers = [750, 1800, 2750, 3550, 4250, 4850, 5450, 5900, 6400, 6800, 7000, 7350, 7700]

        for i, count0 in enumerate(numbers):
            data._data.append(
                {
                    "counts": {"0": count0, "1": 10000 - count0},
                    "metadata": {
                        "xval": 3 * i + 1,
                        "experiment_type": "T1Experiment",
                        "qubit": 0,
                        "unit": "ns",
                        "dt_factor_in_sec": None,
                    },
                }
            )

        res = T1Analysis()._run_analysis(data)[0]
        self.assertEqual(res["quality"], "computer_good")
        self.assertAlmostEqual(res["value"], 25e-9, delta=3)

    def test_t1_metadata(self):
        """
        Test the circuits metadata
        """

        delays = list(range(1, 40, 3))
        exp = T1Experiment(0, delays, unit="ms")
        circs = exp.circuits()

        self.assertEqual(len(circs), len(delays))

        for delay, circ in zip(delays, circs):
            self.assertEqual(
                circ.metadata,
                {
                    "experiment_type": "T1Experiment",
                    "qubit": 0,
                    "xval": delay,
                    "unit": "ms",
                },
            )

    def test_t1_low_quality(self):
        """
        A test where the fit's quality will be low
        """

        data = ExperimentData(None)

        for i in range(10):
            data._data.append(
                {
                    "counts": {"0": 10, "1": 10},
                    "metadata": {
                        "xval": i,
                        "experiment_type": "T1Experiment",
                        "qubit": 0,
                        "unit": "ns",
                        "dt_factor_in_sec": None,
                    },
                }
            )

        res = T1Analysis()._run_analysis(data)[0]
        self.assertEqual(res["quality"], "computer_bad")


if __name__ == "__main__":
    unittest.main()
