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
Test T2Ramsey experiment
"""
import unittest
import numpy as np

from qiskit.utils import apply_prefix
from qiskit.providers import BackendV1
from qiskit.providers.options import Options
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit.test import QiskitTestCase
from qiskit_experiments.composite import ParallelExperiment
from qiskit_experiments.characterization.t2ramsey import T2Ramsey
from qiskit_experiments.characterization.t2ramsey_analysis import T2RamseyAnalysis
#from .mock_job import MockJob


class T2RamseyBackend(BackendV1):
    """
    A simple and primitive backend, to be run by the T2Ramsey tests
    """

    def __init__(
        self,
        p0=None,
        initial_prob_plus=None,
        readout0to1=None,
        readout1to0=None,
        dt_factor=1,
    ):
        """
        Initialize the T2Ramsey backend
        """
        dt_factor_in_ns = dt_factor * 1e9 if dt_factor is not None else None
        configuration = QasmBackendConfiguration(
            backend_name="t2Ramsey_simulator",
            backend_version="0",
            n_qubits=int(1e6),
            basis_gates=["barrier", "h", "p", "delay", "measure"],
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

        self._t2ramsey = p0["t2Ramsey"]
        self._a_guess = p0["a_guess"]
        self._f_guess = p0["f_guess"]
        self._phi_guess = p0["phi_guess"]
        self._b_guess = p0["b_guess"]
        self._initial_prob_plus = initial_prob_plus
        self._readout0to1 = readout0to1
        self._readout1to0 = readout1to0
        self._dt_factor = dt_factor
        self._rng = np.random.default_rng(0)
        super().__init__(configuration)

    @classmethod
    def _default_options(cls):
        """Default options of the test backend."""
        return Options(shots=1024)

    def run(self, run_input, **options):
        """
        Run the T2Ramsey backend
        """
        self.options.update_options(**options)
        shots = self.options.get("shots")

        result = {
            "backend_name": "T2Ramsey backend",
            "backend_version": "0",
            "qobj_id": 0,
            "job_id": 0,
            "success": True,
            "results": [],
        }

        for circ in run_input:
            nqubits = circ.num_qubits
            qubit_indices = {bit: idx for idx, bit in enumerate(circ.qubits)}
            clbit_indices = {bit: idx for idx, bit in enumerate(circ.clbits)}
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
                if self._initial_prob_plus is None:
                    prob_plus = np.ones(nqubits)
                else:
                    prob_plus = self._initial_prob_plus.copy()

                clbits = np.zeros(circ.num_clbits, dtype=int)
                for op, qargs, cargs in circ.data:
                    qubit = qubit_indices[qargs[0]]

                    if op.name == "delay":
                        delay = op.params[0]
                        t2ramsey = self._t2ramsey[qubit] * self._dt_factor
                        freq = self._f_guess[qubit] / self._dt_factor

                        prob_plus[qubit] = (
                            self._a_guess[qubit]
                            * np.exp(-delay / t2ramsey)
                            * np.cos(2 * np.pi * freq * delay + self._phi_guess[qubit])
                            + self._b_guess[qubit]
                        )

                    if op.name == "measure":
                        # we measure in |+> basis which is the same as measuring |0>
                        meas_res = self._rng.binomial(
                            1,
                            (1 - prob_plus[qubit]) * (1 - ro10[qubit])
                            + prob_plus[qubit] * ro01[qubit],
                        )
                        clbit = clbit_indices[cargs[0]]
                        clbits[clbit] = meas_res

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
                    "header": {"metadata": circ.metadata},
                    "data": {"counts": counts},
                }
            )
        #return MockJob(self, Result.from_dict(result))
        return Result.from_dict(result)


class TestT2Ramsey(QiskitTestCase):
    """Test T2Ramsey experiment"""

    def test_t2Ramsey_run_end2end(self):
        """
        Run the T2Ramsey backend on all possible units
        """
        for unit in ["s", "ms", "us", "ns", "dt"]:
            if unit in ("s", "dt"):
                dt_factor = 1
            else:
                dt_factor = apply_prefix(1, unit)
            estimated_t2ramsey = 20
            estimated_freq = 0.1
            # Set up the circuits
            qubit = 0
            if unit == "dt":  # dt requires integer values for delay
                delays = list(range(1, 46))
            else:
                delays = np.append(
                    (np.linspace(1.0, 15.0, num=15)).astype(float),
                    (np.linspace(16.0, 45.0, num=59)).astype(float),
                )

            exp = T2Ramsey(qubit, delays, unit=unit)
            default_p0 = {
                "A": 0.5,
                "t2Ramsey": estimated_t2ramsey,
                "f": estimated_freq,
                "phi": 0,
                "B": 0.5,
            }
            for user_p0 in [default_p0, None]:
                exp.set_analysis_options(user_p0=user_p0)
                backend = T2RamseyBackend(
                    p0={
                        "a_guess": [0.5],
                        "t2Ramsey": [estimated_t2ramsey],
                        "f_guess": [estimated_freq],
                        "phi_guess": [0.0],
                        "b_guess": [0.5],
                    },
                    initial_prob_plus=[0.0],
                    readout0to1=[0.02],
                    readout1to0=[0.02],
                    dt_factor=dt_factor,
                )

            expdata = exp.run(
                backend=backend,
                shots=2000,
            )
            result = expdata.analysis_result(0)
            self.assertAlmostEqual(
                result["t2Ramsey_value"],
                estimated_t2ramsey * dt_factor,
                delta=3 * dt_factor,
            )
            self.assertAlmostEqual(
                result["frequency_value"],
                estimated_freq / dt_factor,
                delta=3 / dt_factor,
            )
            self.assertEqual(
                result["quality"], "computer_good", "Result quality bad for unit " + str(unit)
            )

    def test_t2Ramsey_parallel(self):
        """
        Test parallel experiments of T2Ramsey using a simulator.
        """

        t2ramsey = [30, 25]
        estimated_freq = [0.1, 0.12]
        delays = [list(range(1, 60)), list(range(1, 50))]

        exp0 = T2Ramsey(0, delays[0])
        exp2 = T2Ramsey(2, delays[1])
        par_exp = ParallelExperiment([exp0, exp2])

        p0 = {
            "a_guess": [0.5, None, 0.5],
            "t2Ramsey": [t2ramsey[0], None, t2ramsey[1]],
            "f_guess": [estimated_freq[0], None, estimated_freq[1]],
            "phi_guess": [0, None, 0],
            "b_guess": [0.5, None, 0.5],
        }
        backend = T2RamseyBackend(p0)
        res = par_exp.run(backend=backend, shots=1000)

        for i in range(2):
            sub_res = res.component_experiment_data(i).analysis_result(0)
            self.assertAlmostEqual(sub_res["t2Ramsey_value"], t2ramsey[i], delta=3)
            self.assertAlmostEqual(
                sub_res["frequency_value"],
                estimated_freq[i],
                delta=3,
            )
            sub_res = res.component_experiment_data(i).analysis_result(0)
            self.assertEqual(
                sub_res["quality"],
                "computer_good",
                "Result quality bad for experiment on qubit " + str(i),
            )


if __name__ == "__main__":
    unittest.main()
