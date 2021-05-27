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
Test T2Star experiment
"""
import unittest
import numpy as np

from qiskit.utils import apply_prefix
from qiskit.providers import BaseBackend
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit.test import QiskitTestCase
from qiskit_experiments.composite import ParallelExperiment
from qiskit_experiments.characterization import T2StarExperiment


# Fix seed for simulations
SEED = 9000


class T2starBackend(BaseBackend):
    """
    A simple and primitive backend, to be run by the T2Star tests
    """

    def __init__(
        self, p0=None, initial_prob_plus=None, readout0to1=None, readout1to0=None, dt_factor=1
    ):
        """
        Initialize the T2star backend
        """

        configuration = QasmBackendConfiguration(
            backend_name="t2star_simulator",
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
            dt=dt_factor,
        )

        self._t2star = p0["t2star"]
        self._a_guess = p0["a_guess"]
        self._f_guess = p0["f_guess"]
        self._phi_guess = p0["phi_guess"]
        self._b_guess = p0["b_guess"]
        self._initial_prob_plus = initial_prob_plus
        self._readout0to1 = readout0to1
        self._readout1to0 = readout1to0
        self._dt_factor = dt_factor
        super().__init__(configuration)

    # pylint: disable = arguments-differ
    def run(self, qobj):
        """
        Run the T2star backend
        """
        shots = qobj.config.shots
        result = {
            "backend_name": "T2star backend",
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
                if self._initial_prob_plus is None:
                    prob_plus = np.ones(nqubits)
                else:
                    prob_plus = self._initial_prob_plus.copy()

                clbits = np.zeros(circ.config.memory_slots, dtype=int)
                for op in circ.instructions:
                    qubit = op.qubits[0]

                    if op.name == "delay":
                        delay = op.params[0]
                        t2star = self._t2star[qubit] * self._dt_factor
                        freq = self._f_guess[qubit] / self._dt_factor

                        prob_plus[qubit] = (
                            self._a_guess[qubit]
                            * np.exp(-delay / t2star)
                            * np.cos(2 * np.pi * freq * delay + self._phi_guess[qubit])
                            + self._b_guess[qubit]
                        )

                    if op.name == "measure":
                        # we measure in |+> basis which is the same as measuring |0>
                        meas_res = np.random.binomial(
                            1,
                            (1 - prob_plus[qubit]) * (1 - ro10[qubit])
                            + prob_plus[qubit] * ro01[qubit],
                        )
                        clbits[op.memory[0]] = meas_res

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


class TestT2Star(QiskitTestCase):
    """Test T2Star experiment"""

    def test_t2star_run_end2end(self):
        """
        Run the T2 backend on all possible units
        """
        # For some reason, 'ps' was not precise enough - need to check this
        for unit in ["s", "ms", "us", "ns", "dt"]:
            if unit in ("s", "dt"):
                dt_factor = 1
            else:
                dt_factor = apply_prefix(1, unit)
            estimated_t2star = 20
            estimated_freq = 0.1
            # Set up the circuits
            qubit = 0
            if unit == "dt":
                delays = list(range(1, 46))
            else:
                delays = np.append(
                    (np.linspace(1.0, 15.0, num=15)).astype(float),
                    (np.linspace(16.0, 45.0, num=59)).astype(float),
                )

            # dummy numbers to avoid exception triggerring
            instruction_durations = [
                ("measure", [0], 3, unit),
                ("h", [0], 3, unit),
                ("p", [0], 3, unit),
                ("delay", [0], 3, unit),
            ]

            exp = T2StarExperiment(qubit, delays, unit=unit)
            exp.set_analysis_options(
                user_p0={
                    "A": 0.5,
                    "t2star": estimated_t2star,
                    "f": estimated_freq,
                    "phi": 0,
                    "B": 0.5,
                }
            )

            backend = T2starBackend(
                p0={
                    "a_guess": [0.5],
                    "t2star": [estimated_t2star],
                    "f_guess": [estimated_freq],
                    "phi_guess": [0.0],
                    "b_guess": [0.5],
                },
                initial_prob_plus=[0.0],
                readout0to1=[0.02],
                readout1to0=[0.02],
                dt_factor=dt_factor,
            )
            if unit == "dt":
                dt_factor = getattr(backend._configuration, "dt")

            # run circuits

            expdata = exp.run(
                backend=backend,
                # plot=False,
                instruction_durations=instruction_durations,
                shots=2000,
            )
            result = expdata.analysis_result(0)
            self.assertAlmostEqual(
                result["t2star_value"],
                estimated_t2star * dt_factor,
                delta=0.08 * result["t2star_value"],
            )
            self.assertAlmostEqual(
                result["frequency_value"],
                estimated_freq / dt_factor,
                delta=0.08 * result["frequency_value"],
            )
            self.assertEqual(
                result["quality"], "computer_good", "Result quality bad for unit " + str(unit)
            )

    def test_t2star_parallel(self):
        """
        Test parallel experiments of T2* using a simulator.
        """

        t2star = [30, 25]
        estimated_freq = [0.1, 0.12]
        delays = [list(range(1, 60)), list(range(1, 50))]

        exp0 = T2StarExperiment(0, delays[0])
        exp2 = T2StarExperiment(2, delays[1])
        par_exp = ParallelExperiment([exp0, exp2])

        p0 = {
            "a_guess": [0.5, None, 0.5],
            "t2star": [t2star[0], None, t2star[1]],
            "f_guess": [estimated_freq[0], None, estimated_freq[1]],
            "phi_guess": [0, None, 0],
            "b_guess": [0.5, None, 0.5],
        }
        backend = T2starBackend(p0)
        res = par_exp.run(
            backend=backend,
            # plot=False,
            shots=1000,
        )

        for i in range(2):
            sub_res = res.component_experiment_data(i).analysis_result(0)
            self.assertAlmostEqual(
                sub_res["t2star_value"], t2star[i], delta=0.08 * sub_res["t2star_value"]
            )
            self.assertAlmostEqual(
                sub_res["frequency_value"],
                estimated_freq[i],
                delta=0.08 * sub_res["frequency_value"],
            )
            self.assertEqual(
                sub_res["quality"],
                "computer_good",
                "Result quality bad for experiment on qubit " + str(i),
            )


if __name__ == "__main__":
    unittest.main()
