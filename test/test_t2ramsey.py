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
import numpy as np

from qiskit.utils import apply_prefix
from qiskit.providers import BackendV1
from qiskit.providers.options import Options
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit.test import QiskitTestCase
from qiskit_experiments.framework import ParallelExperiment
from qiskit_experiments.library import T2Ramsey
from qiskit_experiments.test.utils import FakeJob


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
        conversion_factor=1,
    ):
        """
        Initialize the T2Ramsey backend
        """
        dt_factor_in_ns = conversion_factor * 1e9 if conversion_factor is not None else None
        configuration = QasmBackendConfiguration(
            backend_name="T2Ramsey_simulator",
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

        self._t2ramsey = p0["T2star"]
        self._a_param = p0["A"]
        self._freq = p0["f"]
        self._phi = p0["phi"]
        self._b_param = p0["B"]
        self._initial_prob_plus = initial_prob_plus
        self._readout0to1 = readout0to1
        self._readout1to0 = readout1to0
        self._dt_factor = conversion_factor
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
                        freq = self._freq[qubit] / self._dt_factor

                        prob_plus[qubit] = (
                            self._a_param[qubit]
                            * np.exp(-delay / t2ramsey)
                            * np.cos(2 * np.pi * freq * delay + self._phi[qubit])
                            + self._b_param[qubit]
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

        return FakeJob(self, result=Result.from_dict(result))


class TestT2Ramsey(QiskitTestCase):
    """Test T2Ramsey experiment"""

    def test_t2ramsey_run_end2end(self):
        """
        Run the T2Ramsey backend on all possible units
        """
        for unit in ["s", "ms", "us", "ns", "dt"]:
            if unit in ("s", "dt"):
                dt_factor = 1
            else:
                dt_factor = apply_prefix(1, unit)
            osc_freq = 0.1
            estimated_t2ramsey = 20
            estimated_freq = 0.11
            # Set up the circuits
            qubit = 0
            if unit == "dt":  # dt requires integer values for delay
                delays = list(range(1, 46))
            else:
                delays = np.append(
                    (np.linspace(1.0, 15.0, num=15)).astype(float),
                    (np.linspace(16.0, 45.0, num=59)).astype(float),
                )

            exp = T2Ramsey(qubit, delays, unit=unit, osc_freq=osc_freq)
            default_p0 = {
                "A": 0.5,
                "T2star": estimated_t2ramsey,
                "f": estimated_freq,
                "phi": 0,
                "B": 0.5,
            }
            for user_p0 in [default_p0, None]:
                exp.set_analysis_options(user_p0=user_p0, plot=True)
                backend = T2RamseyBackend(
                    p0={
                        "A": [0.5],
                        "T2star": [estimated_t2ramsey],
                        "f": [estimated_freq],
                        "phi": [0.0],
                        "B": [0.5],
                    },
                    initial_prob_plus=[0.0],
                    readout0to1=[0.02],
                    readout1to0=[0.02],
                    conversion_factor=dt_factor,
                )

            expdata = exp.run(backend=backend, shots=2000)
            expdata.block_for_results()  # Wait for job/analysis to finish.
            result = expdata.analysis_results()
            self.assertAlmostEqual(
                result[0].value.value,
                estimated_t2ramsey * dt_factor,
                delta=3 * dt_factor,
            )
            self.assertAlmostEqual(
                result[1].value.value,
                estimated_freq / dt_factor,
                delta=3 / dt_factor,
            )
            for res in result:
                self.assertEqual(res.quality, "good", "Result quality bad for unit " + str(unit))

    def test_t2ramsey_parallel(self):
        """
        Test parallel experiments of T2Ramsey using a simulator.
        """
        t2ramsey = [30, 25]
        estimated_freq = [0.1, 0.12]
        delays = [list(range(1, 60)), list(range(1, 50))]
        dt_factor = 1e-6
        osc_freq = 0.1

        exp0 = T2Ramsey(0, delays[0], osc_freq=osc_freq)
        exp2 = T2Ramsey(2, delays[1], osc_freq=osc_freq)
        par_exp = ParallelExperiment([exp0, exp2])

        p0 = {
            "A": [0.5, None, 0.5],
            "T2star": [t2ramsey[0], None, t2ramsey[1]],
            "f": [estimated_freq[0], None, estimated_freq[1]],
            "phi": [0, None, 0],
            "B": [0.5, None, 0.5],
        }

        backend = T2RamseyBackend(p0)
        expdata = par_exp.run(backend=backend, shots=1000)
        expdata.block_for_results()

        for i in range(2):
            sub_res = expdata.component_experiment_data(i).analysis_results()
            self.assertAlmostEqual(sub_res[0].value.value, t2ramsey[i], delta=3)
            self.assertAlmostEqual(
                sub_res[1].value.value,
                estimated_freq[i] / dt_factor,
                delta=3 / dt_factor,
            )
            for res in sub_res:
                self.assertEqual(
                    res.quality,
                    "good",
                    "Result quality bad for experiment on qubit " + str(i),
                )

    def test_t2ramsey_concat_2_experiments(self):
        """
        Concatenate the data from 2 separate experiments
        """
        unit = "s"
        dt_factor = 1
        estimated_t2ramsey = 30
        estimated_freq = 0.09
        # First experiment
        qubit = 0
        delays0 = list(range(1, 60, 2))
        osc_freq = 0.08

        exp0 = T2Ramsey(qubit, delays0, unit=unit, osc_freq=osc_freq)
        default_p0 = {
            "A": 0.5,
            "T2star": estimated_t2ramsey,
            "f": estimated_freq,
            "phi": 0,
            "B": 0.5,
        }
        exp0.set_analysis_options(user_p0=default_p0)
        backend = T2RamseyBackend(
            p0={
                "A": [0.5],
                "T2star": [estimated_t2ramsey],
                "f": [estimated_freq],
                "phi": [0.0],
                "B": [0.5],
            },
            initial_prob_plus=[0.0],
            readout0to1=[0.02],
            readout1to0=[0.02],
            conversion_factor=1,
        )

        # run circuits
        expdata0 = exp0.run(backend=backend, shots=1000)
        expdata0.block_for_results()
        results0 = expdata0.analysis_results()

        # second experiment
        delays1 = list(range(2, 65, 2))
        exp1 = T2Ramsey(qubit, delays1, unit=unit)
        exp1.set_analysis_options(user_p0=default_p0)
        expdata1 = exp1.run(backend=backend, experiment_data=expdata0, shots=1000)
        expdata1.block_for_results()
        results1 = expdata1.analysis_results()

        self.assertAlmostEqual(
            results1[0].value.value,
            estimated_t2ramsey * dt_factor,
            delta=3 * dt_factor,
        )
        self.assertAlmostEqual(
            results1[1].value.value, estimated_freq / dt_factor, delta=3 / dt_factor
        )
        self.assertLessEqual(results1[0].value.stderr, results0[0].value.stderr)
        self.assertEqual(len(expdata1.data()), len(delays0) + len(delays1))
