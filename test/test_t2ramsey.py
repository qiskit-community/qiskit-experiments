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
from qiskit_experiments.test.t2ramsey_backend import T2RamseyBackend

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
