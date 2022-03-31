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
Test T2Ramsey experiment
"""
from test.base import QiskitExperimentsTestCase
import numpy as np
from qiskit_experiments.framework import ParallelExperiment
from qiskit_experiments.library import T2Ramsey
from qiskit_experiments.library.characterization import T2RamseyAnalysis
from qiskit_experiments.test.t2ramsey_backend import T2RamseyBackend


class TestT2Ramsey(QiskitExperimentsTestCase):
    """Test T2Ramsey experiment"""

    __tolerance__ = 0.1

    def test_t2ramsey_run_end2end(self):
        """
        Run the T2Ramsey backend
        """
        osc_freq = 0.1
        estimated_t2ramsey = 20

        # induce error
        estimated_freq = osc_freq * 1.001

        # Set up the circuits
        qubit = 0
        delays = np.append(
            (np.linspace(1.0, 15.0, num=15)).astype(float),
            (np.linspace(16.0, 45.0, num=59)).astype(float),
        )
        exp = T2Ramsey(qubit, delays, osc_freq=osc_freq)
        default_p0 = {
            "amp": 0.5,
            "tau": estimated_t2ramsey,
            "freq": estimated_freq,
            "phi": 0,
            "base": 0.5,
        }
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
        )
        for user_p0 in [default_p0, dict()]:
            exp.analysis.set_options(p0=user_p0)
            expdata = exp.run(backend=backend, shots=2000)
            self.assertExperimentDone(expdata)
            self.assertRoundTripSerializable(expdata, check_func=self.experiment_data_equiv)
            self.assertRoundTripPickle(expdata, check_func=self.experiment_data_equiv)

            result = expdata.analysis_results("T2star")
            self.assertAlmostEqual(
                result.value.n,
                estimated_t2ramsey,
                delta=TestT2Ramsey.__tolerance__ * result.value.n,
            )
            self.assertEqual(result.quality, "good", "Result quality bad")
            result = expdata.analysis_results("Frequency")
            self.assertAlmostEqual(
                result.value.n,
                estimated_freq,
                delta=TestT2Ramsey.__tolerance__ * result.value.n,
            )
            self.assertEqual(result.quality, "good", "Result quality bad")

    def test_t2ramsey_parallel(self):
        """
        Test parallel experiments of T2Ramsey using a simulator.
        """
        t2ramsey = [30, 25]
        estimated_freq = [0.1, 0.12]
        delays = [list(range(1, 60)), list(range(1, 50))]

        osc_freq = [0.11, 0.11]

        exp0 = T2Ramsey(0, delays[0], osc_freq=osc_freq[0])
        exp2 = T2Ramsey(2, delays[1], osc_freq=osc_freq[1])
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
        self.assertExperimentDone(expdata)

        for i in range(2):
            res_t2star = expdata.child_data(i).analysis_results("T2star")
            self.assertAlmostEqual(
                res_t2star.value.n,
                t2ramsey[i],
                delta=TestT2Ramsey.__tolerance__ * res_t2star.value.n,
            )
            self.assertEqual(
                res_t2star.quality, "good", "Result quality bad for experiment on qubit " + str(i)
            )
            res_freq = expdata.child_data(i).analysis_results("Frequency")
            self.assertAlmostEqual(
                res_freq.value.n,
                estimated_freq[i],
                delta=TestT2Ramsey.__tolerance__ * res_freq.value.n,
            )
            self.assertEqual(
                res_freq.quality, "good", "Result quality bad for experiment on qubit " + str(i)
            )

    def test_t2ramsey_concat_2_experiments(self):
        """
        Concatenate the data from 2 separate experiments
        """
        estimated_t2ramsey = 30
        estimated_freq = 0.09
        # First experiment
        qubit = 0
        delays0 = list(range(1, 60, 2))
        osc_freq = 0.08

        exp0 = T2Ramsey(qubit, delays0, osc_freq=osc_freq)
        default_p0 = {
            "A": 0.5,
            "T2star": estimated_t2ramsey,
            "f": estimated_freq,
            "phi": 0,
            "B": 0.5,
        }
        exp0.analysis.set_options(p0=default_p0)
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
        )

        # run circuits
        expdata0 = exp0.run(backend=backend, shots=1000)
        self.assertExperimentDone(expdata0)
        res_t2star_0 = expdata0.analysis_results("T2star")

        # second experiment
        delays1 = list(range(2, 65, 2))
        exp1 = T2Ramsey(qubit, delays1)
        exp1.analysis.set_options(p0=default_p0)
        expdata1 = exp1.run(backend=backend, analysis=None, shots=1000)
        self.assertExperimentDone(expdata1)
        expdata1.add_data(expdata0.data())
        exp1.analysis.run(expdata1)

        res_t2star_1 = expdata1.analysis_results("T2star")
        res_freq_1 = expdata1.analysis_results("Frequency")

        self.assertAlmostEqual(
            res_t2star_1.value.n,
            estimated_t2ramsey,
            delta=TestT2Ramsey.__tolerance__ * res_t2star_1.value.n,
        )
        self.assertAlmostEqual(
            res_freq_1.value.n,
            estimated_freq,
            delta=TestT2Ramsey.__tolerance__ * res_freq_1.value.n,
        )
        self.assertLessEqual(res_t2star_1.value.s, res_t2star_0.value.s)
        self.assertEqual(len(expdata1.data()), len(delays0) + len(delays1))

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = T2Ramsey(0, [1, 2, 3, 4, 5])
        loaded_exp = T2Ramsey.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = T2Ramsey(0, [1, 2, 3, 4, 5])
        self.assertRoundTripSerializable(exp, self.json_equiv)

    def test_analysis_config(self):
        """ "Test converting analysis to and from config works"""
        analysis = T2RamseyAnalysis()
        loaded = T2RamseyAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())
