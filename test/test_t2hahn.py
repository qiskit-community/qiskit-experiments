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
Test T2Hahn experiment
"""

from test.base import QiskitExperimentsTestCase
import numpy as np
from ddt import ddt, data, unpack
from qiskit_experiments.framework import ParallelExperiment
from qiskit_experiments.library.characterization.t2hahn import T2Hahn
from qiskit_experiments.library.characterization import T2HahnAnalysis
from qiskit_experiments.test.t2hahn_backend import T2HahnBackend


@ddt
class TestT2Hahn(QiskitExperimentsTestCase):
    """Test T2Hahn experiment"""

    __tolerance__ = 0.1

    @data([0], [1], [2])
    @unpack
    def test_t2hahn_run_end2end(self, num_of_echoes: int):
        """
        Run the T2Hahn backend with 'num_of_echoes' echoes.
        """
        osc_freq = 0.1
        estimated_t2hahn = 20
        # Set up the circuits
        qubit = 0
        delays = np.append(
            (np.linspace(1.0, 15.0, num=15)).astype(float),
            (np.linspace(16.0, 45.0, num=59)).astype(float),
        )
        exp = T2Hahn(qubit=qubit, delays=delays, num_echoes=num_of_echoes)
        default_p0 = {
            "A": 0.5,
            "T2": estimated_t2hahn,
            "B": 0.5,
        }
        backend = T2HahnBackend(
            t2hahn=[estimated_t2hahn],
            frequency=[osc_freq],
            initialization_error=[0.0],
            readout0to1=[0.02],
            readout1to0=[0.02],
        )

        for _ in [default_p0, dict()]:
            exp.analysis.set_options(
                p0={"amp": 0.5, "tau": estimated_t2hahn, "base": 0.5}, plot=True
            )
            expdata = exp.run(backend=backend, shots=1000)
            self.assertExperimentDone(expdata, timeout=300)
            self.assertRoundTripSerializable(expdata, check_func=self.experiment_data_equiv)
            self.assertRoundTripPickle(expdata, check_func=self.experiment_data_equiv)
            result = expdata.analysis_results("T2")
            fitval = result.value
            if num_of_echoes != 0:
                self.assertEqual(result.quality, "good")
                self.assertAlmostEqual(fitval.n, estimated_t2hahn, delta=3)

    def test_t2hahn_parallel(self):
        """
        Test parallel experiments of T2Hahn using a simulator.
        """
        t2hahn = [30, 25]
        delays = [list(range(1, 60)), list(range(1, 50))]
        osc_freq = [0.11, 0.11]

        exp0 = T2Hahn(0, delays[0])
        exp2 = T2Hahn(2, delays[1])

        exp0.analysis.set_options(p0={"amp": 0.5, "tau": t2hahn[0], "base": 0.5}, plot=True)
        exp2.analysis.set_options(p0={"amp": 0.5, "tau": t2hahn[1], "base": 0.5}, plot=True)

        par_exp = ParallelExperiment([exp0, exp2])

        p0 = {
            "A": [0.5, None, 0.5],
            "T2": [t2hahn[0], None, t2hahn[1]],
            "frequency": [osc_freq[0], None, osc_freq[1]],
            "B": [0.5, None, 0.5],
        }

        backend = T2HahnBackend(
            t2hahn=p0["T2"],
            frequency=p0["frequency"],
            initialization_error=[0.0],
            readout0to1=[0.02],
            readout1to0=[0.02],
        )
        expdata = par_exp.run(backend=backend, shots=1024).block_for_results()

        for i in range(2):
            res_t2 = expdata.child_data(i).analysis_results("T2")

            fitval = res_t2.value
            self.assertEqual(res_t2.quality, "good")
            self.assertAlmostEqual(fitval.n, t2hahn[i], delta=3)

    def test_t2hahn_concat_2_experiments(self):
        """
        Concatenate the data from 2 separate experiments.
        """
        estimated_t2hahn = 30
        # First experiment
        qubit = 0
        delays0 = list(range(1, 60, 2))
        osc_freq = 0.08

        exp0 = T2Hahn(qubit, delays0)
        exp0.analysis.set_options(p0={"amp": 0.5, "tau": estimated_t2hahn, "base": 0.5}, plot=True)
        backend = T2HahnBackend(
            t2hahn=[estimated_t2hahn],
            frequency=[osc_freq],
            initialization_error=[0.0],
            readout0to1=[0.02],
            readout1to0=[0.02],
        )

        # run circuits
        expdata0 = exp0.run(backend=backend, shots=1000)
        expdata0.block_for_results()

        res_t2_0 = expdata0.analysis_results("T2")
        # second experiment
        delays1 = list(range(2, 65, 2))
        exp1 = T2Hahn(qubit, delays1)
        exp1.analysis.set_options(p0={"amp": 0.5, "tau": estimated_t2hahn, "base": 0.5}, plot=True)
        expdata1 = exp1.run(backend=backend, analysis=None, shots=1000).block_for_results()
        expdata1.add_data(expdata0.data())
        exp1.analysis.run(expdata1)

        res_t2_1 = expdata1.analysis_results("T2")

        fitval = res_t2_1.value
        self.assertEqual(res_t2_1.quality, "good")
        self.assertAlmostEqual(fitval.n, estimated_t2hahn, delta=3)

        self.assertAlmostEqual(
            fitval.n,
            estimated_t2hahn,
            delta=TestT2Hahn.__tolerance__ * res_t2_1.value.n,
        )

        self.assertLessEqual(res_t2_1.value.s, res_t2_0.value.s)
        self.assertEqual(len(expdata1.data()), len(delays0) + len(delays1))

    def test_experiment_config(self):
        """Test converting to and from config works"""
        exp = T2Hahn(0, [1, 2, 3, 4, 5])
        loaded_exp = T2Hahn.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertTrue(self.json_equiv(exp, loaded_exp))

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""

        delays0 = list(range(1, 60, 2))

        exp = T2Hahn(0, delays0)
        self.assertRoundTripSerializable(exp, self.json_equiv)

        osc_freq = 0.08
        estimated_t2hahn = 30
        backend = T2HahnBackend(
            t2hahn=[estimated_t2hahn],
            frequency=[osc_freq],
            initialization_error=[0.0],
            readout0to1=[0.02],
            readout1to0=[0.02],
        )
        exp.analysis.set_options(p0={"amp": 0.5, "tau": estimated_t2hahn, "base": 0.5}, plot=False)
        expdata = exp.run(backend=backend, shots=1000).block_for_results()
        self.assertExperimentDone(expdata)

        # Checking serialization of the experiment data
        self.assertRoundTripSerializable(expdata, self.experiment_data_equiv)

        # Checking serialization of the analysis
        self.assertRoundTripSerializable(expdata.analysis_results(1), self.analysis_result_equiv)

    def test_analysis_config(self):
        """ "Test converting analysis to and from config works"""
        analysis = T2HahnAnalysis()
        loaded = T2HahnAnalysis.from_config(analysis.config())
        self.assertNotEqual(analysis, loaded)
        self.assertEqual(analysis.config(), loaded.config())
