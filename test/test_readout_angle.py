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
Test readout angle experiment
"""

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit_experiments.framework import ExperimentData, ParallelExperiment
from qiskit_experiments.library import ReadoutAngle
from qiskit_experiments.library.characterization import ReadoutAngleAnalysis
from qiskit_experiments.test.mock_iq_backend import MockIQBackend

class ReadoutAngleBackend(MockIQBackend):
    def _compute_probability(self, circ):
        return 1 - circ.metadata["xval"]

class TestReadoutAngle(QiskitTestCase):
    """
    Test the readout angle experiment
    """

    def test_readout_angle_end2end(self):
        """
        Test readout angle experiment using a simulator.
        """
        backend = ReadoutAngleBackend(iq_cluster_centers=(5.0, 5.0, -3.0, 3.0))
        exp = ReadoutAngle(0)
        expdata = exp.run(backend, shots=100000).block_for_results()
        res = expdata.analysis_results(0)
        self.assertAlmostEqual(res.value, np.pi/2, places=2)

    def test_readout_angle_parallel(self):
        """
        Test parallel experiments of readout angle using a simulator.
        """
        backend = ReadoutAngleBackend(iq_cluster_centers=(5.0, 5.0, -3.0, 3.0))
        exp2 = ReadoutAngle(2)
        exp0 = ReadoutAngle(0)
        parexp = ParallelExperiment([exp2, exp0])
        expdata = parexp.run(backend, shots=100000).block_for_results()

        for i in range(2):
            res = expdata.child_data(i).analysis_results(0)
            self.assertAlmostEqual(res, np.pi/2, places=2)

    def test_t1_parallel_different_analysis_options(self):
        """
        Test parallel experiments of T1 using a simulator, for the case where
        the sub-experiments have different analysis options
        """

        t1 = 25
        delays = list(range(1, 40, 3))

        exp0 = T1(0, delays)
        exp0.set_analysis_options(p0={"tau": 30})
        exp1 = T1(1, delays)
        exp1.set_analysis_options(p0={"tau": 1000000})

        par_exp = ParallelExperiment([exp0, exp1])
        res = par_exp.run(T1Backend([t1, t1]))
        res.block_for_results()

        sub_res = []
        for i in range(2):
            sub_res.append(res.child_data(i).analysis_results("T1"))

        self.assertEqual(sub_res[0].quality, "good")
        self.assertAlmostEqual(sub_res[0].value.value, t1, delta=3)
        self.assertEqual(sub_res[1].quality, "bad")

