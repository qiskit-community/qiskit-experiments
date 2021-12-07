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
from qiskit_experiments.library import ReadoutAngle
from qiskit_experiments.test.mock_iq_backend import MockIQBackend


class ReadoutAngleBackend(MockIQBackend):
    """
    Mock IQ backend tailored to the readout angle test
    """

    def _compute_probability(self, circuit):
        return 1 - circuit.metadata["xval"]


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
        self.assertAlmostEqual(res.value % (2 * np.pi), np.pi / 2, places=2)

        backend = ReadoutAngleBackend(iq_cluster_centers=(5.0, 5.0, 0, -3.0))
        exp = ReadoutAngle(0)
        expdata = exp.run(backend, shots=100000).block_for_results()
        res = expdata.analysis_results(0)
        self.assertAlmostEqual(res.value % (2 * np.pi), 15 * np.pi / 8, places=2)
