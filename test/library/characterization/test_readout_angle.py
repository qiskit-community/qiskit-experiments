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

from test.base import QiskitExperimentsTestCase
import numpy as np

from qiskit_experiments.framework import MeasLevel
from qiskit_experiments.library import ReadoutAngle
from qiskit_experiments.test.mock_iq_backend import MockIQBackend
from qiskit_experiments.test.mock_iq_helpers import MockIQReadoutAngleHelper


class TestReadoutAngle(QiskitExperimentsTestCase):
    """
    Test the readout angle experiment
    """

    def test_readout_angle_end2end(self):
        """
        Test readout angle experiment using a simulator.
        """

        backend = MockIQBackend(
            MockIQReadoutAngleHelper(iq_cluster_centers=[((-3.0, 3.0), (5.0, 5.0))]),
        )
        exp = ReadoutAngle([0])
        expdata = exp.run(backend, shots=10000)
        self.assertExperimentDone(expdata)

        res = expdata.analysis_results("readout_angle", dataframe=True).iloc[0]
        self.assertAlmostEqual(res.value % (2 * np.pi), np.pi / 2, places=2)

        backend = MockIQBackend(
            MockIQReadoutAngleHelper(iq_cluster_centers=[((0, -3.0), (5.0, 5.0))]),
        )
        exp = ReadoutAngle([0])
        expdata = exp.run(backend, shots=10000)
        self.assertExperimentDone(expdata)
        res = expdata.analysis_results("readout_angle", dataframe=True).iloc[0]
        self.assertAlmostEqual(res.value % (2 * np.pi), 15 * np.pi / 8, places=2)

    def test_kerneled_expdata_serialization(self):
        """Test experiment data and analysis data JSON serialization"""
        backend = MockIQBackend(
            MockIQReadoutAngleHelper(iq_cluster_centers=[((-3.0, 3.0), (5.0, 5.0))]),
        )

        exp = ReadoutAngle([0])

        # Checking serialization of the experiment obj
        self.assertRoundTripSerializable(exp._transpiled_circuits())

        exp.set_run_options(meas_level=MeasLevel.KERNELED, shots=1024)
        expdata = exp.run(backend)
        self.assertExperimentDone(expdata)

        # Checking serialization of the experiment data
        self.assertRoundTripSerializable(expdata)
