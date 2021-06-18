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

"""Test rough drag calibration experiment."""

from typing import Tuple
import numpy as np

from qiskit import QuantumCircuit

from qiskit.test import QiskitTestCase
from qiskit.circuit import Parameter
from qiskit.qobj.utils import MeasLevel
from qiskit.pulse import DriveChannel, Drag
import qiskit.pulse as pulse

from qiskit_experiments import ExperimentData
from qiskit_experiments.calibration.experiments.rough_drag import RoughDrag
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.data_processing.nodes import Probability
from qiskit_experiments.test.mock_iq_backend import IQTestBackend, DragBackend


class TestRoughDragEndToEnd(QiskitTestCase):
    """Test the rabi experiment."""

    def setUp(self):
        """Setup some schedules."""
        super().setUp()

        beta = Parameter("β")

        with pulse.build(name="xp") as xp:
            pulse.play(Drag(duration=160, amp=0.208519, sigma=40, beta=beta), DriveChannel(0))

        with pulse.build(name="xm") as xm:
            pulse.play(Drag(duration=160, amp=-0.208519, sigma=40, beta=beta), DriveChannel(0))

        self.x_minus = xm
        self.x_plus = xp

    def test_end_to_end(self):
        """Test the Rabi experiment end to end."""

        test_tol = 0.01
        backend = DragBackend()

        rabi = RoughDrag(3)
        rabi.set_experiment_options(xp=self.x_plus, xm=self.x_minus)
        result = rabi.run(backend).analysis_result(0)

        self.assertTrue(abs(result["popt"][6] - backend.ideal_beta) < test_tol)
