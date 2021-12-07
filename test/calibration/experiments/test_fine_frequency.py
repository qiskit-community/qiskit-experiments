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

"""Test the fine frequency characterization and calibration experiments."""

from test.base import QiskitExperimentsTestCase
import unittest
import numpy as np
from ddt import ddt, data

from qiskit import transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import XGate, SXGate
from qiskit.pulse import DriveChannel, Drag
import qiskit.pulse as pulse

from qiskit_experiments.library import (
    FineFrequency,
)
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
from qiskit_experiments.calibration_management import BackendCalibrations
from qiskit_experiments.test.mock_iq_backend import MockFineFreq


@ddt
class TestFineFreqEndToEnd(QiskitExperimentsTestCase):
    """Test the fine freq experiment."""

    def setUp(self):
        """Setup for the test."""
        super().setUp()
        self.inst_map = pulse.InstructionScheduleMap()

        with pulse.build(name="sx") as sx_sxhed:
            pulse.play(pulse.Gaussian(160, 0.5, 40), pulse.DriveChannel(0))

        self.inst_map.add("sx", 0, sx_sxhed)

    @data(-0.5e6, -0.1e6, 0.1e6, 0.5e6)
    def test_end_to_end_under_rotation(self, freq_shift):
        """Test the experiment end to end."""

        backend = MockFineFreq(freq_shift)

        freq_exp = FineFrequency(0, backend)
        freq_exp.set_transpile_options(inst_map=self.inst_map)
        #freq_exp.set_experiment_options(add_sx=True)
        #freq_exp.analysis.set_options(angle_per_gate=np.pi, phase_offset=np.pi / 2)

        expdata = freq_exp.run().block_for_results()
        result = expdata.analysis_results(1)
        d_theta = result.value.value

        tol = 0.04

        print(d_theta)

        #self.assertAlmostEqual(d_theta, error, delta=tol)
        #self.assertEqual(result.quality, "good")