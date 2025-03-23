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

"""Test the fine amplitude characterization and calibration experiments."""
from test.base import QiskitExperimentsTestCase

import numpy as np
from ddt import ddt, data

from qiskit.circuit.library import XGate, SXGate
from qiskit_ibm_runtime.fake_provider import FakeArmonkV2

from qiskit_experiments.library import (
    FineXAmplitude,
    FineSXAmplitude,
)
from qiskit_experiments.test.mock_iq_backend import MockIQBackend
from qiskit_experiments.test.mock_iq_helpers import MockIQFineAmpHelper as FineAmpHelper


@ddt
class TestFineAmpEndToEnd(QiskitExperimentsTestCase):
    """Test the fine amplitude experiment."""

    @data(0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
    def test_end_to_end_under_rotation(self, pi_ratio):
        """Test the experiment end to end."""

        amp_exp = FineXAmplitude([0])

        error = -np.pi * pi_ratio
        backend = MockIQBackend(FineAmpHelper(error, np.pi, "x"))

        expdata = amp_exp.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results("d_theta", dataframe=True).iloc[0]
        d_theta = result.value.n

        tol = 0.04

        self.assertAlmostEqual(d_theta, error, delta=tol)
        self.assertEqual(result.quality, "good")

    @data(0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
    def test_end_to_end_over_rotation(self, pi_ratio):
        """Test the experiment end to end."""

        amp_exp = FineXAmplitude([0])

        error = np.pi * pi_ratio
        backend = MockIQBackend(FineAmpHelper(error, np.pi, "x"))
        expdata = amp_exp.run(backend)
        self.assertExperimentDone(expdata)
        result = expdata.analysis_results("d_theta", dataframe=True).iloc[0]
        d_theta = result.value.n

        tol = 0.04

        self.assertAlmostEqual(d_theta, error, delta=tol)
        self.assertEqual(result.quality, "good")

    def test_circuits_serialization(self):
        """Test circuits serialization of the experiment."""
        backend = FakeArmonkV2()
        amp_exp = FineXAmplitude([0], backend=backend)
        self.assertRoundTripSerializable(amp_exp._transpiled_circuits())


class TestFineAmplitudeCircuits(QiskitExperimentsTestCase):
    """Test the circuits."""

    def test_xp(self):
        """Test a circuit with the x gate."""

        amp_cal = FineXAmplitude([0])
        circs = amp_cal.circuits()

        self.assertTrue(circs[0].data[0].operation.name == "measure")
        self.assertTrue(circs[1].data[0].operation.name == "x")

        for idx, circ in enumerate(circs[2:]):
            self.assertTrue(circ.data[0].operation.name == "sx")
            self.assertEqual(circ.count_ops().get("x", 0), idx + 1)

    def test_x90p(self):
        """Test circuits with an x90p pulse."""

        amp_cal = FineSXAmplitude([0])

        expected = [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 21, 23, 25]
        for idx, circ in enumerate(amp_cal.circuits()):
            self.assertEqual(circ.count_ops().get("sx", 0), expected[idx])


@ddt
class TestSpecializations(QiskitExperimentsTestCase):
    """Test the options of the specialized classes."""

    def test_fine_x_amp(self):
        """Test the fine X amplitude."""

        exp = FineXAmplitude([0])

        self.assertTrue(exp.experiment_options.add_cal_circuits)
        self.assertDictEqual(
            exp.analysis.options.fixed_parameters,
            {"angle_per_gate": np.pi, "phase_offset": np.pi / 2},
        )
        self.assertEqual(exp.experiment_options.gate, XGate())

    def test_x_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = FineXAmplitude([0])
        self.assertRoundTripSerializable(exp)

    def test_fine_sx_amp(self):
        """Test the fine SX amplitude."""

        exp = FineSXAmplitude([0])

        self.assertFalse(exp.experiment_options.add_cal_circuits)

        expected = [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 21, 23, 25]
        self.assertEqual(exp.experiment_options.repetitions, expected)
        self.assertDictEqual(
            exp.analysis.options.fixed_parameters,
            {"angle_per_gate": np.pi / 2, "phase_offset": np.pi},
        )
        self.assertEqual(exp.experiment_options.gate, SXGate())

    def test_sx_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = FineSXAmplitude([0])
        self.assertRoundTripSerializable(exp)
