# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test ZZ Phase experiments."""

from itertools import product

from test.base import QiskitExperimentsTestCase

from ddt import ddt, idata, named_data, unpack

from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeVigoV2

from qiskit_experiments.library import ZZRamsey
from qiskit_experiments.test.zzramsey_test_backend import ZZRamseyTestBackend


@ddt
class TestZZRamsey(QiskitExperimentsTestCase):
    """Tests for the ZZ Ramsey experiment."""

    test_tol = 0.05

    @named_data(
        ["no_backend", None], ["fake_backend", FakeVigoV2()], ["aer_backend", AerSimulator()]
    )
    def test_circuits(self, backend):
        """Test circuit generation"""
        t_min = 0
        t_max = 5e-6
        num = 50

        ramsey_min_max = ZZRamsey(
            (0, 1),
            backend,
            min_delay=t_min,
            max_delay=t_max,
            num_delays=num,
        )
        ramsey_with_delays = ZZRamsey(
            (0, 1),
            backend,
            delays=ramsey_min_max.delays(),
        )

        # Check that the right number of circuits are generated
        self.assertEqual(len(ramsey_min_max.circuits()), num * 2)
        # Test setting min/max and setting exact delays give same results
        self.assertEqual(ramsey_min_max.circuits(), ramsey_with_delays.circuits())

    @idata(product([2e5, -3e5], [4, 5]))
    @unpack
    def test_end_to_end(self, zz_freq, num_rotations):
        """Test that we can run on a mock backend and perform a fit."""
        backend = ZZRamseyTestBackend(
            zz_frequency=zz_freq,
            t2hahn=10e-6,
            initialization_error=0.05,
        )

        ramsey = ZZRamsey((0, 1), backend, num_rotations=num_rotations)
        test_data = ramsey.run()
        self.assertExperimentDone(test_data)

        result = test_data.analysis_results("zz")
        meas_shift = result.value.n
        self.assertLess(abs(meas_shift - zz_freq), abs(self.test_tol * zz_freq))
        self.assertEqual(result.quality, "good")

    def test_experiment_config(self):
        """Test config roundtrips"""
        exp = ZZRamsey((0, 1))
        loaded_exp = ZZRamsey.from_config(exp.config())
        self.assertNotEqual(exp, loaded_exp)
        self.assertEqualExtended(exp, loaded_exp)

    def test_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        exp = ZZRamsey((0, 1))
        self.assertRoundTripSerializable(exp)

    def test_circuit_roundtrip_serializable(self):
        """Test round trip JSON serialization"""
        # backend is needed for serialization of the metadata in the experiment.
        backend = FakeVigoV2()
        t_min = 0
        t_max = 5e-6
        num = 50

        ramsey_min_max = ZZRamsey(
            (0, 1),
            backend,
            min_delay=t_min,
            max_delay=t_max,
            num_delays=num,
        )
        # Check that the circuit are serializable
        self.assertRoundTripSerializable(ramsey_min_max._transpiled_circuits())
