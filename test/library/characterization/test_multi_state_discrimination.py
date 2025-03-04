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

"""Test the multi state discrimination experiments."""
from functools import wraps
from test.base import QiskitExperimentsTestCase
from unittest import SkipTest

from ddt import ddt, data

from qiskit.exceptions import MissingOptionalLibraryError

from qiskit_experiments.library import MultiStateDiscrimination
from qiskit_experiments.test.mock_iq_backend import MockMultiStateBackend

from qiskit_experiments.framework.package_deps import HAS_SKLEARN


def requires_sklearn(func):
    """Decorator to check for SKLearn."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            HAS_SKLEARN.require_now("SKLearn discriminator testing")
        except MissingOptionalLibraryError as exc:
            raise SkipTest("SKLearn is required for test.") from exc

        func(*args, **kwargs)

    return wrapper


@ddt
class TestMultiStateDiscrimination(QiskitExperimentsTestCase):
    """Tests of the multi state discrimination experiment."""

    def setUp(self):
        """Setup test variables."""
        super().setUp()

        self.backend = MockMultiStateBackend(iq_centers=[1, 1j, -1], rng_seed=1234)

        # Build x12 schedule
        self.qubit = 0

    @data(2, 3)
    @requires_sklearn
    def test_circuit_generation(self, n_states):
        """Test the experiment circuit generation"""
        exp = MultiStateDiscrimination([self.qubit], n_states=n_states, backend=self.backend)
        self.assertEqual(len(exp.circuits()), n_states)

        # check the metadata
        self.assertEqual(exp.circuits()[-1].metadata["label"], n_states - 1)

    @data(2, 3)
    @requires_sklearn
    def test_discrimination_analysis(self, n_states):
        """Test the discrimination analysis"""
        exp = MultiStateDiscrimination([self.qubit], n_states=n_states, backend=self.backend)

        exp_data = exp.run()

        fidelity = exp_data.analysis_results("fidelity").value

        self.assertGreaterEqual(fidelity, 0.93)

        # check that the discriminator differentiates n different states
        discrim_lbls = exp_data.analysis_results("discriminator_config").value["attributes"][
            "classes_"
        ]
        self.assertEqual(len(discrim_lbls), n_states)

    def test_circuit_roundtrip_serializable(self):
        """Test round trip JSON serialization for the experiment circuits."""
        exp = MultiStateDiscrimination([self.qubit], n_states=3, backend=self.backend)
        self.assertRoundTripSerializable(exp._transpiled_circuits())
