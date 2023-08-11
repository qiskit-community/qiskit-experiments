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
from test.data_processing import BaseDataProcessorTest
from unittest import SkipTest
import numpy as np

from ddt import ddt, data

from qiskit import pulse
from qiskit.exceptions import MissingOptionalLibraryError

from qiskit_experiments.library import MultiStateDiscrimination
from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend
from qiskit_experiments.data_processing import SkQDA
from qiskit_experiments.data_processing.nodes import DiscriminatorNode

from qiskit_experiments.warnings import HAS_SKLEARN


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
class TestMultiStateDiscrimination(BaseDataProcessorTest):
    """Tests of the multi state discrimination experiment."""

    def setUp(self):
        """Setup test variables."""
        super().setUp()

        self.backend = SingleTransmonTestBackend(noise=False, seed=0)

        # Build x12 schedule
        self.qubit = 0

        anharm = self.backend.anharmonicity

        d0 = pulse.DriveChannel(self.qubit)

        sch_map = self.backend.defaults().instruction_schedule_map
        pulse_x = sch_map.get("x", (self.qubit,)).instructions[0][1].pulse
        amp_x = pulse_x.amp
        dur_x = pulse_x.duration
        sigma_x = pulse_x.sigma
        beta_x = pulse_x.beta
        with pulse.build(name="x12") as x12:
            pulse.shift_frequency(anharm, d0)
            pulse.play(
                pulse.Gaussian(dur_x, amp_x * self.backend.rabi_rate_12, sigma_x, beta_x), d0
            )
            pulse.shift_frequency(-anharm, d0)

        self.schedules = {"x12": x12}

    @data(2, 3)
    @requires_sklearn
    def test_circuit_generation(self, n_states):
        """Test the experiment circuit generation"""
        exp = MultiStateDiscrimination(
            [self.qubit], n_states=n_states, backend=self.backend, schedules=self.schedules
        )
        self.assertEqual(len(exp.circuits()), n_states)

        # check the metadata
        self.assertEqual(exp.circuits()[-1].metadata["label"], str(n_states - 1))

    @data(2, 3)
    @requires_sklearn
    def test_discrimination_analysis(self, n_states):
        """Test the discrimination analysis"""
        exp = MultiStateDiscrimination(
            [self.qubit], n_states=n_states, backend=self.backend, schedules=self.schedules
        )

        exp_data = exp.run()

        fidelity = exp_data.analysis_results("fidelity").value

        self.assertGreaterEqual(fidelity, 0.96)

        # check that the discriminator differentiates n different states
        discrim_lbls = exp_data.analysis_results("discriminator_config").value["attributes"][
            "classes_"
        ]
        self.assertEqual(len(discrim_lbls), n_states)

    @requires_sklearn
    def test_discriminator_data_processing(self):
        """Test that the discriminator experiment works with the discriminator node."""
        discriminator = MultiStateDiscrimination([self.qubit], n_states=2, backend=self.backend)
        discriminator_data = discriminator.run().block_for_results()
        qda = SkQDA.from_config(discriminator_data.analysis_results("discriminator_config").value)
        discriminatornode = DiscriminatorNode(discriminators=qda)

        iq_data = [
            [
                [[0.8, -1.0], [0.1, 0.5], [-0.3, 0.4]],
                [[-0.2, 0.4], [0.2, -1.0], [-0.5, 0.3]],
            ],
            [
                [[0, -1.0], [0.1, -0.5], [0.9, 0]],
                [[-0.8, -0.5], [-0.1, 0.5], [0.2, 1.5]],
            ],
        ]

        self.create_experiment_data(np.array(iq_data) * 1e16, single_shot=True)
        fake_data = np.asarray([datum["memory"] for datum in self.iq_experiment.data()])
        classified = discriminatornode(fake_data)
        expected = [["110", "101"], ["000", "111"]]

        self.assertListEqual(classified.tolist(), expected)
