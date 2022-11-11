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
from test.base import QiskitExperimentsTestCase
from ddt import ddt, data

from qiskit import pulse
from qiskit_experiments.library import MultiStateDiscrimination
from qiskit_experiments.test.pulse_backend import SingleTransmonTestBackend


@ddt
class TestMultiStateDiscrimination(QiskitExperimentsTestCase):
    """Tests of the multi state discrimination experiment."""

    def setUp(self):
        """Setup test variables."""
        super().setUp()

        self.backend = SingleTransmonTestBackend(noise=False)

        # Build x12 schedule
        self.qubit = 0

        anharm = self.backend.anharmonicity

        d0 = pulse.DriveChannel(self.qubit)

        sch_map = self.backend.defaults().instruction_schedule_map
        pulse_x = sch_map.get('x', (self.qubit,)).instructions[0][1].pulse
        amp_x = pulse_x.amp
        dur_x = pulse_x.duration
        sigma_x = pulse_x.sigma
        beta_x = pulse_x.beta
        with pulse.build(name='x12') as x12:
            pulse.shift_frequency(anharm, d0)
            pulse.play(pulse.Gaussian(dur_x, amp_x * self.backend.rabi_rate_12, sigma_x, beta_x), d0)
            pulse.shift_frequency(-anharm, d0)

        self.schedules = {'x12': x12}

    @data(2, 3)
    def test_circuit_generation(self, n_states):
        """Test the experiment circuit generation"""
        exp = MultiStateDiscrimination(self.qubit, n_states=n_states, backend=self.backend,
                                       schedules=self.schedules)
        self.assertEqual(len(exp.circuits()), n_states)

        # check the metadata
        self.assertEqual(exp.circuits()[-1].metadata['label'], n_states - 1)

    @data(2, 3)
    def test_discrimination_analysis(self, n_states):
        """Test the discrimination analysis"""
        exp = MultiStateDiscrimination(self.qubit, n_states=n_states, backend=self.backend,
                                       schedules=self.schedules)

        exp_data = exp.run()

        fidelity = exp_data.analysis_results('fidelity').value

        self.assertGreaterEqual(fidelity, 0.96)

        # check that the discriminator differentiates n different states
        discrim_lbls = exp_data.analysis_results('discriminator_config').value['attributes']['classes_']
        self.assertEqual(len(discrim_lbls), n_states)
