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
Tests for RB sampling utils.
"""

from test.base import QiskitExperimentsTestCase
from ddt import ddt, data

from qiskit.circuit.library import XGate, CXGate, CCXGate
from qiskit.exceptions import QiskitError

from qiskit_experiments.library.randomized_benchmarking.sampling_utils import (
    SingleQubitSampler,
    EdgeGrabSampler,
    GenericClifford,
)
from qiskit_experiments.library.randomized_benchmarking.clifford_utils import CliffordUtils


@ddt
class TestSamplingUtils(QiskitExperimentsTestCase):
    """Tests for the Sampler classes."""

    seed = 1

    def test_gate_distribution(self):
        """Test the gate distribution is calculated correctly."""
        sampler = SingleQubitSampler(seed=self.seed)
        sampler.gate_distribution = [(0.8, GenericClifford(1)), (0.2, XGate())]
        dist = sampler._probs_by_gate_size(sampler.gate_distribution)
        self.assertEqual(len(dist[1][0]), 25)
        for i in dist[1][0]:
            self.assertTrue(i == XGate() or 0 <= i < CliffordUtils.NUM_CLIFFORD_1_QUBIT)

    def test_1q_custom_gate(self):
        """Test that the single qubit sampler works with custom gates."""
        sampler = SingleQubitSampler(seed=self.seed)
        sampler.gate_distribution = [(1, XGate())]
        layer = list(sampler((0,), 3))
        self.assertEqual(
            layer,
            [(((0,), XGate()),), (((0,), XGate()),), (((0,), XGate()),)],
        )

    def test_1q_cliffords(self):
        """Test that the single qubit sampler can generate clifford layers."""
        sampler = SingleQubitSampler(seed=self.seed)
        sampler.gate_distribution = [(1, GenericClifford(1))]
        layer = sampler((0,), 3)
        for i in layer:
            self.assertTrue(i[0][1] < CliffordUtils.NUM_CLIFFORD_1_QUBIT and i[0][1] >= 0)

    def test_edgegrab(self):
        """Test that the edge grab sampler behaves as expected."""
        sampler = EdgeGrabSampler(seed=self.seed)
        sampler.gate_distribution = [(0.5, GenericClifford(1)), (0.5, CXGate())]
        layer = sampler((0,), 3)
        for i in layer:
            self.assertTrue(
                0 <= i[0].op < CliffordUtils.NUM_CLIFFORD_1_QUBIT or i[0].op == CXGate()
            )

    def test_edgegrab_all_2q(self):
        """Test that the edge grab sampler behaves as expected when two qubit density is
        1."""
        sampler = EdgeGrabSampler(seed=self.seed)
        sampler.gate_distribution = [(0, GenericClifford(1)), (1, CXGate())]
        sampler.coupling_map = [[k, k + 1] for k in range(9)]
        layer = sampler(range(10), 3)
        for i in layer:
            self.assertTrue(i[0].op == CXGate())

    def test_invalid_1q_distribution(self):
        """Test that the single qubit sampler rejects incorrect
        distributions."""
        sampler = SingleQubitSampler(seed=self.seed)
        with self.assertRaises(QiskitError):
            sampler.gate_distribution = [(0.8, GenericClifford(1)), (0.2, CXGate())]

    @data(
        [(1, GenericClifford(2))],
        [(0.5, GenericClifford(2)), (0.5, GenericClifford(3))],
        [(0.2, GenericClifford(2)), (0.3, XGate()), (0.5, CCXGate())],
    )
    def test_invalid_edgegrab_distribution(self, distribution):
        """Test that the edge grab sampler rejects incorrect
        distributions."""
        sampler = EdgeGrabSampler(seed=self.seed)
        with self.assertRaises(QiskitError):
            sampler.gate_distribution = distribution
