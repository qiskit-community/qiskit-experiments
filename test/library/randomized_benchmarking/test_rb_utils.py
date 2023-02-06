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
A Tester for the RB utils module
"""
from test.base import QiskitExperimentsTestCase
from ddt import ddt

import qiskit_experiments.library.randomized_benchmarking as rb


@ddt
class TestRBUtilities(QiskitExperimentsTestCase):
    """
    A test class for additional functionality provided by the StandardRB
    class.
    """

    seed = 42

    def test_coherence_limit(self):
        """Test coherence_limit."""
        t1 = 100.0
        t2 = 100.0
        gate_2_qubits = 0.5
        gate_1_qubit = 0.1
        twoq_coherence_err = rb.RBUtils.coherence_limit(2, [t1, t1], [t2, t2], gate_2_qubits)

        oneq_coherence_err = rb.RBUtils.coherence_limit(1, [t1], [t2], gate_1_qubit)

        self.assertAlmostEqual(oneq_coherence_err, 0.00049975, 6, "Error: 1Q Coherence Limit")

        self.assertAlmostEqual(twoq_coherence_err, 0.00597, 5, "Error: 2Q Coherence Limit")
