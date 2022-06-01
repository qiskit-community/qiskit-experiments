# This code is part of Qiskit.
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for randomized benchmarking experiments."""

from test.base import QiskitExperimentsTestCase
import numpy as np
from ddt import ddt, data, unpack
import time

from qiskit import IBMQ

from qiskit_experiments.library import randomized_benchmarking as rb

class RBTestCase(QiskitExperimentsTestCase):
    """Base test case for randomized benchmarking defining a common noise model."""

    def __init__(self):
        """Set up the tests."""

        # basis gates
        self.basis_gates = ["sx", "rz"]

        # Need level1 for consecutive gate cancellation for reference EPC value calculation
        self.transpiler_options = {
            "basis_gates": self.basis_gates,
            "optimization_level": 1,
        }
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub="ibm-q-internal", group="deployed", project="default")
        self.backend = provider.backend.ibmq_armonk

@ddt
class TestStandardRB(RBTestCase):
    """Test for standard RB."""

    def test_single_qubit(self, lengths):
        """Test single qubit RB."""
        exp = rb.StandardRB(
            qubits=(0,),
            lengths = lengths,
            seed=123,
            backend=self.backend,
        )
        exp.analysis.set_options(gate_error_ratio=None)
        exp.set_transpile_options(**self.transpiler_options)
        #assertAllIdentity(exp.circuits())
        expdata = exp.run()
        epc = expdata.analysis_results("EPC")
        epc_expected = 1 - (1 - 1 / 2 * self.p1q) ** 1.0
        self.assertAlmostEqual(epc.value.n, epc_expected, delta=0.15 * epc_expected)

    def time_rb_single_qubit(self):
        maxcliff = 4000
        nCliffs = np.logspace(np.log10(1.5), np.log10(maxcliff), int(np.log10(maxcliff) * 4), dtype=int)
        lengths_list = [list(range(1, 300, 30)),
                        list(range(1, 1002, 100)),
                        nCliffs]

        iters = 3
        result_times = {}
        for lengths in lengths_list:
            sum_time = 0.0
            for iter in range(iters):
                start = time.time()
                self.test_single_qubit(lengths=lengths)
                end = time.time()
                sum_time += end-start
            sum_time /= iters
            result_times[lengths[-1]] = sum_time
            print("lengths = " + str(lengths))
        print(result_times)



# For testing the methods standalone
test1 = TestStandardRB()
test1.time_rb_single_qubit()

