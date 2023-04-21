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
"""Mixins for randomized benchmarking module tests"""
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator


class RBTestMixin:
    """Mixin for RB tests."""

    def assertAllIdentity(self, circuits):
        """Test if all experiment circuits are identity."""
        for circ in circuits:
            num_qubits = circ.num_qubits
            qc_iden = QuantumCircuit(num_qubits)
            circ.remove_final_measurements()
            self.assertTrue(Operator(circ).equiv(Operator(qc_iden)))
