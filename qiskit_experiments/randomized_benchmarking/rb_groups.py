# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Methods for handling groups (Clifford, CNOT Dihedral etc.)
in randomized benchmarking"""

import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.symplectic import Clifford
from qiskit.quantum_info.operators.dihedral import CNOTDihedral
from qiskit.quantum_info.random import random_clifford
from qiskit.quantum_info.random import random_cnotdihedral

class RBgroup():
    """Class that handles the group operations needed for RB."""

    def __init__(self, group_gates, num_qubits=2):
        """Initialization from num_qubits and group_gates"""
        self._num_qubits = num_qubits
        self._group_gates = group_gates
        self._rb_circ_type = 'rb'
        self._group_gates_type = 0

        if group_gates is None or group_gates in ('0',
                                                  'Clifford',
                                                  'clifford'):
            self._rb_group = Clifford
        elif group_gates in ('1', 'Non-Clifford',
                             'NonClifford',
                             'CNOTDihedral',
                             'CNOT-Dihedral'):
            self._rb_group = CNOTDihedral
            self._rb_circ_type += '_cnotdihedral'
            self._group_gates_type = 1
        else:
            raise QiskitError("Unknown group or set of gates.")

    def num_qubits(self):
        """Return the number of qubits."""
        return self._num_qubits

    def group_gates_type(self):
        """Return an integer of the group type"""
        return self._group_gates_type

    def rb_circ_type(self):
        """Return a string of type for the circuit name"""
        return self._rb_circ_type

    def iden(self, num_qubits):
        """Initialize an identity group element"""
        self._num_qubits = num_qubits
        if self._group_gates_type:
            return CNOTDihedral(num_qubits)
        else:
            return Clifford(np.eye(2 * num_qubits))

    def random(self, num_qubits, rand_seed=None):
        """Generate a random group element"""
        self._num_qubits = num_qubits
        if self._group_gates_type:
            return random_cnotdihedral(num_qubits, seed=rand_seed)
        else:
            return random_clifford(num_qubits, seed=rand_seed)

    @staticmethod
    def compose(elem, other):
        """Compose two group elements: orig and other"""
        return elem.compose(other)

    @staticmethod
    def inverse(elem):
        """Computes the inverse QuantumCircuit"""
        # decompose the group element into a QuantumCircuit
        circ = elem.to_circuit()
        # invert the QuantumCircuit
        return circ.inverse()

    @staticmethod
    def to_circuit(elem):
        """Returns the corresponding QuantumCircuit"""
        return elem.to_circuit()
