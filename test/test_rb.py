# -*- coding: utf-8 -*-

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

import numpy as np
import qiskit_experiments as qe
from qiskit.quantum_info import Clifford
from qiskit.test import QiskitTestCase
from numpy.random import Generator, default_rng
from typing import Union, Iterable, Optional
from qiskit.test.mock import FakeParis
import unittest
from qiskit.exceptions import QiskitError


@ddt
class TestRB(QiskitTestCase):

    @staticmethod
    def RB_parameters_2_qubit():
        exp_data = {'qubits': [0, 1], 'lengths': [1, 3, 5, 7, 9], 'num_samples': 1, 'seed': 100}
        rb = qe.randomized_benchmarking
        RB_Test = rb.RBExperiment(exp_data["qubits"], exp_data["lengths"], num_samples=exp_data["num_samples"],
                                  seed=exp_data["seed"])
        return exp_data, RB_Test

    def is_identity(self, circuits):
        """Standard randomized benchmarking test - Identity check (assuming all the operator are spanned by clifford group)
        Args:
            quantum_Circuits: list of the circuits which we want to check
        """

        identity = True
        for qc in circuits:
            num_qubits = qc.num_qubits
            qc.remove_final_measurements()
            # Checking if the matrix representation is the identity matrix
            self.assertEqual(np.allclose(Clifford(qc).to_matrix(), np.identity(2 ** num_qubits)),
                             'Clifford sequence doesn\'t result in the identity matrix.')
            # identity = identity and array_equal(a1, a2, equal_nan=False)
        # self.assertEqual(identity, True,'Clifford sequence doesn\'t result in the identity matrix.')

    def validate_metadata(self, circuits, exp_data: dict):
        """

        Args:
            circuits:
            exp_data: 

        Returns:

        """
        for ind, qc in enumerate(circuits):
            self.assertEqual(qc.metadata['xdata'], self._lengths[ind],
                             'The length of the experiment doen\'t match to the one provided.')
            self.assertEqual(qc.metadata['qubits'], tuple(self._qubits[ind]),
                             'The qubits indices doesn\'t match the ran qubit indices.')

    def test_RB_circuits(self):
        exp_2_qubit_data_dict, exp_2_qubit_exp = self.RB_parameters_2_qubit()
        exp_2_qubit_circ = exp_2_qubit_exp.circuits()
        self.is_identity(exp_2_qubit_circ)
        self.validate_metadata(exp_2_qubit_circ, exp_2_qubit_data_dict)

