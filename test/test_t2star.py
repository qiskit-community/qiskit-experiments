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
Test Ramsey experiment
"""

import unittest
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.base_analysis import BaseAnalysis

from qiskit_experiments.characterization import RamseyExperiment

class TestRamsey(unittest.TestCase):
    """
    Test measurement of T1
    """

    def test_Ramsey_end2end(self):
        """
        Test Ramsey experiment using a simulator.
        """
exp = RamseyExperiment(qubit=0, delays=list(range(1, 10, 1)), unit='dt', nosc=20)
circs = exp.circuits()
for c in circs:
    print(c.name)
    print(c)
