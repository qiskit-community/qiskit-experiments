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
Test TiledExperiment
"""

from test.base import QiskitExperimentsTestCase

from qiskit_experiments.library import StandardRB
from qiskit_experiments.framework.composite import TiledExperiment


class TestTiledExperiment(QiskitExperimentsTestCase):
    """
    Test TiledExperiment
    """

    def test_tiled_experiment_qubits(self):
        """
        Verify that the constructed experiment covers the specified qubits
        """

        rb_exp = StandardRB([0], [10, 20, 30])
        groups = [[[0]], [[1], [2], [3]]]
        exp = TiledExperiment(rb_exp, groups)

        qubit_groups = []
        for parexp in exp.component_experiment():
            local_group = []
            for rbexp in parexp.component_experiment():
                qubits = list(rbexp.physical_qubits)
                local_group.append(qubits)

            qubit_groups.append(local_group)

        self.assertEqual(qubit_groups, groups)

    def test_tiled_experiment_edges(self):
        """
        Verify that the constructed experiment covers the specified edges
        """

        rb_exp = StandardRB([0, 1], [10, 20, 30])
        groups = [[[0, 1]], [[1, 2], [3, 0]]]
        exp = TiledExperiment(rb_exp, groups)

        edge_groups = []
        for parexp in exp.component_experiment():
            local_group = []
            for rbexp in parexp.component_experiment():
                qubits = list(rbexp.physical_qubits)
                local_group.append(qubits)

            edge_groups.append(local_group)

        self.assertEqual(edge_groups, groups)
