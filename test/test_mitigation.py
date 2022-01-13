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

"""
A Tester for the Readout Mitigation experiment
"""


import unittest
from test.base import QiskitExperimentsTestCase
import numpy as np
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit_experiments.library import ReadoutMitigationExperiment
from qiskit_experiments.framework import ExperimentData


class TestMitigation(QiskitExperimentsTestCase):
    """Test ReadoutMitigationExperiment"""

    def test_local_analysis(self):
        """Tests local mitigator generation from experimental data"""
        qubits = [0, 1, 2]
        run_data = [
            {
                "counts": {"000": 986, "010": 10, "100": 16, "001": 12},
                "metadata": {"label": "000"},
                "shots": 1024,
            },
            {
                "counts": {"111": 930, "110": 39, "011": 24, "101": 29, "010": 1, "100": 1},
                "metadata": {"label": "111"},
                "shots": 1024,
            },
        ]
        expected_assignment_matrices = [
            np.array([[0.98828125, 0.04003906], [0.01171875, 0.95996094]]),
            np.array([[0.99023438, 0.02929688], [0.00976562, 0.97070312]]),
            np.array([[0.984375, 0.02441406], [0.015625, 0.97558594]]),
        ]
        run_meta = {"physical_qubits": qubits}
        expdata = ExperimentData()
        expdata.add_data(run_data)
        expdata._metadata = run_meta
        exp = ReadoutMitigationExperiment(qubits)
        result = exp.analysis.run(expdata)
        mitigator = result.analysis_results(0).value

        self.assertEqual(len(qubits), mitigator._num_qubits)
        self.assertEqual(qubits, mitigator._qubits)
        self.assertTrue(matrix_equal(expected_assignment_matrices, mitigator._assignment_mats))

    def test_correlated_analysis(self):
        """Tests correlated mitigator generation from experimental data"""
        qubits = [0, 2, 3]
        run_data = [
            {
                "counts": {"000": 989, "010": 12, "100": 7, "001": 15, "101": 1},
                "metadata": {"label": "000"},
                "shots": 1024,
            },
            {
                "counts": {"001": 971, "101": 15, "000": 36, "011": 2},
                "metadata": {"label": "001"},
                "shots": 1024,
            },
            {
                "counts": {"000": 30, "010": 965, "110": 15, "011": 11, "001": 2, "100": 1},
                "metadata": {"label": "010"},
                "shots": 1024,
            },
            {
                "counts": {"011": 955, "111": 15, "010": 26, "001": 27, "110": 1},
                "metadata": {"label": "011"},
                "shots": 1024,
            },
            {
                "counts": {"100": 983, "101": 8, "110": 13, "000": 20},
                "metadata": {"label": "100"},
                "shots": 1024,
            },
            {
                "counts": {"101": 947, "001": 34, "100": 32, "111": 11},
                "metadata": {"label": "101"},
                "shots": 1024,
            },
            {
                "counts": {"100": 26, "110": 965, "010": 21, "111": 11, "000": 1},
                "metadata": {"label": "110"},
                "shots": 1024,
            },
            {
                "counts": {"111": 938, "011": 23, "110": 35, "101": 27, "100": 1},
                "metadata": {"label": "111"},
                "shots": 1024,
            },
        ]

        expected_assignment_matrix = np.array(
            [
                [0.96582031, 0.03515625, 0.02929688, 0.0, 0.01953125, 0.0, 0.00097656, 0.0],
                [0.01464844, 0.94824219, 0.00195312, 0.02636719, 0.0, 0.03320312, 0.0, 0.0],
                [0.01171875, 0.0, 0.94238281, 0.02539062, 0.0, 0.0, 0.02050781, 0.0],
                [0.0, 0.00195312, 0.01074219, 0.93261719, 0.0, 0.0, 0.0, 0.02246094],
                [0.00683594, 0.0, 0.00097656, 0.0, 0.95996094, 0.03125, 0.02539062, 0.00097656],
                [0.00097656, 0.01464844, 0.0, 0.0, 0.0078125, 0.92480469, 0.0, 0.02636719],
                [0.0, 0.0, 0.01464844, 0.00097656, 0.01269531, 0.0, 0.94238281, 0.03417969],
                [0.0, 0.0, 0.0, 0.01464844, 0.0, 0.01074219, 0.01074219, 0.91601562],
            ]
        )
        run_meta = {"physical_qubits": qubits}
        expdata = ExperimentData()
        expdata.add_data(run_data)
        expdata._metadata = run_meta
        exp = ReadoutMitigationExperiment(qubits, method="correlated")
        result = exp.analysis.run(expdata)
        mitigator = result.analysis_results(0).value

        self.assertEqual(len(qubits), mitigator._num_qubits)
        self.assertEqual(qubits, mitigator._qubits)
        self.assertTrue(matrix_equal(expected_assignment_matrix, mitigator.assignment_matrix()))


if __name__ == "__main__":
    unittest.main()
