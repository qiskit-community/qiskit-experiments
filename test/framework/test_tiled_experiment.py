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

from qiskit_experiments.library import StandardRB, T1
from qiskit_experiments.framework.composite import TiledExperiment
from qiskit_experiments.test.noisy_delay_aer_simulator import NoisyDelayAerBackend


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

    def test_tiled_run(self):
        """
        Test full run of TiledExperiment using T1 as the template.

        This test validates that TiledExperiment correctly remaps circuits
        to different qubits and produces accurate T1 measurements.
        """
        t1 = [25, 20, 15, 18]
        t2 = [value / 2 for value in t1]
        delays = list(range(1, 40, 3))

        backend = NoisyDelayAerBackend(t1, t2)

        # Create a template T1 experiment for qubit 0
        template_exp = T1(physical_qubits=[0], delays=delays)

        # Define groups for tiling - batch two groups of parallel experiments
        groups = [
            [[0], [2]],  # First batch: qubits 0 and 2 in parallel
            [[1], [3]],  # Second batch: qubits 1 and 3 in parallel
        ]

        # Create tiled experiment - TiledExperiment uses flatten_results=True by default
        # but we can check the structure by looking at component experiments
        tiled_exp = TiledExperiment(template_exp, groups)

        # Verify the experiment structure before running
        self.assertEqual(len(tiled_exp.component_experiment()), 2)  # 2 batches
        self.assertEqual(
            len(tiled_exp.component_experiment(0).component_experiment()), 2
        )  # 2 parallel in first
        self.assertEqual(
            len(tiled_exp.component_experiment(1).component_experiment()), 2
        )  # 2 parallel in second

        # Run the experiment
        res = tiled_exp.run(backend=backend, shots=10000, seed_simulator=1)
        self.assertExperimentDone(res)

        # With flatten_results=True (default), all results are flattened
        # We should have analysis results for all 4 qubits
        t1_results = res.analysis_results("T1", dataframe=True)
        self.assertEqual(len(t1_results), 4)

        # Verify T1 values - results should be in order: qubits 0, 2, 1, 3
        expected_qubit_order = [0, 2, 1, 3]
        for idx, qb in enumerate(expected_qubit_order):
            sub_res = t1_results.iloc[idx]
            self.assertEqual(sub_res.quality, "good")
            self.assertAlmostEqual(sub_res.value.n, t1[qb], delta=3)
