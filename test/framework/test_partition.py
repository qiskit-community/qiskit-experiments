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
Test functions for graph partitioning
"""

from test.base import QiskitExperimentsTestCase

import rustworkx as rx

from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2, FakeLimaV2

from qiskit_experiments.framework.backend_partition import (
    build_coupling_graph,
    build_line_graph,
    build_distance_graph,
    partition_nodes,
    partition_edges,
    partition_qubits,
    partition_qubit_pairs,
)


class TestPartition(QiskitExperimentsTestCase):
    """
    Test functions for graph partitioning
    """

    def verify_node_groups(self, graph, node_groups, distance, node_subset=None):
        """
        We verify:
        - Every node is contained in exactly one group.
        - The distance between nodes belonging to the same group is at least `distance`.
        """

        if node_subset is None:
            node_subset = graph.nodes()

        map_node_labels_to_dist_indices = {
            node_label: dist_index for dist_index, node_label in enumerate(graph.nodes())
        }

        # Compute the distances between nodes
        distance_matrix = rx.distance_matrix(graph)

        found_nodes = []
        for group in node_groups:
            for i, node1 in enumerate(group):
                # Verify that no node repeats twice (either in the same group
                # or in different groups)
                index1 = map_node_labels_to_dist_indices[node1[0]]
                self.assertTrue(index1 not in found_nodes)
                self.assertTrue(node1[0] in node_subset)
                found_nodes.append(index1)

                # Verify that the minimum distance between nodes
                # that belong to the same group is at least `distance`
                for j in range(i):
                    node2 = group[j]
                    index2 = map_node_labels_to_dist_indices[node2[0]]
                    self.assertTrue(distance_matrix[index1, index2] >= distance)

        # Verify that every node belongs to some group.
        self.assertEqual(len(found_nodes), len(node_subset))

    def verify_edge_groups(self, graph, edge_groups, distance, edge_subset=None, node_subset=None):
        """
        We verify:
        - Every edge in the graph is contained in exactly one group.
        - Every edge in any of the groups is in the graph.
        - The distance between edges belonging to the same group is at least `distance`.
          The distance between edges is defined as the minimum distance between endpoint nodes of
          the edges, plus one.

        In order not to repeat, in the test, bugs that the test actually aims to detect,
        we intentionally perform computations differently from `partition_edges`:
        Instead of working with the line graph, we work directly on the input graph.
        """

        if edge_subset is None:
            edge_subset = graph.edges()

        if node_subset is None:
            node_subset = graph.nodes()

        edges = graph.edges()
        map_node_labels_to_dist_indices = {
            node_label: dist_index for dist_index, node_label in enumerate(graph.nodes())
        }

        # Compute the distances between nodes
        distance_matrix = rx.distance_matrix(graph)

        found_edges = []
        for group in edge_groups:
            for i, edge1 in enumerate(group):
                # Verify that no edge repeats twice (either in the same group
                # or in different groups)
                self.assertTrue(edge1 not in found_edges)
                self.assertTrue(edge1 in edge_subset)
                self.assertTrue(edge1[0] in node_subset and edge1[1] in node_subset)
                found_edges.append(edge1)

                # Verify that all edges in the groups belong to the graph
                self.assertTrue(edge1 in edges)

                # Verify that the minimum distance between end nodes of two edges,
                # that belong to the same group, is at least `distance`-1
                for j in range(i):
                    edge2 = group[j]
                    for node1 in edge1:
                        for node2 in edge2:
                            index1 = map_node_labels_to_dist_indices[node1]
                            index2 = map_node_labels_to_dist_indices[node2]
                            self.assertTrue(distance_matrix[index1, index2] >= distance - 1)

        # Verify that every edge in the graph belongs to some group.
        for edge in edge_subset:
            if edge[0] in node_subset and edge[1] in node_subset:
                self.assertTrue(edge in found_edges)

    def test_build_coupling_graph(self):
        """
        Test the method `build_coupling_graph`
        """
        backend = FakeLimaV2()

        expected_graph1, expected_graph2 = [rx.PyGraph(multigraph=False), rx.PyGraph()]
        for expected_graph in [expected_graph1, expected_graph2]:
            expected_graph.add_nodes_from([0, 1, 2, 3, 4])
            expected_graph.add_edge(0, 1, [1, 0])
            expected_graph.add_edge(1, 2, [2, 1])
            expected_graph.add_edge(1, 3, [3, 1])
            expected_graph.add_edge(3, 4, [4, 3])

        graph1 = build_coupling_graph(backend)
        self.assertTrue(
            rx.is_isomorphic(
                graph1,
                expected_graph1,
                node_matcher=lambda x, y: x == y,
                edge_matcher=lambda x, y: x == y,
            )
        )

        expected_graph2.add_edge(0, 1, [0, 1])
        expected_graph2.add_edge(1, 2, [1, 2])
        expected_graph2.add_edge(1, 3, [1, 3])
        expected_graph2.add_edge(3, 4, [3, 4])

        graph2 = build_coupling_graph(backend, multigraph=True)
        self.assertTrue(
            rx.is_isomorphic(
                graph2,
                expected_graph2,
                node_matcher=lambda x, y: x == y,
                edge_matcher=lambda x, y: x == y,
            )
        )

    def test_build_line_graph(self):
        """
        test the method `build_line_graph`
        """
        backend = FakeLimaV2()
        graph = build_coupling_graph(backend)
        graph.remove_node(2)
        line_graph = build_line_graph(graph)

        expected_line_graph = rx.PyGraph()
        expected_line_graph.add_nodes_from([[1, 0], [3, 1], [4, 3]])
        expected_line_graph.add_edge(0, 1, None)
        expected_line_graph.add_edge(1, 2, None)

        self.assertTrue(
            rx.is_isomorphic(
                line_graph,
                expected_line_graph,
                node_matcher=lambda x, y: x == y,
                edge_matcher=lambda x, y: x == y,
            )
        )

    def test_build_distance_graph(self):
        """
        Test the method `build_distance_graph`
        """
        backend = FakeLimaV2()
        graph = build_coupling_graph(backend)
        graph.remove_node(2)
        dist_graph = build_distance_graph(graph, 3)

        expected_dist_graph = rx.PyGraph()
        expected_dist_graph.add_nodes_from([0, 1, 3, 4])
        expected_dist_graph.add_edge(0, 1, None)
        expected_dist_graph.add_edge(0, 2, None)
        expected_dist_graph.add_edge(1, 2, None)
        expected_dist_graph.add_edge(1, 3, None)
        expected_dist_graph.add_edge(2, 3, None)

        self.assertTrue(
            rx.is_isomorphic(
                dist_graph,
                expected_dist_graph,
                node_matcher=lambda x, y: x == y,
                edge_matcher=lambda x, y: x == y,
            )
        )

        dist_graph_subset = build_distance_graph(graph, 3, [0, 3, 4])

        expected_dist_graph_subset = rx.PyGraph()
        expected_dist_graph_subset.add_nodes_from([0, 3, 4])
        expected_dist_graph_subset.add_edge(0, 1, None)
        expected_dist_graph_subset.add_edge(1, 2, None)

        self.assertTrue(
            rx.is_isomorphic(
                dist_graph_subset,
                expected_dist_graph_subset,
                node_matcher=lambda x, y: x == y,
                edge_matcher=lambda x, y: x == y,
            )
        )

    def test_partition_qubits(self):
        """
        Verify correctness of `partition_qubits`
        (see details in the documentation of verify_node_groups)
        """

        distance = 2
        backend = FakeGuadalupeV2()
        graph = build_coupling_graph(backend)
        qubit_groups = partition_qubits(backend, distance)
        self.verify_node_groups(graph, qubit_groups, distance)

        node_subset = [0, 1, 3, 4, 5, 6, 7, 8, 9]
        qubit_groups_subset = partition_qubits(backend, distance, node_subset)
        self.verify_node_groups(graph, qubit_groups_subset, distance, node_subset)

    def test_partition_qubit_pairs(self):
        """
        Verify correctness of `partition_qubit_pairs`
        (see details in the documentation of verify_edge_groups)
        """
        distance = 2
        backend = FakeGuadalupeV2()
        graph = build_coupling_graph(backend, multigraph=True)
        edge_groups = partition_qubit_pairs(backend, distance, multigraph=True)
        self.verify_edge_groups(graph, edge_groups, distance)

        edge_subset = [[0, 1], [1, 4], [4, 7], [6, 7], [7, 10], [10, 12]]
        edge_groups_edge_subset = partition_qubit_pairs(
            backend, distance, multigraph=True, edge_subset=edge_subset
        )
        self.verify_edge_groups(graph, edge_groups_edge_subset, distance, edge_subset)

        node_subset = [0, 1, 3, 4, 5, 6, 7, 8, 9]
        edge_groups_node_subset = partition_qubit_pairs(backend, distance, True, None, node_subset)
        self.verify_edge_groups(graph, edge_groups_node_subset, distance, None, node_subset)

    def test_partition_nodes(self):
        """
        Verify correctness of `partition_nodes`
        (see details in the documentation of verify_node_groups)
        """
        distance = 3
        backend = FakeGuadalupeV2()
        graph = build_coupling_graph(backend)
        graph.remove_node(3)
        node_groups = partition_nodes(graph, distance)
        self.verify_node_groups(graph, node_groups, distance)

        node_subset = [0, 1, 2, 4, 5, 6, 7, 8, 9]
        node_groups_subset = partition_nodes(graph, distance, node_subset)
        self.verify_node_groups(graph, node_groups_subset, distance, node_subset)

    def test_partition_edges(self):
        """
        Verify correctness of `partition_edges`
        (see details in the documentation of verify_edge_groups)
        """
        distance = 3
        backend = FakeGuadalupeV2()
        graph = build_coupling_graph(backend, multigraph=True)
        graph.remove_node(3)
        edge_groups = partition_edges(graph, distance)
        self.verify_edge_groups(graph, edge_groups, distance)

        edge_subset = [[0, 1], [1, 2], [1, 4], [4, 7], [6, 7], [7, 10]]
        edge_groups_subset = partition_edges(graph, distance, edge_subset=edge_subset)
        self.verify_edge_groups(graph, edge_groups_subset, distance, edge_subset)

        node_subset = [0, 1, 2, 4, 5, 6, 7, 8, 9]
        edge_groups_subset = partition_edges(graph, distance, None, node_subset)
        self.verify_edge_groups(graph, edge_groups_subset, distance, None, node_subset)
