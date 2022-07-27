"""
Functions to partition qubits and edges into groups, such that qubits/edges
in a group are distance from each other
"""

import retworkx as rx


def distance_graph(graph, distance):
    """
    The vertices of the distance graph are the same as
    in the original graph. Two vertices are connected by an edge if the distance between them
    in the original graph is smaller than `distance`.
    """
    distance_matrix = rx.distance_matrix(graph)
    dist_graph = rx.PyGraph()
    indexes = graph.node_indices()

    for graph_vertex1 in indexes:
        dist_graph.add_node(graph[graph_vertex1])
        for graph_vertex2 in range(graph_vertex1):
            if distance_matrix[graph_vertex1, graph_vertex2] < distance:
                dist_graph.add_edge(graph_vertex1, graph_vertex2, None)

    return dist_graph


def partition_qubits(backend, distance):
    """
    Partitions the qubits into groups, such that in each group the
    minimum distance (number of edges in the shortest path) between qubits is at least
    `distance`.

    Returns a list of list of integers, i.e., a list of groups of qubits.
    """

    coupling = backend.configuration().coupling_map

    # Construct the coupling graph using retworkx
    graph = rx.PyGraph()
    graph.add_nodes_from(list(range(backend.configuration().num_qubits)))
    for coupling_edge in coupling:
        graph.add_edge(coupling_edge[0], coupling_edge[1], None)

    # A very naive algorithm, which it was chosen (at least for now) because it's easy
    # to implement. I didn't check if it has any guarantees about performance, number of qubit
    # groups (if we want to minimize it - this translates to minimizing the number of circuits),
    # or equitability (if we want the groups - namely the circuits - to be more-or-less of
    # equal size). The literature is filled with algorithms for vertex colorings,
    # including for the case of the distance>2 constraint. In particular,
    # we don't exploit information that we have about the structure of the coupling map, like
    # the fact that the degree of the coupling graph is upper-bounded.

    # Construct the distance graph: an edge between two qubits
    # if the distance between them is smaller than `distance`
    dist_graph = distance_graph(graph, distance)

    # Color the distance graph: qubits that are adjacent in the distance graph
    # will be assigned different colors
    colors = rx.graph_greedy_color(dist_graph)

    # Partition the qubits according to their colors
    return [
        [[graph[qubit]] for qubit in graph.node_indices() if colors[qubit] == c]
        for c in set(colors.values())
    ]


def partition_edges(backend, distance):
    """
    Partitions the edges in the coupling map into groups, such that in each group the
    minimum distance (one plus number of edges in the shortest path) between edges is at least
    `distance`.

    Returns a list of list of pairs of integers, i.e., a list of groups of edges.
    """

    coupling = backend.configuration().coupling_map

    # Algorithm:
    # - We name by "coupling graph" the graph whose vertices are the qubits and edges
    #   are the edges that are in the coupling map. In the code here we don't construct this
    #   graph.
    # - We construct the line graph of the coupling graph. In the line graph,
    #   every vertex represents an edge of the coupling graph. Vertices in the line graph
    #   are connected by an edge if the respective edges in the coupling graph share at least
    #   one vertex.
    # - We construct the distance graph. The vertices of the distance graph are the same as
    #   in the line graph. Two vertices are connected by an edge if the distance between them
    #   in the line graph is smaller than `distance`.
    # - We color the vertices of the distance graph (coloring a graph means that adjacent
    #   vertices are assigned different colors). This induces a coloring to the edges of the
    #   coupling graph that satisfies the distance constraint.
    #
    # This is a very naive algorithm, and it was chosen (at least for now) because it's easy
    # to implement. I didn't check if it has any guarantees about performance, number of edge
    # groups (if we want to minimize it - this translates to minimizing the number of circuits),
    # or equitability (if we want the groups - namely the circuits - to be more-or-less of
    # equal size). The literature is filled with algorithms for vertex and edge colorings,
    # including for the case of the distance>2 constraint. In particular,
    # we don't exploit information that we have about the structure of the coupling map, like
    # the fact that the degrees of the coupling and line graphs are upper-bounded.

    # Construct the line graph
    line_graph = rx.PyGraph()
    # By "coupling edge" we mean "edge of the coupling graph"
    for coupling_edge in coupling:
        # By "line vertex" we mean "vertex of the line graph"
        # Each vertex in the line graph is originated from an edge in the coupling map,
        # we keep the original coupling edge in the label of the line vertex
        line_vertex = line_graph.add_node(coupling_edge)
        for existing_line_vertex in range(line_vertex):
            existing_coupling_edge = line_graph[existing_line_vertex]
            if set(coupling_edge).intersection(set(existing_coupling_edge)):
                line_graph.add_edge(line_vertex, existing_line_vertex, None)

    # Construct the distance graph
    dist_graph = distance_graph(line_graph, distance)

    # Color the distance graph
    colors = rx.graph_greedy_color(dist_graph)

    return [
        [
            line_graph[line_vertex]
            for line_vertex in line_graph.node_indices()
            if colors[line_vertex] == c
        ]
        for c in set(colors.values())
    ]


def verify_qubit_groups(backend, qubit_groups, distance):
    """
    We verify:
    - Every qubit is contained in exactly one group.
    - The distance between qubits belonging to the same group is at least `distance`.
    """

    nqubits = backend.configuration().n_qubits
    coupling = backend.configuration().coupling_map

    # Build the coupling graph:
    # - Vertices are the qubits.
    # - Edges are the edges of the coupling map.
    coupling_graph = rx.PyGraph()
    coupling_graph.add_nodes_from(list(range(nqubits)))
    for edge in coupling:
        coupling_graph.add_edge(edge[0], edge[1], None)

    # Compute the distances between vertices
    distance_matrix = rx.distance_matrix(coupling_graph)

    found_qubits = []
    for group in qubit_groups:
        for i, qubit1 in enumerate(group):
            # Verify that no qubit repeats twice (either in the same group
            # or in different groups)
            assert 0 <= qubit1[0] < nqubits
            assert qubit1[0] not in found_qubits
            found_qubits.append(qubit1[0])

            # Verify that the minimum distance between qubits
            # that belong to the same group is at least `distance`
            for j in range(i):
                qubit2 = group[j]
                assert distance_matrix[qubit1[0], qubit2[0]] >= distance

    # Verify that every qubit belongs to some group.
    assert len(found_qubits) == nqubits


def verify_edge_groups(backend, edge_groups, distance):
    """
    We verify:
    - Every edge in the coupling map is contained in exactly one group.
    - Every edge in any of the groups is in the coupling map.
    - The distance (one plus number of edges in the coupling map) between edges
      belonging to the same group is at least `distance`.

    In order not to repeat, in the test, bugs that the test actually aims to detect,
    we intentionally perform computations differently from `partition_edges`.
    Instead of working with the line graph, we work directly on the coupling graph.
    To check the distance between edges in the coupling graph, we check the distance between
    their end vertices.
    """

    nqubits = backend.configuration().n_qubits
    coupling = backend.configuration().coupling_map

    # Build the coupling graph:
    # - Vertices are the qubits.
    # - Edges are the edges of the coupling map.
    coupling_graph = rx.PyGraph()
    coupling_graph.add_nodes_from(list(range(nqubits)))
    for edge in coupling:
        coupling_graph.add_edge(edge[0], edge[1], None)

    # Compute the distances between vertices
    distance_matrix = rx.distance_matrix(coupling_graph)

    found_edges = []
    for group in edge_groups:
        for i, edge1 in enumerate(group):
            # Verify that no edge repeats twice (either in the same group
            # or in different groups)
            assert edge1 not in found_edges
            found_edges.append(edge1)

            # Verify that all edges in the groups belong to the coupling map
            assert edge1 in coupling

            # Verify that the minimum distance between end nodes of two edges,
            # that belong to the same group, is at least `distance`-1
            for j in range(i):
                edge2 = group[j]
                for vertex1 in edge1:
                    for vertex2 in edge2:
                        assert distance_matrix[vertex1, vertex2] >= distance - 1

    # Verify that every edge in the coupling map belongs to some group.
    assert len(found_edges) == len(coupling)
