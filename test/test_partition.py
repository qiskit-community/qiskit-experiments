"""
Test partition_qubits and partition_edges
"""
from test.base import QiskitExperimentsTestCase

from qiskit.providers.fake_provider import FakeGuadalupe

from qiskit_experiments.whole_backend import (
    partition_qubits,
    partition_edges,
    verify_qubit_groups,
    verify_edge_groups,
)


class TestPartition(QiskitExperimentsTestCase):
    """
    Test partition_qubits and partition_edges
    """

    def test_partition_qubits(self):
        """
        Verify correctness of `partition_qubits`
        (see details in the documentation of verify_qubit_groups)
        """

        distance = 2
        backend = FakeGuadalupe()
        qubit_groups = partition_qubits(backend, distance)
        verify_qubit_groups(backend, qubit_groups, distance)

    def test_partition_edges(self):
        """
        Verify correctness of `partition_edges`
        (see details in the documentation of verify_edge_groups)
        """

        distance = 3
        backend = FakeGuadalupe()
        edge_groups = partition_edges(backend, distance)
        verify_edge_groups(backend, edge_groups, distance)
