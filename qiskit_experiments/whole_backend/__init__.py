"""
Functions to facilitate experiments on entire backends
"""

from qiskit_experiments.whole_backend.whole_backend import build_whole_backend_experiment
from qiskit_experiments.whole_backend.partition import (
    partition_qubits,
    partition_edges,
    verify_qubit_groups,
    verify_edge_groups,
)
