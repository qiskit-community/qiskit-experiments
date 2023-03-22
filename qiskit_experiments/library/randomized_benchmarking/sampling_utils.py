# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Utilities for sampling layers in randomized benchmarking experiments
"""

import warnings
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple, TypeVar

from numpy.random import Generator, default_rng, BitGenerator, SeedSequence

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.circuit.library import CXGate
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import random_pauli, Operator
from .clifford_utils import CliffordUtils

GateType = TypeVar("Gate", Instruction, Operator, QuantumCircuit, BaseOperator)


class RBSampler(ABC):
    """Sampling distribution for randomized benchmarking experiments.
    Subclasses must implement the ``__call__()`` method."""

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, qubits, seed=None, **params) -> List[Tuple[Tuple[int, ...], GateType]]:
        """Samplers should define this method such that it returns sampled layers
        given the input parameters. Each layer is represented by a list of size-2 tuples
        where the first element is a tuple of qubit indices, and the second
        element represents the gate that should be applied to the indices.

        Args:
            qubits: A sequence of qubits to generate layers for.
            seed: Seed for random generation.

        """
        return None


class SingleQubitSampler(RBSampler):
    """A sampler that samples layers of random single-qubit gates from a specified gate set."""

    # pylint: disable=arguments-differ
    def __call__(
        self,
        qubits,
        length,
        gate_set: Optional[Union[str, List]] = "clifford",
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
    ) -> List[Tuple[Tuple[int, ...], GateType]]:
        """Samples random single-qubit gates from a specified gate set.

        Args:
            qubits: A sequence of qubits to generate layers for.
            length: The length of the sequence to output.
            gate_set: The one qubit gate set to sample from. Can be either a
                list of gates, "clifford", or "pauli". Default is "clifford".
            seed: Seed for random generation.

        Returns:
            A ``length``-long list of :class:`qiskit.circuit.QuantumCircuit` layers over
            ``num_qubits`` qubits. Each layer is represented by a list of tuples which
            are in the format ((one or more qubit indices), gate). Single-qubit
            Cliffords are represented by integers for speed. For two qubits, length
            3, and the default Clifford gate set, an example output would be

                .. parsed-literal::

                    [(((0,), 2), ((1,), 2)), (((0,), 11), ((1,), 4)), (((0,), 17), ((1,), 17))]
        """
        rng = default_rng(seed=seed)
        num_qubits = len(qubits)
        if gate_set == "clifford":
            layers = []
            for i in rng.integers(CliffordUtils.NUM_CLIFFORD_1_QUBIT, size=(length, num_qubits)):
                layers.append(tuple(zip(((j,) for j in qubits), i)))
        elif gate_set == "pauli":
            layers = []
            for i in range(length):
                layers.append(
                    tuple(zip(((j,) for j in qubits), random_pauli(num_qubits, seed=rng)))
                )
        else:
            layers = []
            for i in rng.integers(len(gate_set), size=(length, num_qubits)):
                layers.append(tuple(zip(((j,) for j in qubits), gate_set[i])))
        return layers


class EdgeGrabSampler(RBSampler):
    r"""A sampler that uses the edge grab algorithm for sampling one- and two-qubit gate
     layers.

    # section: overview

        The edge grab sampler, given a list of :math:`w` qubits, their connectivity
        graph, and the desired two-qubit gate density :math:`\xi_s`, outputs a layer
        as follows:

            1. Begin with the empty set :math:`E` and :math:`E_r`, the set of all edges
               in the connectivity graph. Select an edge from :math:`E_r` at random and
               add it to :math:`E`, removing all edges that share a qubit with the edge
               from :math:`E_r`.
            2. Select edges from :math:`E` with the probability :math:`w\xi/2|E|`. These
               edges will have two-qubit gates in the output layer.

        This produces a layer with an expected two-qubit gate density :math:`\xi`. In
        the default mirror RB configuration where these layers are dressed with
        single-qubit Pauli layers, this means the overall two-qubit gate density will be
        :math:`\xi_s/2=\xi`. The overall density will converge to :math:`\xi` as the
        circuit size increases.

    # section: reference
        .. ref_arxiv:: 1 2008.11294

    """

    # pylint: disable=arguments-differ
    def __call__(
        self,
        qubits: int,
        two_qubit_gate_density: float,
        coupling_map: List[List[int]],
        length: int,
        one_qubit_gate_set: Optional[Union[str, List]] = "clifford",
        two_qubit_gate_set: Optional[Union[str, List]] = "cx",
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
    ) -> List[Tuple[Tuple[int, ...], GateType]]:
        """Sample layers using the edge grab algorithm.

        Args:
            qubits: A sequence of qubits to generate layers for.
            one_qubit_gate_set: The one qubit gate set to sample from. Can be either a
                list of gates or "clifford".
            two_qubit_gate_set: The two qubit gate set to sample from. Can be either a
                list of gates or one of "cx", "cy", "cz", or "csx".
            two_qubit_gate_density: the expected fraction of two-qubit gates in the
                sampled layer.
            coupling_map: List of pairs of connected edges between qubits.
            length: The length of the sequence to output.
            seed: Seed for random generation.

        Raises:
            Warning: If the coupling map has no connectivity or
                ``two_qubit_gate_density`` is too high.
            TypeError: If invalid gate set(s) are specified.

        Returns:
            A ``length``-long list of :class:`qiskit.circuit.QuantumCircuit` layers over
            ``num_qubits`` qubits. Each layer is represented by a list of tuples which
            are in the format ((one or more qubit indices), gate). Single-qubit
            Cliffords are represented by integers for speed. Here's an example with the
            default choice of Cliffords for the single-qubit gates and CXs for the
            two-qubit gates:

            .. parsed-literal::

                (((1, 2), 0), ((0,), 12), ((3,), 20))

            This represents a layer where the 12th Clifford is performed on qubit 0,
            a CX is performed with control qubit 1 and target qubit 2, and the 20th
            Clifford is performed on qubit 3.

        """
        rng = default_rng(seed=seed)
        num_qubits = len(qubits)

        if not isinstance(one_qubit_gate_set, list) and one_qubit_gate_set.casefold() != "clifford":
            raise TypeError("one_qubit_gate_set must be a list of gates or 'clifford'.")

        if num_qubits == 1:
            if one_qubit_gate_set.casefold() == "clifford":
                return (
                    (((qubits[0],), i),)
                    for i in rng.integers(CliffordUtils.NUM_CLIFFORD_1_QUBIT, size=length)
                )
            else:
                return list(enumerate((0, i) for i in rng.choice(one_qubit_gate_set, size=length)))

        if not isinstance(two_qubit_gate_set, list) and two_qubit_gate_set.casefold() != "cx":
            raise TypeError("two_qubit_gate_set must be a list of gates or 'cx'.")

        layer_list = []
        for _ in list(range(length)):
            all_edges = coupling_map[:]  # make copy of coupling map from which we pop edges
            selected_edges = []
            while all_edges:
                rand_edge = all_edges.pop(rng.integers(len(all_edges)))
                selected_edges.append(
                    rand_edge
                )  # move random edge from all_edges to selected_edges
                old_all_edges = all_edges[:]
                all_edges = []
                # only keep edges in all_edges that do not share a vertex with rand_edge
                for edge in old_all_edges:
                    if rand_edge[0] not in edge and rand_edge[1] not in edge:
                        all_edges.append(edge)

            two_qubit_prob = 0
            try:
                # need to divide by 2 since each two-qubit gate spans two lattice sites
                two_qubit_prob = num_qubits * two_qubit_gate_density / 2 / len(selected_edges)
            except ZeroDivisionError:
                warnings.warn("Device has no connectivity. All gates will be single-qubit.")
            if two_qubit_prob > 1:
                warnings.warn(
                    "Mean number of two-qubit gates is higher than the number of selected edges. "
                    + "Actual density of two-qubit gates will likely be lower than input density."
                )

            put_1_qubit_gates = set(qubits)
            # put_1_qubit_gates is a list of qubits that aren't assigned to a 2-qubit gate
            # 1-qubit gates will be assigned to these edges
            layer = []
            for edge in selected_edges:
                if rng.random() < two_qubit_prob:
                    # with probability two_qubit_prob, place a two-qubit gate from the
                    # gate set on edge in selected_edges
                    if two_qubit_gate_set == "cx":
                        layer.append((tuple(edge), CXGate()))
                    else:
                        layer.append(edge, rng.choice(two_qubit_gate_set))
                    # remove these qubits from put_1_qubit_gates
                    put_1_qubit_gates.remove(edge[0])
                    put_1_qubit_gates.remove(edge[1])
            for q in put_1_qubit_gates:
                layer.append(((q,), rng.integers(CliffordUtils.NUM_CLIFFORD_1_QUBIT)))
            layer_list.append(tuple(layer))
        return layer_list
