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
import math
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple, Sequence, TypeVar
from collections import defaultdict

from numpy.random import Generator, default_rng, BitGenerator, SeedSequence

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import Operator
from qiskit.exceptions import QiskitError

from .clifford_utils import CliffordUtils, _CLIFF_SINGLE_GATE_MAP_1Q

GateType = TypeVar("GateType", str, int, Instruction, Operator, QuantumCircuit, BaseOperator)


class RBSampler(ABC):
    """Base class for the sampling distribution for randomized benchmarking experiments.
    Subclasses must implement the ``__call__()`` method."""

    def __init__(
        self,
        gate_distribution=None,
        coupling_map: Optional[List[List[int]]] = None,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
    ):
        """
        Args:
            seed: Seed for random generation.
            gate_distribution: The gate distribution for sampling.
        """
        self._gate_distribution = gate_distribution
        self._coupling_map = coupling_map
        self.seed = default_rng(seed)

    @property
    def coupling_map(self):
        """The coupling map of the system to sample over."""
        return self._coupling_map

    @coupling_map.setter
    def coupling_map(self, coupling_map):
        self._coupling_map = coupling_map

    @property
    def seed(self):
        """The seed for random generation."""
        return self._rng

    @seed.setter
    def seed(self, seed):
        self._rng = default_rng(seed)

    @property
    def gate_distribution(self):
        """The gate distribution for sampling. The distribution is a list of tuples
        of the form ``(probability, width of gate, gate)``. Gates can be actual
        operators, circuits, or the special keywords "clifford", "pauli", "idle". An
        example distribution for the edge grab sampler might be

        .. parsed-literal::
            [(0.8, 1, "clifford"), (0.2, 2, CXGate)]

        The probabilities must sum to 1."""
        return self._gate_distribution

    @gate_distribution.setter
    def gate_distribution(self, dist: List[Tuple[float, int, GateType]]):
        """Set the distribution of gates used in the sampler.

        Args:
            dist: A list of tuples with format ``(probability, width of gate, gate)``.
        """
        if not isinstance(dist, List):
            raise TypeError("The gate distribution should be a list.")
        if sum(list(zip(*dist))[0]) != 1:
            raise QiskitError("Gate distribution probabilities must sum to 1.")

        self._gate_distribution = dist

    def _probs_by_gate_size(self):
        """Return a list of gates and their probabilities indexed by the size of the gate. The
        probability distributions are normalized to 1 within each gate size category.
        """
        if not self._gate_distribution:
            raise QiskitError("Gate distribution must be set before sampling.")
        gate_probs = defaultdict(list)

        for gate in self.gate_distribution:
            if gate[2] == "clifford" and gate[1] == 1:
                gateset = list(range(CliffordUtils.NUM_CLIFFORD_1_QUBIT))
                probs = [
                    gate[0] / CliffordUtils.NUM_CLIFFORD_1_QUBIT
                ] * CliffordUtils.NUM_CLIFFORD_1_QUBIT
            elif gate[2] == "pauli" and gate[1] == 1:
                gateset = [
                    _CLIFF_SINGLE_GATE_MAP_1Q[("id", (0,))],
                    _CLIFF_SINGLE_GATE_MAP_1Q[("x", (0,))],
                    _CLIFF_SINGLE_GATE_MAP_1Q[("y", (0,))],
                    _CLIFF_SINGLE_GATE_MAP_1Q[("z", (0,))],
                ]
                probs = [gate[0] / 4] * 4
            elif gate[2] == "clifford" and gate[1] == 2:
                gateset = list(range(CliffordUtils.NUM_CLIFFORD_2_QUBIT))
                probs = [
                    gate[0] / CliffordUtils.NUM_CLIFFORD_2_QUBIT
                ] * CliffordUtils.NUM_CLIFFORD_2_QUBIT
            else:
                gateset = [gate[2]]
                probs = [gate[0]]
            if len(gate_probs[gate[1]]) == 0:
                gate_probs[gate[1]] = [gateset, probs]
            else:
                gate_probs[gate[1]][0].extend(gateset)
                gate_probs[gate[1]][1].extend(probs)
        return gate_probs

    @abstractmethod
    def __call__(self, qubits: Sequence, length: int = 1) -> List[Tuple[Tuple[int, ...], GateType]]:
        """Samplers should define this method such that it returns sampled layers
        given the input parameters. Each layer is represented by a list of size-2 tuples
        where the first element is a tuple of qubit indices, and the second
        element represents the gate that should be applied to the indices.

        Args:
            qubits: A sequence of qubits to generate layers for.
            length: The number of layers to generate. Defaults to 1.
        """
        return None


class SingleQubitSampler(RBSampler):
    """A sampler that samples layers of random single-qubit gates from a specified gate set."""

    def __call__(
        self,
        qubits: Sequence,
        length: int = 1,
    ) -> List[List[Tuple[Tuple[int, ...], GateType]]]:
        """Samples random single-qubit gates from the specified gate set.

        Args:
            qubits: A sequence of qubits to generate layers for.
            length: The length of the sequence to output.
            seed: Seed for random generation.

        Returns:
            A ``length``-long list of :class:`qiskit.circuit.QuantumCircuit` layers over
            ``qubits``. Each layer is represented by a list of tuples which are in the
            format ((one or more qubit indices), gate). Single-qubit Cliffords are
            represented by integers for speed. For two qubits, length 3, and the
            all-Clifford distribution ``[(1, "clifford")]``, an example output would be

            .. parsed-literal::
                [[((0,), 2), ((1,), 2)], [((0,), 11), ((1,), 4)], [((0,), 17), ((1,), 17)]]
        """

        gateset = self._probs_by_gate_size()
        if not math.isclose(sum(gateset[1][1]), 1):
            raise QiskitError(
                "The distribution for SingleQubitSampler should be all single qubit gates."
            )

        samples = self._rng.choice(
            gateset[1][0],
            size=(length, len(qubits)),
            p=gateset[1][1],
        )

        layers = []
        for layer in samples:
            layers.append(tuple(zip(((j,) for j in qubits), layer)))
        return layers


class EdgeGrabSampler(RBSampler):
    r"""A sampler that uses the edge grab algorithm [1] for sampling gate layers.

    The edge grab sampler, given a list of :math:`w` qubits, their connectivity
    graph, and the desired two-qubit gate density :math:`\xi_s`, outputs a layer
    as follows:

    1. Begin with the empty set :math:`E` and :math:`E_r`, the set of all edges
    in the connectivity graph. Select an edge from :math:`E_r` at random and
    add it to :math:`E`, removing all edges that share a qubit with the edge
    from :math:`E_r`.

    2. Select edges from :math:`E` with the probability :math:`w\xi/2|E|`. These
    edges will have two-qubit gates in the output layer.

    |

    This produces a layer with an expected two-qubit gate density :math:`\xi`. In
    the default mirror RB configuration where these layers are dressed with
    single-qubit Pauli layers, this means the overall two-qubit gate density will be
    :math:`\xi_s/2=\xi`. The overall density will converge to :math:`\xi` as the
    circuit size increases.

    .. ref_arxiv:: 1 2008.11294

    """

    def __call__(
        self,
        qubits: Sequence,
        length: int = 1,
    ) -> List[Tuple[Tuple[int, ...], GateType]]:
        """Sample layers using the edge grab algorithm.

        Args:
            qubits: A sequence of qubits to generate layers for.
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
                (((1, 2), CXGate), ((0,), 12), ((3,), 20))

            This represents a layer where the 12th Clifford is performed on qubit 0,
            a CX is performed with control qubit 1 and target qubit 2, and the 20th
            Clifford is performed on qubit 3.

        """
        num_qubits = len(qubits)
        gateset = self._probs_by_gate_size()
        norm1q = sum(gateset[1][1])
        norm2q = sum(gateset[2][1])
        two_qubit_gate_density = sum(gateset[2][1]) / (sum(gateset[2][1]) + sum(gateset[1][1]))
        if num_qubits == 1:
            return [
                (((qubits[0],), i),)
                for i in self._rng.choice(
                    gateset[1][0], p=[i / norm1q for i in gateset[1][1]], size=length
                )
            ]

        if not isinstance(self.coupling_map, List):
            raise QiskitError("The coupling map must be set correctly before sampling.")

        layer_list = []
        for _ in list(range(length)):
            all_edges = self.coupling_map[:]  # make copy of coupling map from which we pop edges
            selected_edges = []
            while all_edges:
                rand_edge = all_edges.pop(self._rng.integers(len(all_edges)))
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
            if two_qubit_prob > 1 and not np.isclose(two_qubit_prob, 1):
                warnings.warn(
                    "Mean number of two-qubit gates is higher than the number of selected edges. "
                    + "Actual density of two-qubit gates will likely be lower than input density."
                )

            put_1q_gates = set(qubits)
            # put_1q_gates is a list of qubits that aren't assigned to a 2-qubit gate
            # 1-qubit gates will be assigned to these edges
            layer = []
            for edge in selected_edges:
                if self._rng.random() < two_qubit_prob:
                    # with probability two_qubit_prob, place a two-qubit gate from the
                    # gate set on edge in selected_edges
                    if len(gateset[2][0]) == 1:
                        layer.append((tuple(edge), gateset[2][0][0]))
                    else:
                        layer.append(
                            (
                                tuple(edge),
                                self._rng.choice(
                                    gateset[2][0],
                                    p=[x / norm2q for x in gateset[2][1]],
                                ),
                            ),
                        )
                    # remove these qubits from put_1q_gates
                    put_1q_gates.remove(edge[0])
                    put_1q_gates.remove(edge[1])
            for q in put_1q_gates:
                if sum(gateset[1][1]) > 0:
                    layer.append(
                        (
                            (q,),
                            self._rng.choice(gateset[1][0], p=[x / norm1q for x in gateset[1][1]]),
                        )
                    )
                else:  # edge case of two qubit density of 1 where we still fill gaps
                    layer.append(
                        (
                            (q,),
                            self._rng.choice(gateset[1][0]),
                        )
                    )
            layer_list.append(tuple(layer))
        return layer_list
