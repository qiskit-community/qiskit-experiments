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
from typing import Optional, Union, List, Tuple, Sequence, NamedTuple, Dict
from collections import defaultdict
from typing import Iterator
from numpy.random import Generator, default_rng, BitGenerator, SeedSequence
import numpy as np

from qiskit.circuit import Instruction
from qiskit.circuit.gate import Gate
from qiskit.exceptions import QiskitError
from qiskit.transpiler import CouplingMap

from .clifford_utils import CliffordUtils, _CLIFF_SINGLE_GATE_MAP_1Q


class GenericClifford(Gate):
    """Representation of a generic multi-qubit Clifford gate for sampling."""

    def __init__(self, n_qubits):
        super().__init__("generic_clifford", n_qubits, [])


class GenericPauli(Gate):
    """Representation of a generic multi-qubit Pauli gate for sampling."""

    def __init__(self, n_qubits):
        super().__init__("generic_pauli", n_qubits, [])


class GateInstruction(NamedTuple):
    """Named tuple class for sampler output."""

    # the list of qubits to apply the operation on
    qargs: tuple
    # the operation to apply
    op: Instruction


class GateDistribution(NamedTuple):
    """Named tuple class for sampler input distribution."""

    # probability with which to sample the instruction
    prob: float
    # the instruction to include in sampling
    op: Instruction


class BaseSampler(ABC):
    """Base class for samplers that generate circuit layers based on a defined
    algorithm and gate set. Subclasses must implement the ``__call__()`` method
    which outputs a number of circuit layers."""

    def __init__(
        self,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
    ) -> None:
        """Initializes the sampler.

        Args:
            seed: Seed for random generation.
            gate_distribution: The gate distribution for sampling.
        """
        self.seed = seed

    @property
    def seed(self) -> Union[int, SeedSequence, BitGenerator, Generator]:
        """The seed for random generation."""
        return self._rng

    @seed.setter
    def seed(self, seed) -> None:
        self._rng = default_rng(seed)

    @property
    def gate_distribution(self) -> List[GateDistribution]:
        """The gate distribution for sampling. The distribution is a list of
        ``GateDistribution`` named tuples with field names ``(prob, op)``, where
        the probabilites must sum to 1 and ``op`` is the Instruction instance to
        be sampled. An example distribution for the edge grab sampler is

        .. parsed-literal::
            [(0.8, GenericClifford(1)), (0.2, CXGate())]
        """
        return self._gate_distribution

    @gate_distribution.setter
    def gate_distribution(self, dist: List[GateDistribution]) -> None:
        """Set the distribution of gates used in the sampler.

        Args:
            dist: A list of tuples with format ``(probability, gate)``.
        """
        # cast to named tuple
        try:
            dist = [GateDistribution(*elem) for elem in dist]
        except TypeError as exc:
            raise TypeError(
                "The gate distribution should be a sequence of (prob, op) tuples."
            ) from exc
        if sum(list(zip(*dist))[0]) != 1:
            raise QiskitError("Gate distribution probabilities must sum to 1.")
        for gate in dist:
            if not isinstance(gate.op, Instruction):
                raise TypeError(
                    "The only allowed gates in the distribution are Instruction instances."
                )
        self._gate_distribution = dist

    def _probs_by_gate_size(self, distribution: Sequence[GateDistribution]) -> Dict:
        """Return a list of gates and their probabilities indexed by the size of the gate."""

        gate_probs = defaultdict(list)

        for gate in distribution:
            if gate.op.name == "generic_clifford":
                if gate.op.num_qubits == 1:
                    gateset = list(range(CliffordUtils.NUM_CLIFFORD_1_QUBIT))
                    probs = [
                        gate.prob / CliffordUtils.NUM_CLIFFORD_1_QUBIT
                    ] * CliffordUtils.NUM_CLIFFORD_1_QUBIT
                elif gate.op.num_qubits == 2:
                    gateset = list(range(CliffordUtils.NUM_CLIFFORD_2_QUBIT))
                    probs = [
                        gate.prob / CliffordUtils.NUM_CLIFFORD_2_QUBIT
                    ] * CliffordUtils.NUM_CLIFFORD_2_QUBIT
                else:
                    raise QiskitError(
                        "Generic Cliffords larger than 2-qubit are not currently supported."
                    )
            elif gate.op.name == "generic_pauli":
                if gate.op.num_qubits == 1:
                    gateset = [
                        _CLIFF_SINGLE_GATE_MAP_1Q[("id", (0,))],
                        _CLIFF_SINGLE_GATE_MAP_1Q[("x", (0,))],
                        _CLIFF_SINGLE_GATE_MAP_1Q[("y", (0,))],
                        _CLIFF_SINGLE_GATE_MAP_1Q[("z", (0,))],
                    ]
                    probs = [gate.prob / len(gateset)] * len(gateset)
                else:
                    raise QiskitError(
                        "Generic Paulis larger than 1-qubit are not currently supported."
                    )
            else:
                gateset = [gate.op]
                probs = [gate.prob]
            if len(gate_probs[gate.op.num_qubits]) == 0:
                gate_probs[gate.op.num_qubits] = [gateset, probs]
            else:
                gate_probs[gate.op.num_qubits][0].extend(gateset)
                gate_probs[gate.op.num_qubits][1].extend(probs)
        return gate_probs

    @abstractmethod
    def __call__(self, qubits: Sequence, length: int = 1) -> Iterator[Tuple[GateInstruction, ...]]:
        """Samplers should define this method such that it returns sampled layers
        given the input parameters. Each layer is represented by a list of
        ``GateInstruction`` namedtuples, where ``GateInstruction.op`` is the gate to be
        applied and ``GateInstruction.qargs`` is the tuple of qubit indices to
        apply the gate to.

        Args:
            qubits: A sequence of qubits to generate layers for.
            length: The number of layers to generate. Defaults to 1.

        Returns:
            A generator of layers consisting of GateInstruction objects.
        """
        raise NotImplementedError


class SingleQubitSampler(BaseSampler):
    """A sampler that samples layers of random single-qubit gates from a specified gate set."""

    @BaseSampler.gate_distribution.setter
    def gate_distribution(self, dist: List[GateDistribution]) -> None:
        """Set the distribution of gates used in the sampler.

        Args:
            dist: A list of tuples with format ``(probability, gate)``.
        """
        super(SingleQubitSampler, type(self)).gate_distribution.fset(self, dist)

        gateset = self._probs_by_gate_size(dist)
        if not math.isclose(sum(gateset[1][1]), 1):
            raise QiskitError(
                "The distribution for SingleQubitSampler should be all single qubit gates."
            )

    def __call__(
        self,
        qubits: Sequence,
        length: int = 1,
    ) -> Iterator[Tuple[GateInstruction]]:
        """Samples random single-qubit gates from the specified gate set. The
        input gate distribution must consist solely of single qubit gates.

        Args:
            qubits: A sequence of qubits to generate layers for.
            length: The length of the sequence to output.

        Returns:
            A ``length``-long iterator of :class:`qiskit.circuit.QuantumCircuit`
            layers over ``qubits``. Each layer is represented by a list of
            ``GateInstruction`` tuples where ``GateInstruction.op`` is the gate
            to be applied and ``GateInstruction.qargs`` is the tuple of qubit
            indices to apply the gate to. Single-qubit Cliffords are represented
            by integers for speed.
        """

        gateset = self._probs_by_gate_size(self._gate_distribution)

        samples = self._rng.choice(
            np.array(gateset[1][0], dtype=object),
            size=(length, len(qubits)),
            p=gateset[1][1],
        )

        for samplelayer in samples:
            yield tuple(GateInstruction(*ins) for ins in zip(((j,) for j in qubits), samplelayer))


class EdgeGrabSampler(BaseSampler):
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

    This produces a layer with an expected two-qubit gate density :math:`\xi`. In the
    default mirror RB configuration where these layers are dressed with single-qubit
    Pauli layers, this means the overall expected two-qubit gate density will be
    :math:`\xi_s/2=\xi`. The actual density will converge to :math:`\xi_s` as the
    circuit size increases.

    .. ref_arxiv:: 1 2008.11294

    """

    def __init__(
        self,
        gate_distribution=None,
        coupling_map: Optional[Union[List[List[int]], CouplingMap]] = None,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
    ) -> None:
        """Initializes the sampler.

        Args:
            seed: Seed for random generation.
            gate_distribution: The gate distribution for sampling.
            coupling_map: The coupling map between the qubits.
        """
        super().__init__(seed)
        self._gate_distribution = gate_distribution
        self.coupling_map = coupling_map

    @property
    def coupling_map(self) -> CouplingMap:
        """The coupling map of the system to sample over."""
        return self._coupling_map

    @coupling_map.setter
    def coupling_map(self, coupling_map: Union[List[List[int]], CouplingMap]) -> None:
        try:
            self._coupling_map = CouplingMap(coupling_map)
        except (ValueError, TypeError) as exc:
            raise TypeError("Invalid coupling map provided.") from exc

    def __call__(
        self,
        qubits: Sequence,
        length: int = 1,
    ) -> Iterator[Tuple[GateInstruction]]:
        """Sample layers using the edge grab algorithm.

        Args:
            qubits: A sequence of qubits to generate layers for.
            length: The length of the sequence to output.

        Raises:
            Warning: If the coupling map has no connectivity or
                ``two_qubit_gate_density`` is too high.
            TypeError: If invalid gate set(s) are specified.
            QiskitError: If the coupling map is invalid.

        Returns:
            A ``length``-long iterator of :class:`qiskit.circuit.QuantumCircuit`
            layers over ``num_qubits`` qubits. Each layer is represented by a
            list of ``GateInstruction`` named tuples which are in the format
            (qargs, gate). Single-qubit Cliffords are represented by integers
            for speed. Here's an example with the default choice of Cliffords
            for the single-qubit gates and CXs for the two-qubit gates:

            .. parsed-literal::
                (((1, 2), CXGate()), ((0,), 12), ((3,), 20))

            This represents a layer where the 12th Clifford is performed on qubit 0,
            a CX is performed with control qubit 1 and target qubit 2, and the 20th
            Clifford is performed on qubit 3.

        """
        num_qubits = len(qubits)
        gateset = self._probs_by_gate_size(self._gate_distribution)
        try:
            norm1q = sum(gateset[1][1])
            norm2q = sum(gateset[2][1])
        except KeyError as exc:
            raise QiskitError(
                "The edge grab sampler requires 1-qubit and 2-qubit gates to be specified."
            ) from exc
        if not np.isclose(norm1q + norm2q, 1):
            raise QiskitError("The edge grab sampler only supports 1- and 2-qubit gates.")
        two_qubit_gate_density = norm2q / (norm1q + norm2q)

        for _ in range(length):
            all_edges = self.coupling_map.get_edges()[
                :
            ]  # make copy of coupling map from which we pop edges
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
                        layer.append(GateInstruction(tuple(edge), gateset[2][0][0]))
                    else:
                        layer.append(
                            GateInstruction(
                                tuple(edge),
                                self._rng.choice(
                                    np.array(gateset[2][0], dtype=Instruction),
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
                        GateInstruction(
                            (q,),
                            self._rng.choice(
                                np.array(gateset[1][0], dtype=Instruction),
                                p=[x / norm1q for x in gateset[1][1]],
                            ),
                        ),
                    )
                else:  # edge case of two qubit density of 1 where we still fill gaps
                    layer.append(
                        GateInstruction(
                            (q,), self._rng.choice(np.array(gateset[1][0], dtype=Instruction))
                        ),
                    )
            yield tuple(layer)
