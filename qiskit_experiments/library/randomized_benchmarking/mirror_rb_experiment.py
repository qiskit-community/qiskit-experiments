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
Mirror RB Experiment class.
"""
import warnings
from abc import ABC, abstractmethod
from typing import Union, Iterable, Optional, List, Sequence
from itertools import permutations
from numbers import Integral

from numpy.random import Generator, BitGenerator, SeedSequence, default_rng

from qiskit import QuantumCircuit, QiskitError
from qiskit.circuit import Instruction
from qiskit.quantum_info import Clifford, random_pauli, random_clifford
from qiskit.quantum_info.operators import Pauli
from qiskit.providers.backend import Backend
from qiskit.providers.options import Options
from qiskit.transpiler.basepasses import TransformationPass

from .rb_experiment import StandardRB
from .mirror_rb_analysis import MirrorRBAnalysis
from .clifford_utils import CliffordUtils

SequenceElementType = Union[Clifford, Integral, QuantumCircuit]


class MirrorRBDistribution(ABC):
    """Sampling distribution for the mirror randomized benchmarking experiment."""

    def __init__(self, seed=None):
        self.rng = default_rng(seed)

    @abstractmethod
    def __call__(self, qubits, two_qubit_density, coupling_map, **params):
        self.qubits = qubits


class RandomEdgeGrabDistribution(MirrorRBDistribution):
    """The edge grab algorithm for sampling one- and two-qubit layers.

    # section: overview



    # section: reference
        .. ref_arxiv:: 1 2008.11294

    """

    def __init__(self, seed=None):
        super().__init__(seed)

    def __call__(self, qubits, two_qubit_gate_density, coupling_map, length, seed=None):
        """Sample layers using the ege grab algorithm.

        Args:
            qubits: The number of qubits in the circuit.
            two_qubit_gate_density: :math:`1/2` times the expected fraction
                of qubits with CX gates.
            coupling_map: List of pairs of connected edges between qubits.
            length: The length of the sequence to output.
            seed: Seed for random generation.

        Raises:
            Warning: If device has no connectivity or two_qubit_gate_density is too high

        Returns:
            List of QuantumCircuits

        """
        self.two_qubit_density = two_qubit_gate_density
        self.coupling_map = coupling_map

        num_qubits = len(qubits)

        if num_qubits == 1:
            return rng.integers(CliffordUtils.NUM_CLIFFORD_1_QUBIT, size=length)

        if rng is None:
            rng = default_rng(seed=seed)

        if isinstance(rng, int):
            rng = default_rng(rng)

        qc_list = []
        for _ in list(range(size)):
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

            qr = QuantumRegister(num_qubits)
            qc = QuantumCircuit(qr)
            two_qubit_prob = 0
            try:
                two_qubit_prob = num_qubits * two_qubit_gate_density / len(selected_edges)
            except ZeroDivisionError:
                warnings.warn(
                    "Device has no connectivity. All cliffords will be single-qubit Cliffords"
                )
            if two_qubit_prob > 1:
                warnings.warn(
                    "Mean number of two-qubit gates is higher than number of selected edges for CNOTs. "
                    + "Actual density of two-qubit gates will likely be lower than input density"
                )
            selected_edges_logical = [
                [np.where(q == np.asarray(qubits))[0][0] for q in edge] for edge in selected_edges
            ]
            # selected_edges_logical is selected_edges with logical qubit labels rather than physical
            # ones. Example: qubits = (8,4,5,3,7), selected_edges = [[4,8],[7,5]]
            # ==> selected_edges_logical = [[1,0],[4,2]]
            put_1_qubit_clifford = np.arange(num_qubits)
            # put_1_qubit_clifford is a list of qubits that aren't assigned to a 2-qubit Clifford
            # 1-qubit Clifford will be assigned to these edges
            for edge in selected_edges_logical:
                if rng.random() < two_qubit_prob:
                    # with probability two_qubit_prob, place CNOT on edge in selected_edges
                    qc.cx(edge[0], edge[1])
                    # remove these qubits from put_1_qubit_clifford
                    put_1_qubit_clifford = np.setdiff1d(put_1_qubit_clifford, edge)
            for q in put_1_qubit_clifford:
                clifford1q = self.clifford_1_qubit_circuit(rng.integers(24))
                insts = [datum[0] for datum in clifford1q.data]
                for inst in insts:
                    qc.compose(inst, [q], inplace=True)
            qc_list.append(qc)
        return qc_list


class MirrorRB(StandardRB):
    """Mirror randomized benchmarking experiment.

    # section: overview
        Mirror Randomized Benchmarking (RB) is a method to estimate the average
        error-rate of quantum gates that is more scalable than the standard RB methods.

        A mirror RB experiment generates circuits of layers of Cliffords interleaved
        with layers of Pauli gates and capped at the start and end by a layer of
        single-qubit Cliffords. The second half of the Clifford layers are the
        inverses of the first half of Clifford layers. After running the circuits on
        a backend, various quantities (success probability, adjusted success
        probability, and effective polarization) are computed and used to fit an
        exponential decay curve and calculate the EPC (error per Clifford, also
        referred to as the average gate infidelity) and entanglement infidelity. Find
        more on adjusted success probability, effective polarization, and
        entanglement infidelity in Refs. [1, 2, 3].

    # section: analysis_ref
        :py:class:`MirrorRBAnalysis`

    # section: reference
        .. ref_arxiv:: 1 2112.09853
        .. ref_arxiv:: 2 2008.11294
        .. ref_arxiv:: 3 2204.07568

    """

    def __init__(
        self,
        physical_qubits: Sequence[int],
        lengths: Iterable[int],
        distribution: MirrorRBDistribution,
        local_clifford: bool = True,
        pauli_randomize: bool = True,
        two_qubit_gate_density: float = 0.2,
        backend: Optional[Backend] = None,
        num_samples: int = 3,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        full_sampling: bool = False,
        inverting_pauli_layer: bool = False,
    ):
        """Initialize a mirror randomized benchmarking experiment.

        Args:
            physical_qubits: A list of physical qubits for the experiment.
            lengths: A list of RB sequences lengths.
            distribution: The probability distribution over the layer set to sample.
            local_clifford: If True, begin the circuit with uniformly random 1-qubit
                            Cliffords and end the circuit with their inverses.
            pauli_randomize: If True, surround each inner Clifford layer with
                             uniformly random Paulis.
            two_qubit_gate_density: Expected proportion of qubits with CNOTs based on
                                    the backend coupling map.
            backend: The backend to run the experiment on.
            num_samples: Number of samples to generate for each
                         sequence length.
            seed: Optional, seed used to initialize ``numpy.random.default_rng``.
                  when generating circuits. The ``default_rng`` will be initialized
                  with this seed value everytime :meth:`circuits` is called.
            full_sampling: If True all Cliffords are independently sampled for
                           all lengths. If False for sample of lengths longer
                           sequences are constructed by appending additional
                           Clifford samples to shorter sequences.
            inverting_pauli_layer: If True, a layer of Pauli gates is appended at the
                                   end of the circuit to set all qubits to 0 (with
                                   possibly a global phase)

        Raises:
            QiskitError: if an odd length or a negative two qubit gate density is provided
        """
        # All lengths must be even
        if not all(length % 2 == 0 for length in lengths):
            raise QiskitError("All lengths must be even")

        # Two-qubit density must be non-negative
        if two_qubit_gate_density < 0:
            raise QiskitError("Two-qubit gate density must be non-negative")

        super().__init__(
            physical_qubits,
            lengths,
            backend=backend,
            num_samples=num_samples,
            seed=seed,
            full_sampling=full_sampling,
        )

        self._local_clifford = local_clifford
        self._pauli_randomize = pauli_randomize
        self._two_qubit_gate_density = two_qubit_gate_density

        self._full_sampling = full_sampling
        self._clifford_utils = CliffordUtils()
        self._distribution = distribution

        # By default, the inverting Pauli layer at the end of the circuit is not added
        self._inverting_pauli_layer = inverting_pauli_layer

        # Set analysis options
        self.analysis = MirrorRBAnalysis()

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            lengths (List[int]): A list of RB sequences lengths.
            num_samples (int): Number of samples to generate for each sequence length.
            seed (None or int or SeedSequence or BitGenerator or Generator): A seed
                used to initialize ``numpy.random.default_rng`` when generating circuits.
                The ``default_rng`` will be initialized with this seed value everytime
                :meth:`circuits` is called.
        """
        options = super()._default_experiment_options()
        options.update_options(
            distribution=None,
            num_samples=None,
            seed=None,
            full_sampling=None,
        )

        return options

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of Mirror RB circuits.

        Returns:
            A list of :class:`QuantumCircuit`.
        """
        sequences = self._sample_sequences()
        circuits = self._sequences_to_circuits(sequences)

        for circ, seq in zip(circuits, sequences):
            circ.metadata = {
                "experiment_type": self._type,
                "xval": len(seq),
                "group": "Clifford",
                "physical_qubits": self.physical_qubits,
                "target": self._clifford_utils.compute_target_bitstring(circ),
                "inverting_pauli_layer": self._inverting_pauli_layer,
            }
        return circuits

    def _sample_sequences(self) -> List[Sequence[SequenceElementType]]:
        """Sample mirror RB sequences using the provided distribution.

        Returns:
            A list of mirror RB sequences.
        """

        rng = default_rng(seed=self.experiment_options.seed)
        sequences = []
        if self.experiment_options.full_sampling:
            for _ in range(self.experiment_options.num_samples):
                for length in self.experiment_options.lengths:
                    sequences.append(self._distribution(length, rng))
        else:
            for _ in range(self.experiment_options.num_samples):
                longest_seq = self.__sample_sequence(max(self.experiment_options.lengths), rng)
                for length in self.experiment_options.lengths:
                    sequences.append(longest_seq[:length])

        return sequences

    def _sample_circuits(self, lengths, rng) -> List[QuantumCircuit]:
        """Sample Mirror RB circuits.

        Steps:
        1. Sample length/2 layers of random Cliffords
        2. Compute inverse of each layer in the first half of the circuit and append to circuit
        3. Sample the random paulis and interleave them between the Clifford layers
        4. Sample the 1-qubit local Clifford and add them to the beginning and end of the circuit

        Args:
            lengths: List of lengths to run Mirror RB
            rng: Generator seed

        Returns:
            List of QuantumCircuits

        Raises:
            QiskitError: if backend without a coupling map is provided
        """

        # Backend must have a coupling map
        if not self._backend:
            raise QiskitError("Must provide a backend")

        circuits = []
        lengths_half = [length // 2 for length in lengths]

        # Coupling map is full connectivity by default. If backend has a coupling map,
        # get backend coupling map and create coupling map for physical qubits
        coupling_map = list(permutations(range(max(self.physical_qubits) + 1), 2))
        if self._backend.configuration().coupling_map:
            coupling_map = self._backend.configuration().coupling_map
        experiment_coupling_map = []
        for edge in coupling_map:
            if edge[0] in self.physical_qubits and edge[1] in self.physical_qubits:
                experiment_coupling_map.append(edge)

        for length in lengths_half if self._full_sampling else [lengths_half[-1]]:
            # Sample Clifford layer elements for first half of mirror circuit
            elements = self._clifford_utils.random_edgegrab_clifford_circuits(
                self.physical_qubits,
                experiment_coupling_map,
                self._two_qubit_gate_density,
                length,
                rng,
            )

            # Append inverses of Clifford elements to second half of circuit
            for element in elements[::-1]:
                elements.append(element.inverse())
            element_lengths = [len(elements)] if self._full_sampling else lengths

            # Interleave random Paulis if set by user
            if self._pauli_randomize:
                elements = self._pauli_dress(elements, rng)
                element_lengths = [length * 2 + 1 for length in element_lengths]

            # Add start and end local cliffords if set by user
            if self._local_clifford:
                element_lengths = [length + 2 for length in element_lengths]
                elements = self._start_end_cliffords(elements, rng)
            mirror_circuits = self._generate_mirror(elements, element_lengths)
            for circuit in mirror_circuits:
                # Use "boolean arithmetic" to calculate xval correctly for each circuit
                pauli_scale = self._pauli_randomize + 1
                clifford_const = self._local_clifford * 2
                circuit.metadata["xval"] = (
                    circuit.metadata["xval"] - self._pauli_randomize - clifford_const
                ) // pauli_scale
                circuit.metadata["mirror"] = True
            circuits += mirror_circuits

        # Append inverting Pauli layer at end of circuit if set by user
        if self._inverting_pauli_layer:
            for circuit in circuits:
                # Get target bitstring (ideal bitstring outputted by the circuit)
                target = circuit.metadata["target"]

                # Pauli gates to apply to each qubit to reset each to the state 0.
                # E.g., if the ideal bitstring is 01001, the Pauli label is IXIIX,
                # which sets all qubits to 0 (up to a global phase)
                label = "".join(["X" if char == "1" else "I" for char in target])
                circuit.remove_final_measurements()
                circuit.append(Pauli(label), list(range(self._num_qubits)))
                circuit.measure_all()

        return circuits

    def _pauli_dress(self, element_list: List, rng: Optional[Union[int, Generator]]) -> List:
        """Interleaving layers of random Paulis inside the element list.

        Args:
            element_list: The list of elements we add the interleaved Paulis to.
            rng: (Seed for) random number generator

        Returns:
            The new list of elements with the Paulis interleaved.
        """
        # Generate random Pauli
        rand_pauli = random_pauli(self._num_qubits, seed=rng).to_instruction()
        rand_pauli_op = Clifford(rand_pauli)
        new_element_list = [(rand_pauli, rand_pauli_op)]
        for element in element_list:
            new_element_list.append(element)
            rand_pauli = random_pauli(self._num_qubits, seed=rng).to_instruction()
            rand_pauli_op = Clifford(rand_pauli)
            new_element_list.append((rand_pauli, rand_pauli_op))
        return new_element_list

    def _start_end_cliffords(
        self, elements: Iterable[Clifford], rng: Optional[Union[int, Generator]]
    ) -> List[QuantumCircuit]:
        """Add a layer of uniformly random  1-qubit Cliffords to the beginning of the list
           and its inverse to the end of the list

        Args:
            element_list: The list of elements we add the Clifford layers to
            rng: (Seed for) random number generator

        Returns:
            The new list of elements with the start and end local (1-qubit) Cliffords.
        """
        rand_clifford = [
            self._clifford_utils.random_cliffords(num_qubits=1, rng=rng)[0]
            for _ in self.physical_qubits
        ]
        tensor_op = rand_clifford[0]
        for cliff in rand_clifford[1:]:
            tensor_op = tensor_op ^ cliff
        tensor_circ = tensor_op.to_circuit()

        rand_clifford = random_clifford(self.num_qubits, seed=rng).to_circuit()
        return [tensor_circ] + elements + [tensor_circ.inverse()]

    def _generate_mirror(
        self, elements: Iterable[Clifford], lengths: Iterable[int]
    ) -> List[QuantumCircuit]:
        """Return the RB circuits constructed from the given element list with the second
           half as the inverse of the first half

        Args:
            elements: A list of Clifford elements
            lengths: A list of RB sequences lengths.

        Returns:
            A list of :class:`QuantumCircuit`s.

        Additional information:
            The circuits are constructed iteratively; each circuit is obtained
            by extending the previous circuit (without the inversion and measurement gates)
        """
        qubits = list(range(self.num_qubits))
        circuits = []

        circs = [QuantumCircuit(self.num_qubits) for _ in range(len(lengths))]

        for current_length, group_elt_circ in enumerate(elements[: (len(elements) // 2)]):
            if isinstance(group_elt_circ, tuple):
                group_elt_gate = group_elt_circ[0]
            else:
                group_elt_gate = group_elt_circ

            if not isinstance(group_elt_gate, Instruction):
                group_elt_gate = group_elt_gate.to_instruction()
            for circ in circs:
                circ.barrier(qubits)
                circ.append(group_elt_gate, qubits)

            double_current_length = (
                (current_length + 1) * 2 + 1 if len(elements) % 2 == 1 else (current_length + 1) * 2
            )
            if double_current_length in lengths:
                rb_circ = circs.pop()
                inv_start = (
                    (-(current_length + 1) - 1) if len(elements) % 2 == 1 else -(current_length + 1)
                )
                for inv in elements[inv_start:]:
                    if isinstance(inv, tuple):
                        group_elt_gate = inv[0]
                    else:
                        group_elt_gate = inv

                    if not isinstance(group_elt_gate, Instruction):
                        group_elt_gate = group_elt_gate.to_instruction()
                    rb_circ.barrier(qubits)
                    rb_circ.append(group_elt_gate, qubits)
                rb_circ.metadata = {
                    "experiment_type": self._type,
                    "xval": double_current_length,
                    "group": "Clifford",
                    "physical_qubits": self.physical_qubits,
                    "target": self._clifford_utils.compute_target_bitstring(rb_circ),
                    "inverting_pauli_layer": self._inverting_pauli_layer,
                }
                rb_circ.measure_all()
                circuits.append(rb_circ)
        return circuits
