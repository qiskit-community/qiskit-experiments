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
from typing import Union, Iterable, Optional, List, Sequence
from numbers import Integral
from itertools import permutations


from numpy.random import Generator, BitGenerator, SeedSequence, default_rng

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit, Instruction, Barrier
from qiskit.quantum_info import Clifford, random_pauli, random_clifford
from qiskit.quantum_info.operators import Pauli
from qiskit.providers.backend import Backend
from qiskit.providers.options import Options
from qiskit.transpiler.basepasses import TransformationPass
from qiskit_experiments.warnings import deprecate_arguments

from .rb_experiment import StandardRB, SequenceElementType
from .mirror_rb_analysis import MirrorRBAnalysis
from .clifford_utils import compute_target_bitstring
from .sampling_utils import MirrorRBSampler, EdgeGrabSampler


class MirrorRB(StandardRB):
    """An experiment to measure gate infidelity using random mirrored layers of Clifford
    and Pauli gates.

    # section: overview
        Mirror randomized benchmarking (mirror RB) is a method to estimate the average
        error-rate of quantum gates that is more scalable than the standard RB methods.

        A mirror RB experiment generates circuits of layers of Cliffords interleaved
        with layers of Pauli gates and capped at the start and end by a layer of
        single-qubit Cliffords. The second half of the Clifford layers are the
        inverses of the first half of Clifford layers. After running the circuits on
        a backend, various quantities (success probability, adjusted success
        probability, and effective polarization) are computed and used to fit an
        exponential decay curve and calculate the EPC (error per Clifford, also
        referred to as the average gate infidelity) and entanglement infidelity (see
        references for more info).

    # section: analysis_ref
        :class:`MirrorRBAnalysis`

    # section: reference
        .. ref_arxiv:: 1 2112.09853
        .. ref_arxiv:: 2 2008.11294
        .. ref_arxiv:: 3 2204.07568

    """

    @deprecate_arguments({"qubits": "physical_qubits"}, "0.5")
    def __init__(
        self,
        physical_qubits: Sequence[int],
        lengths: Iterable[int],
        distribution: MirrorRBSampler = EdgeGrabSampler,
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
                Defaults to the :class:`.EdgeGrabSampler`.
            local_clifford: If True, begin the circuit with uniformly random 1-qubit
                Cliffords and end the circuit with their inverses.
            pauli_randomize: If True, surround each inner Clifford layer with
                uniformly random Paulis.
            two_qubit_gate_density: Expected proportion of qubits with CNOTs based on
                the backend coupling map.
            backend: The backend to run the experiment on.
            num_samples: Number of samples to generate for each sequence length.
            seed: Optional, seed used to initialize ``numpy.random.default_rng``.
                when generating circuits. The ``default_rng`` will be initialized
                with this seed value everytime :meth:`circuits` is called.
            full_sampling: If True all Cliffords are independently sampled for
                all lengths. If False for sample of lengths longer sequences are
                constructed by appending additional Clifford samples to shorter
                sequences.
            inverting_pauli_layer: If True, a layer of Pauli gates is appended at the
                end of the circuit to set all qubits to 0.

        Raises:
            QiskitError: if an odd length or a negative two qubit gate density is provided
        """

        if not all(length % 2 == 0 for length in lengths):
            raise QiskitError("All lengths must be even")

        if two_qubit_gate_density < 0 or two_qubit_gate_density > 1:
            raise QiskitError("Two-qubit gate density must be between 0 and 1.")

        super().__init__(
            physical_qubits,
            lengths,
            backend=backend,
            num_samples=num_samples,
            seed=seed,
            full_sampling=full_sampling,
        )

        self._distribution = distribution()

        self.set_experiment_options(
            distribution=distribution,
            local_clifford=local_clifford,
            pauli_randomize=pauli_randomize,
            inverting_pauli_layer=inverting_pauli_layer,
            two_qubit_gate_density=two_qubit_gate_density,
        )

        self.analysis = MirrorRBAnalysis()

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            local_clifford (bool): Whether to begin the circuit with uniformly random 1-qubit
                Cliffords and end the circuit with their inverses.
            pauli_randomize (bool): Whether to surround each inner Clifford layer with
                uniformly random Paulis.
            inverting_pauli_layer (bool): Whether to append a layer of Pauli gates at the
                end of the circuit to set all qubits to 0
            two_qubit_gate_density (float): Expected proportion of two-qubit gates in
                the mirror circuit layers (not counting Clifford or Pauli layers at the
                start and end).
            num_samples (int): Number of samples to generate for each sequence length.
        """
        options = super()._default_experiment_options()
        options.update_options(
            local_clifford=True,
            pauli_randomize=True,
            two_qubit_gate_density=0.2,
            distribution=None,
            inverting_pauli_layer=False,
        )

        return options

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of Mirror RB circuits.

        Returns:
            A list of :class:`QuantumCircuit`.
        """
        sequences = self._sample_sequences()
        circuits = self._sequences_to_circuits(sequences)

        return circuits

    def _sample_sequences(self) -> List[Sequence[SequenceElementType]]:
        """Sample layers of mirror RB using the provided distribution and user options.
        First, layers are sampled using the distribution, then Pauli-dressed if
        ``pauli_randomize`` is ``True``. The inverse of the resulting circuit is
        appended to the end. If ``local_clifford`` is ``True``, then cliffords are added
        to the beginning and end. If ``inverting_pauli_layer`` is ``True``, a Pauli
        layer will be appended at the end to set the output bitstring to all zeros.

        Raises:
            QiskitError: If no backend is provided.

        Returns:
            A list of mirror RB sequences. Each element is a list of layers with length
            matching the corresponding element in ``lengths``.
        """
        rng = default_rng(seed=self.experiment_options.seed)
        if not self._backend:
            raise QiskitError("A backend must be provided.")

        # Coupling map is full connectivity by default. If backend has a coupling map,
        # get backend coupling map and create coupling map for physical qubits
        coupling_map = list(permutations(range(max(self.physical_qubits) + 1), 2))
        if self._backend_data.coupling_map:
            coupling_map = self._backend_data.coupling_map
        experiment_coupling_map = []
        for edge in coupling_map:
            if edge[0] in self.physical_qubits and edge[1] in self.physical_qubits:
                experiment_coupling_map.append(edge)

        # adjust the density based on whether the pauli layers are in
        if self.experiment_options.pauli_randomize:
            adjusted_2q_density = self.experiment_options.two_qubit_gate_density * 2
        else:
            adjusted_2q_density = self.experiment_options.two_qubit_gate_density

        sequences = []

        if self.experiment_options.full_sampling:
            for _ in range(self.experiment_options.num_samples):
                for length in self.experiment_options.lengths:
                    # Sample Clifford layer elements for first half of mirror circuit
                    elements = self._distribution(
                        self.num_qubits,
                        adjusted_2q_density,
                        experiment_coupling_map,
                        length // 2,
                        seed=rng,
                    )
                    # Append inverses of Clifford elements to second half of circuit
                    for element in elements[::-1]:
                        elements.append(self._adjoint_clifford(element))

                    # Interleave random Paulis if set by user
                    if self.experiment_options.pauli_randomize:
                        elements = self._pauli_dress(elements, rng)

                    # Add start and end local cliffords if set by user
                    if self.experiment_options.local_clifford:
                        elements = self._start_end_cliffords(elements, rng)

                    sequences.append(elements)
        else:
            for _ in range(self.experiment_options.num_samples):
                longest_seq = self._distribution(
                    self.num_qubits,
                    adjusted_2q_density,
                    experiment_coupling_map,
                    max(self.experiment_options.lengths) // 2,
                    seed=rng,
                )
                element_lengths = self.experiment_options.lengths

                # Append inverses of Clifford elements to second half of circuit
                for element in longest_seq[::-1]:
                    longest_seq.append(self._adjoint_clifford(element))
                # Interleave random Paulis if set by user
                if self.experiment_options.pauli_randomize:
                    longest_seq = self._pauli_dress(longest_seq, rng)
                    element_lengths = [length * 2 + 1 for length in element_lengths]

                # Add start and end local cliffords if set by user
                if self.experiment_options.local_clifford:
                    longest_seq = self._start_end_cliffords(longest_seq, rng)
                    element_lengths = [length + 2 for length in element_lengths]

                # Construct the remaining sequences from the longest
                for real_length in element_lengths:
                    sequences.append(
                        longest_seq[: real_length // 2] + longest_seq[-real_length // 2 :]
                    )

        return sequences

    def _adjoint_clifford(self, op: SequenceElementType) -> SequenceElementType:
        if isinstance(op, QuantumCircuit):
            return Clifford.from_circuit(op).adjoint()
        return op.adjoint()

    def _sequences_to_circuits(
        self, sequences: List[Sequence[SequenceElementType]]
    ) -> List[QuantumCircuit]:
        """Convert Mirror RB sequences into mirror circuits.

        Args:
            sequences: List of sequences whose elements are full circuit layers.

        Returns:
            A list of RB circuits.
        """
        basis_gates = self._get_basis_gates()
        circuits = []
        for i, seq in enumerate(sequences):
            circ = QuantumCircuit(self.num_qubits)
            for elem in seq:
                circ.append(elem.to_instruction(), circ.qubits)
                circ.append(Barrier(self.num_qubits), circ.qubits)

            circ.metadata = {
                "xval": self.experiment_options.lengths[i % len(self.experiment_options.lengths)],
                "group": "Clifford",
                "physical_qubits": self.physical_qubits,
                "target": compute_target_bitstring(circ),
                "inverting_pauli_layer": self.experiment_options.inverting_pauli_layer,
            }

            if self.experiment_options.inverting_pauli_layer:
                # Get target bitstring (ideal bitstring outputted by the circuit)
                target = circ.metadata["target"]

                # Pauli gates to apply to each qubit to reset each to the state 0.
                # E.g., if the ideal bitstring is 01001, the Pauli label is IXIIX,
                # which sets all qubits to 0 (up to a global phase)
                label = "".join(["X" if char == "1" else "I" for char in target])
                circ.append(Pauli(label), list(range(self._num_qubits)))

            circ.measure_all()  # includes insertion of the barrier before measurement
            circuits.append(circ)
        return circuits

    def _pauli_dress(self, element_list: List, rng: Generator) -> List:
        """Interleaving layers of random Paulis inside the element list.

        Args:
            element_list: The list of elements we add the interleaved Paulis to.
            rng: Randomness generator

        Returns:
            The new list of elements with the Paulis interleaved.
        """
        rand_pauli = random_pauli(self._num_qubits, seed=rng)
        new_element_list = [rand_pauli]
        for element in element_list:
            new_element_list.append(element)
            rand_pauli = random_pauli(self._num_qubits, seed=rng)
            new_element_list.append(rand_pauli)
        return new_element_list

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        return super()._transpiled_circuits(custom_transpile=True)

    def _start_end_cliffords(
        self, elements: Iterable[Clifford], rng: Generator
    ) -> List[QuantumCircuit]:
        """Add a layer of uniformly random 1-qubit Cliffords to the beginning of the list
           and its inverse to the end of the list.

        Args:
            element_list: The list of elements we add the Clifford layers to
            rng: Randomness generator

        Returns:
            The new list of elements with the start and end local (1-qubit) Cliffords.
        """

        rand_clifford = [random_clifford(1, rng) for i in range(self.num_qubits)]
        tensor_op = rand_clifford[0]
        for cliff in rand_clifford[1:]:
            tensor_op = tensor_op ^ cliff
        tensor_circ = tensor_op.to_circuit()

        rand_clifford = random_clifford(self.num_qubits, seed=rng).to_circuit()
        return [tensor_circ] + elements + [tensor_circ.inverse()]
