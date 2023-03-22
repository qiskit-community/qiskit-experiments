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
from typing import Union, Iterable, Optional, List, Sequence, Tuple
from numbers import Integral
import itertools
from numpy.random import Generator, BitGenerator, SeedSequence, default_rng

from qiskit.circuit import QuantumCircuit, Instruction, Barrier
from qiskit.quantum_info.operators import Pauli
from qiskit.providers.backend import Backend
from qiskit.providers.options import Options
from qiskit.exceptions import QiskitError

from qiskit_experiments.warnings import deprecate_arguments
from .rb_experiment import StandardRB, SequenceElementType
from .mirror_rb_analysis import MirrorRBAnalysis
from .clifford_utils import (
    compute_target_bitstring,
    inverse_1q,
    _clifford_1q_int_to_instruction,
)
from .sampling_utils import RBSampler, EdgeGrabSampler, SingleQubitSampler, GateType


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
        distribution: RBSampler = EdgeGrabSampler,
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
                with this seed value every time :meth:`circuits` is called.
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
                end of the circuit to set all qubits to 0.
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

        Returns:
            A list of mirror RB sequences. Each element is a list of layers with length
            matching the corresponding element in ``lengths``. The layers are made up
            of tuples in the format ((one or more qubit indices), gate). Single-qubit
            Cliffords are represented by integers for speed.

        Raises:
            QiskitError: If no backend is provided.
        """
        if not self._backend:
            raise QiskitError("A backend must be provided for circuit generation.")

        # Coupling map is full connectivity by default. If backend has a coupling map,
        # get backend coupling map and create coupling map for physical qubits converted
        # to qubits 0, 1...n
        coupling_map = list(itertools.permutations(range(max(self.physical_qubits) + 1), 2))
        if self._backend_data.coupling_map:
            coupling_map = self._backend_data.coupling_map

        qmap = {self.physical_qubits[i]: i for i in range(len(self.physical_qubits))}
        experiment_coupling_map = []
        for edge in coupling_map:
            if edge[0] in self.physical_qubits and edge[1] in self.physical_qubits:
                experiment_coupling_map.append((qmap[edge[0]], qmap[edge[1]]))

        rng = default_rng(seed=self.experiment_options.seed)

        sequences = []

        # Adjust the density based on whether the pauli layers are in
        if self.experiment_options.pauli_randomize:
            adjusted_2q_density = self.experiment_options.two_qubit_gate_density * 2
        else:
            adjusted_2q_density = self.experiment_options.two_qubit_gate_density

        # Sequence of lengths to sample for
        if not self.experiment_options.full_sampling:
            seqlens = (max(self.experiment_options.lengths),)
        else:
            seqlens = self.experiment_options.lengths

        for _ in range(self.experiment_options.num_samples):
            for seqlen in seqlens:
                seq = []

                # Sample the first half of the mirror layers
                layers = list(
                    self._distribution(
                        range(self.num_qubits),
                        adjusted_2q_density,
                        experiment_coupling_map,
                        seqlen // 2,
                        seed=rng,
                    )
                )

                if not self.experiment_options.full_sampling:
                    build_seq_lengths = self.experiment_options.lengths

                seq.extend(layers)

                # Add the second half mirror layers
                for i in range(len(list(layers))):
                    seq.append(self._inverse_layer(layers[-i - 1]))

                # Interleave random Paulis if set by user
                if self.experiment_options.pauli_randomize:
                    sampler = SingleQubitSampler()
                    pauli_layers = sampler(range(self.num_qubits), seqlen + 1, "pauli", rng)
                    seq = list(itertools.chain(*zip(pauli_layers[:-1], seq)))
                    seq.append(pauli_layers[-1])
                    if not self.experiment_options.full_sampling:
                        build_seq_lengths = [length * 2 + 1 for length in build_seq_lengths]

                # Add start and end local cliffords if set by user
                if self.experiment_options.local_clifford:
                    cseq = []
                    sampler = SingleQubitSampler()
                    clifford_layers = sampler(range(self.num_qubits), 1, "clifford", rng)
                    cseq.append(clifford_layers[0])
                    cseq.extend(seq)
                    cseq.append(self._inverse_layer(clifford_layers[0]))
                    if not self.experiment_options.full_sampling:
                        build_seq_lengths = [length + 2 for length in build_seq_lengths]
                    seq = cseq

                if self.experiment_options.full_sampling:
                    sequences.append(seq)

            # Construct the rest of the sequences from the longest if `full_sampling` is
            # off
            if not self.experiment_options.full_sampling:
                for real_length in build_seq_lengths:
                    sequences.append(seq[: real_length // 2] + seq[-real_length // 2 :])

        return sequences

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
            # Hack to get target bitstrings until qiskit-terra#9475 is resolved
            circ_target = QuantumCircuit(self.num_qubits)
            for layer in seq:
                for elem in layer:
                    circ.append(self._to_instruction(elem[1], basis_gates), elem[0])
                    circ_target.append(self._to_instruction(elem[1]), elem[0])
                circ.append(Barrier(self.num_qubits), circ.qubits)
            circ.metadata = {
                "xval": self.experiment_options.lengths[i % len(self.experiment_options.lengths)],
                "group": "Clifford",
                "physical_qubits": self.physical_qubits,
                "target": compute_target_bitstring(circ_target),
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

            circ.measure_all()
            circuits.append(circ)
        return circuits

    def _to_instruction(
        self,
        elem: SequenceElementType,
        basis_gates: Optional[Tuple[str, ...]] = None,
    ) -> Instruction:
        # Overriding the default RB, which uses ``self.num_qubits`` to assume Clifford size
        if isinstance(elem, Integral):
            return _clifford_1q_int_to_instruction(elem, basis_gates)
        elif isinstance(elem, Instruction):
            return elem
        return elem.to_instruction()

    def _inverse_layer(
        self, layer: List[Tuple[Tuple[int, ...], GateType]]
    ) -> List[Tuple[Tuple[int, ...], GateType]]:
        """Generates the inverse layer of a Clifford mirror RB layer by inverting the
        single-qubit Cliffords and keeping the CXs identical. See
        :class:`.RBSampler` for the format of the layer.

        Args:
            layer: The input layer.

        Returns:
            The layer that performs the inverse operation to the input layer.

        Raises:
            QiskitError: If the layer has invalid format.
        """
        inverse_layer = []
        for elem in layer:
            if len(elem[0]) == 1:
                inverse_layer.append((elem[0], inverse_1q(elem[1])))
            elif len(elem[0]) == 2:
                inverse_layer.append(elem)
            else:
                raise QiskitError("Invalid layer from sampler.")
        return tuple(inverse_layer)
