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
from typing import Union, Iterable, Optional, List, Sequence, Tuple
from numbers import Integral
import itertools
import numpy as np
from numpy.random import Generator, BitGenerator, SeedSequence

from qiskit.circuit import QuantumCircuit, Instruction, Barrier
from qiskit.quantum_info.operators import Pauli
from qiskit.providers.backend import Backend
from qiskit.providers.options import Options
from qiskit.exceptions import QiskitError
from qiskit.circuit.library import CXGate, CYGate, CZGate, ECRGate, SwapGate, iSwapGate

from .standard_rb import StandardRB, SequenceElementType
from .mirror_rb_analysis import MirrorRBAnalysis
from .clifford_utils import (
    compute_target_bitstring,
    inverse_1q,
    _clifford_1q_int_to_instruction,
)
from .sampling_utils import (
    EdgeGrabSampler,
    SingleQubitSampler,
    GateInstruction,
    GateDistribution,
    GenericClifford,
    GenericPauli,
)

# two qubit gates that are their own inverse
_self_adjoint_gates = [CXGate, CYGate, CZGate, ECRGate, SwapGate, iSwapGate]


class MirrorRB(StandardRB):
    """An experiment to measure gate infidelity using mirrored circuit
    layers sampled from a defined distribution.

    # section: overview
        Mirror randomized benchmarking (mirror RB) estimates the average error rate of
        quantum gates using gate layers sampled from a distribution that are then
        inverted in the second half of the circuit.

        The default mirror RB experiment generates circuits of layers of Cliffords,
        consisting of single-qubit Cliffords and a two-qubit gate such as CX,
        interleaved with layers of Pauli gates and capped at the start and end by a
        layer of single-qubit Cliffords. The second half of the Clifford layers are the
        inverses of the first half of Clifford layers. After running the circuits on a
        backend, various quantities (success probability, adjusted success probability,
        and effective polarization) are computed and used to fit an exponential decay
        curve and calculate the EPC (error per Clifford, also referred to as the average
        gate infidelity) and entanglement infidelity (see references for more info).

    # section: analysis_ref
        :class:`MirrorRBAnalysis`

    # section: manual
        :doc:`/manuals/verification/mirror_rb`

    # section: reference
        .. ref_arxiv:: 1 2112.09853
        .. ref_arxiv:: 2 2008.11294
        .. ref_arxiv:: 3 2204.07568

    """

    sampler_map = {"edge_grab": EdgeGrabSampler, "single_qubit": SingleQubitSampler}

    def __init__(
        self,
        physical_qubits: Sequence[int],
        lengths: Iterable[int],
        start_end_clifford: bool = True,
        pauli_randomize: bool = True,
        sampling_algorithm: str = "edge_grab",
        two_qubit_gate_density: float = 0.2,
        two_qubit_gate: Instruction = CXGate(),
        sampler_opts: dict = {},
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
            sampling_algorithm: The sampling algorithm to use for generating
                circuit layers. Defaults to "edge_grab" which uses :class:`.EdgeGrabSampler`.
            start_end_clifford: If True, begin the circuit with uniformly random 1-qubit
                Cliffords and end the circuit with their inverses.
            pauli_randomize: If True, surround each sampled circuit layer with layers of
                uniformly random 1-qubit Paulis.
            two_qubit_gate_density: Expected proportion of qubit sites with two-qubit
                gates over all circuit layers (not counting optional layers at the start
                and end). Only has effect if the default sampler
                :class:`.EdgeGrabSampler` is used.
            two_qubit_gate: The two-qubit gate to use. Defaults to
                :class:`~qiskit.circuit.library.CXGate`. Only has effect if the
                default sampler :class:`.EdgeGrabSampler` is used.
            sampler_opts: Dictionary of keyword arguments to pass to the sampler.
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

        self.set_experiment_options(
            sampling_algorithm=sampling_algorithm,
            sampler_opts=sampler_opts,
            start_end_clifford=start_end_clifford,
            pauli_randomize=pauli_randomize,
            two_qubit_gate=two_qubit_gate,
            two_qubit_gate_density=two_qubit_gate_density,
            inverting_pauli_layer=inverting_pauli_layer,
        )

        self._distribution = self.sampler_map.get(sampling_algorithm)(seed=seed, **sampler_opts)
        self.analysis = MirrorRBAnalysis()

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default mirror RB experiment options.

        Experiment Options:
            sampling_algorithm (str): Name of sampling algorithm to use.
            start_end_clifford (bool): Whether to begin the circuit with uniformly random 1-qubit
                Cliffords and end the circuit with their inverses.
            pauli_randomize (bool): Whether to surround each inner Clifford layer with
                layers of uniformly random 1-qubit Paulis.
            inverting_pauli_layer (bool): Whether to append a layer of Pauli gates at the
                end of the circuit to set all qubits to 0.
            sampler_opts (dict): The keyword arguments to pass to the sampler.
            two_qubit_gate_density (float): Expected proportion of qubit sites with two-qubit
                gates over all circuit layers (not counting optional layers at the start
                and end). Only has effect if the default sampler
                :class:`.EdgeGrabSampler` is used.
            two_qubit_gate (:class:`.Instruction`): The two-qubit gate to use. Defaults to
                :class:`~qiskit.circuit.library.CXGate`. Only has effect if the
                default sampler :class:`.EdgeGrabSampler` is used.
            num_samples (int): Number of samples to generate for each sequence length.
        """
        options = super()._default_experiment_options()
        options.update_options(
            sampling_algorithm="edge_grab",
            start_end_clifford=True,
            pauli_randomize=True,
            two_qubit_gate_density=0.2,
            two_qubit_gate=CXGate(),
            sampler_opts={},
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

    def _set_distribution_options(self):
        """Set the coupling map and gate distribution of the sampler
        based on experiment options. This method is currently implemented
        for the default "edge_grab" sampler."""

        if self.experiment_options.sampling_algorithm != "edge_grab":
            raise QiskitError(
                "Unsupported sampling algorithm provided. You must implement"
                "a custom `_set_distribution_options` method."
            )

        self._distribution.seed = self.experiment_options.seed

        # Coupling map is full connectivity by default. If backend has a coupling map,
        # get backend coupling map and create coupling map for physical qubits converted
        # to qubits 0, 1...n
        if self.backend and self._backend_data.coupling_map:
            coupling_map = self._backend_data.coupling_map
        else:
            coupling_map = list(itertools.permutations(range(max(self.physical_qubits) + 1), 2))
        qmap = {self.physical_qubits[i]: i for i in range(len(self.physical_qubits))}
        experiment_coupling_map = []

        for edge in coupling_map:
            if edge[0] in self.physical_qubits and edge[1] in self.physical_qubits:
                experiment_coupling_map.append((qmap[edge[0]], qmap[edge[1]]))

        self._distribution.coupling_map = experiment_coupling_map

        # Adjust the density based on whether the pauli layers are in
        if self.experiment_options.pauli_randomize:
            adjusted_2q_density = self.experiment_options.two_qubit_gate_density * 2
        else:
            adjusted_2q_density = self.experiment_options.two_qubit_gate_density

        if adjusted_2q_density > 1:
            warnings.warn("Two-qubit gate density is too high, capping at 1.")
            adjusted_2q_density = 1

        if self.experiment_options.pauli_randomize:
            self._distribution.gate_distribution = [
                GateDistribution(
                    prob=adjusted_2q_density, op=self.experiment_options.two_qubit_gate
                ),
                GateDistribution(prob=1 - adjusted_2q_density, op=GenericClifford(1)),
            ]

        else:
            self._distribution.gate_distribution = [
                GateDistribution(
                    prob=adjusted_2q_density, op=self.experiment_options.two_qubit_gate
                ),
                GateDistribution(prob=1 - adjusted_2q_density, op=GenericClifford(1)),
            ]

    def _sample_sequences(self) -> List[Sequence[SequenceElementType]]:
        """Sample layers of mirror RB using the provided distribution and user options.

        First, layers are sampled using the distribution, then Pauli-dressed if
        ``pauli_randomize`` is ``True``. The inverse of the resulting circuit is
        appended to the end. If ``start_end_clifford`` is ``True``, then cliffords are added
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

        self._set_distribution_options()

        # Sequence of lengths to sample for
        if not self.experiment_options.full_sampling:
            seqlens = (max(self.experiment_options.lengths),)
        else:
            seqlens = self.experiment_options.lengths

        if self.experiment_options.pauli_randomize:
            pauli_sampler = SingleQubitSampler(seed=self.experiment_options.seed)
            pauli_sampler.gate_distribution = [GateDistribution(prob=1, op=GenericPauli(1))]

        if self.experiment_options.start_end_clifford:
            clifford_sampler = SingleQubitSampler(seed=self.experiment_options.seed)
            clifford_sampler.gate_distribution = [GateDistribution(prob=1, op=GenericClifford(1))]

        sequences = []

        for _ in range(self.experiment_options.num_samples):
            for seqlen in seqlens:
                seq = []

                # Sample the first half of the mirror layers
                layers = list(
                    self._distribution(
                        qubits=range(self.num_qubits),
                        length=seqlen // 2,
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
                    pauli_layers = pauli_sampler(range(self.num_qubits), length=seqlen + 1)
                    seq = list(itertools.chain(*zip(pauli_layers[:-1], seq)))
                    seq.append(pauli_layers[-1])
                    if not self.experiment_options.full_sampling:
                        build_seq_lengths = [length * 2 + 1 for length in build_seq_lengths]

                # Add start and end cliffords if set by user
                if self.experiment_options.start_end_clifford:
                    cseq = []
                    clifford_layers = clifford_sampler(range(self.num_qubits), length=1)
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
                    circ.append(self._to_instruction(elem.op, basis_gates), elem.qargs)
                    circ_target.append(self._to_instruction(elem.op), elem.qargs)
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
        """Convert the sampled object to an instruction."""
        if isinstance(elem, Integral):
            return _clifford_1q_int_to_instruction(elem, basis_gates)
        elif isinstance(elem, Instruction):
            return elem
        elif getattr(elem, "to_instruction", None):
            return elem.to_instruction()
        else:
            return elem()

    def _inverse_layer(
        self, layer: List[Tuple[GateInstruction, ...]]
    ) -> List[Tuple[GateInstruction, ...]]:
        """Generates the inverse layer of a Clifford mirror RB layer by inverting the
        single-qubit Cliffords and keeping the two-qubit gate identical. See
        :class:`.BaseSampler` for the format of the layer.

        Args:
            layer: The input layer.

        Returns:
            The layer that performs the inverse operation to the input layer.

        Raises:
            QiskitError: If the layer has invalid format.
        """
        inverse_layer = []
        for elem in layer:
            if len(elem.qargs) == 1 and np.issubdtype(type(elem.op), int):
                inverse_layer.append(GateInstruction(elem.qargs, inverse_1q(elem.op)))
            elif len(elem.qargs) == 2 and elem.op in _self_adjoint_gates:
                inverse_layer.append(elem)
            else:
                try:
                    inverse_layer.append(GateInstruction(elem.qargs, elem.op.inverse()))
                except TypeError as exc:
                    raise QiskitError("Invalid layer supplied.") from exc
        return tuple(inverse_layer)
