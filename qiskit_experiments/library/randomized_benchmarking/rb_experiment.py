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
Standard RB Experiment class.
"""

import logging
from collections import defaultdict
from numbers import Integral
from typing import Union, Iterable, Optional, List, Sequence
import numpy as np
from numpy.random import Generator, default_rng
from numpy.random.bit_generator import BitGenerator, SeedSequence

from qiskit import QuantumCircuit, ClassicalRegister, QiskitError
from qiskit.circuit import Clbit
from qiskit.circuit import Instruction
from qiskit.providers.backend import Backend
from qiskit.compiler import transpile
from qiskit.quantum_info import Clifford, random_clifford

from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.framework.restless_mixin import RestlessMixin
from .clifford_utils import (
    CliffordUtils,
    _clifford_1q_int_to_instruction,
    _clifford_2q_int_to_instruction,
)
from .rb_analysis import RBAnalysis

LOG = logging.getLogger(__name__)

SequenceElementType = Union[Clifford, Integral]

class StandardRB(BaseExperiment, RestlessMixin):
    """Standard randomized benchmarking experiment.

    # section: overview
        Randomized Benchmarking (RB) is an efficient and robust method
        for estimating the average error-rate of a set of quantum gate operations.
        See `Qiskit Textbook
        <https://qiskit.org/textbook/ch-quantum-hardware/randomized-benchmarking.html>`_
        for an explanation on the RB method.

        A standard RB experiment generates sequences of random Cliffords
        such that the unitary computed by the sequences is the identity.
        After running the sequences on a backend, it calculates the probabilities to get back to
        the ground state, fits an exponentially decaying curve, and estimates
        the Error Per Clifford (EPC), as described in Refs. [1, 2].

    # section: analysis_ref
        :py:class:`RBAnalysis`

    # section: reference
        .. ref_arxiv:: 1 1009.3639
        .. ref_arxiv:: 2 1109.6887

    """

    default_basis_gates = {"rz", "sx", "cx"}
    _clifford_utils = None

    def __init__(
        self,
        qubits: Sequence[int],
        lengths: Iterable[int],
        backend: Optional[Backend] = None,
        num_samples: int = 3,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        full_sampling: Optional[bool] = False,
    ):
        """Initialize a standard randomized benchmarking experiment.

        Args:
            qubits: list of physical qubits for the experiment.
            lengths: A list of RB sequences lengths.
            backend: The backend to run the experiment on.
            num_samples: Number of samples to generate for each sequence length.
            seed: Optional, seed used to initialize ``numpy.random.default_rng``.
                  when generating circuits. The ``default_rng`` will be initialized
                  with this seed value everytime :meth:`circuits` is called.
            full_sampling: If True all Cliffords are independently sampled for all lengths.
                           If False for sample of lengths longer sequences are constructed
                           by appending additional samples to shorter sequences.
                           The default is False.

        Raises:
            QiskitError: if any invalid argument is supplied.
        """
        # Initialize base experiment
        super().__init__(qubits, analysis=RBAnalysis(), backend=backend)

        # Verify parameters
        if any(length <= 0 for length in lengths):
            raise QiskitError(
                f"The lengths list {lengths} should only contain " "positive elements."
            )
        if len(set(lengths)) != len(lengths):
            raise QiskitError(
                f"The lengths list {lengths} should not contain " "duplicate elements."
            )
        if num_samples <= 0:
            raise QiskitError(f"The number of samples {num_samples} should " "be positive.")

        # Set configurable options
        self.set_experiment_options(
            lengths=sorted(lengths), num_samples=num_samples, seed=seed, full_sampling=full_sampling
        )
        self.analysis.set_options(outcome="0" * self.num_qubits)

        # Set fixed options
        self._full_sampling = full_sampling

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
            lengths=None,
            num_samples=None,
            seed=None,
            full_sampling=None,
        )

        return options

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of RB circuits.

        Returns:
            A list of :class:`QuantumCircuit`.

        """
        self._set_basis_gates()
        self._initialize_clifford_utils()
        rng = default_rng(seed=self.experiment_options.seed)
        circuits = []

        if self.num_qubits in [1, 2]:
            for _ in range(self.experiment_options.num_samples):
                rb_circuits = self._build_rb_circuits(self.experiment_options.lengths, rng)
                circuits += rb_circuits
        else:
            for _ in range(self.experiment_options.num_samples):
                circuits += self._sample_circuits()

        return circuits

    # The following methods are used for RB with more than 2 qubits
    def _sample_circuits(self):
        sequences = self._sample_sequences()
        return self._sequences_to_circuits(sequences)

    def _sample_sequences(self) -> List[Sequence[SequenceElementType]]:
        """Sample RB sequences

        Returns:
            A list of RB sequences.
        """
        rng = default_rng(seed=self.experiment_options.seed)
        sequences = []
        if self.experiment_options.full_sampling:
            for _ in range(self.experiment_options.num_samples):
                for length in self.experiment_options.lengths:
                    sequences.append(self.__sample_sequence(length, rng))
        else:
            for _ in range(self.experiment_options.num_samples):
                longest_seq = self.__sample_sequence(max(self.experiment_options.lengths), rng)
                for length in self.experiment_options.lengths:
                    sequences.append(longest_seq[:length])

        return sequences

    def _sequences_to_circuits(
        self, sequences: List[Sequence[SequenceElementType]]
    ) -> List[QuantumCircuit]:
        """Convert an RB sequence into circuit and append the inverse to the end.

        Returns:
            A list of RB circuits.
        """
        circuits = []
        for i, seq in enumerate(sequences):
            if (
                self.experiment_options.full_sampling
                or i % len(self.experiment_options.lengths) == 0
            ):
                prev_elem, prev_seq = self.__identity_clifford(), []

            qubits = list(range(self.num_qubits))
            circ = QuantumCircuit(self.num_qubits)
            circ.barrier(qubits)
            for elem in seq:
                circ.append(self._to_instruction(elem), qubits)
                circ.barrier(qubits)

            # Compute inverse, compute only the difference from the previous shorter sequence
            for elem in seq[len(prev_seq) :]:
                prev_elem = self.__compose_clifford(prev_elem, elem)
            prev_seq = seq
            inv = self.__adjoint_clifford(prev_elem)

            circ.append(self._to_instruction(inv), qubits)
            circ.measure_all()  # includes insertion of the barrier before measurement
            # Add metadata
            circ.metadata = {
                "experiment_type": self._type,
                "xval": len(seq),
                "group": "Clifford",
                "physical_qubits": self.physical_qubits,
            }
            circuits.append(circ)
        return circuits

    def __sample_sequence(self, length: int, rng: Generator) -> Sequence[SequenceElementType]:
        # Sample a RB sequence with the given length.
        # Return integer instead of Clifford object for 1 or 2 qubit case for speed
        if self.num_qubits == 1:
            return rng.integers(24, size=length)
        if self.num_qubits == 2:
            return rng.integers(11520, size=length)

        return [random_clifford(self.num_qubits, rng) for _ in range(length)]

    def _to_instruction(self, elem: SequenceElementType) -> Instruction:
        # TODO: basis transformation in 1Q (and 2Q) cases for speed
        # Switching for speed up
        if isinstance(elem, Integral):
            if self.num_qubits == 1:
                return _clifford_1q_int_to_instruction(elem)
            if self.num_qubits == 2:
                return _clifford_2q_int_to_instruction(elem)
        return elem.to_instruction()

    def __identity_clifford(self) -> SequenceElementType:
        if self.num_qubits <= 2:
            return 0
        return Clifford(np.eye(2 * self.num_qubits))

    def __compose_clifford(
        self, lop: SequenceElementType, rop: SequenceElementType
    ) -> SequenceElementType:
        # TODO: Speed up 1Q (and 2Q) cases using integer clifford composition
        # Integer clifford composition has not yet supported
        if self.num_qubits == 1:
            if isinstance(lop, Integral):
                lop = CliffordUtils.clifford_1_qubit(lop)
            if isinstance(rop, Integral):
                rop = CliffordUtils.clifford_1_qubit(rop)
        if self.num_qubits == 2:
            if isinstance(lop, Integral):
                lop = CliffordUtils.clifford_2_qubit(lop)
            if isinstance(rop, Integral):
                rop = CliffordUtils.clifford_2_qubit(rop)
        return lop.compose(rop)

    def __adjoint_clifford(self, op: SequenceElementType) -> SequenceElementType:
        # TODO: Speed up 1Q and 2Q cases using integer clifford inversion
        # Integer clifford inversion has not yet supported
        if isinstance(op, Integral):
            if self.num_qubits == 1:
                return CliffordUtils.clifford_1_qubit(op).adjoint()
            if self.num_qubits == 2:
                return CliffordUtils.clifford_2_qubit(op).adjoint()
        return op.adjoint()

    # The followng methods are used for RB with 1 or 2 qubits
    def _build_rb_circuits(self, lengths: List[int], rng: Generator) -> List[QuantumCircuit]:
        """
        build_rb_circuits
        Args:
                lengths: A list of RB sequence lengths. We create random circuits
                         where the number of cliffords in each is defined in 'lengths'.
                rng: Generator object for random number generation.
                     If None, default_rng will be used.

        Returns:
                The transpiled RB circuits.

        Additional information:
            To create the RB circuit, we use a mapping between Cliffords and integers
            defined in the file clifford_data.py. The operations compose and inverse are much faster
            when performed on the integers rather than on the Cliffords themselves.
        """
        if self._full_sampling:
            return self._build_rb_circuits_full_sampling(lengths, rng)
        max_qubit = max(self.physical_qubits) + 1
        all_rb_circuits = []

        # When full_sampling==False, each circuit is the prefix of the next circuit (without the
        # inverse Clifford at the end of the circuit. The variable 'circ' will contain
        # the growing circuit.
        # When each circuit reaches its length, we copy it to rb_circ, append the inverse,
        # and add it to the list of circuits.
        n = self.num_qubits
        qubits = list(range(n))
        clbits = list(range(n))
        circ = QuantumCircuit(max_qubit, n)
        circ.barrier(qubits)
        # We transpile the empty circuit to match the backend qubits
        circ = transpile(
            circuits=circ,
            optimization_level=1,
            basis_gates=self.transpile_options.basis_gates,
            backend=self._backend,
        )

        # composed_cliff_num is the number representing the composition of all the Cliffords up to now
        composed_cliff_num = 0  # 0 is the Clifford that is Id
        prev_length = 0

        for length in lengths:
            for i in range(prev_length, length):
                circ, _, composed_cliff_num = self._add_random_cliff_to_circ(
                    circ, composed_cliff_num, qubits, rng
                )

                if i == length - 1:
                    rb_circ = circ.copy()  # circ is used as the prefix of the next circuit
                    rb_circ = self._add_inverse_to_circ(rb_circ, composed_cliff_num, qubits, clbits)

                    rb_circ.metadata = {
                        "experiment_type": "rb",
                        "xval": length,
                        "group": "Clifford",
                        "physical_qubits": self.physical_qubits,
                    }
                    all_rb_circuits.append(rb_circ)
                prev_length = i + 1
        return all_rb_circuits

    def _build_rb_circuits_full_sampling(
        self, lengths: List[int], rng: Generator
    ) -> List[QuantumCircuit]:
        """
        _build_rb_circuits_full_sampling
        Args:
                lengths: A list of RB sequence lengths. We create random circuits
                    where the number of cliffords in each is defined in 'lengths'.
                rng: Generator object for random number generation.
                    If None, default_rng will be used.

        Returns:
                The transpiled RB circuits.

        Additional information:
            This is similar to _build_rb_circuits for the case of full_sampling.
        """
        all_rb_circuits = []
        n = self.num_qubits
        qubits = list(range(n))
        clbits = list(range(n))
        max_qubit = max(self.physical_qubits) + 1
        for length in lengths:
            # We define the circuit size here, for the layout that will
            # be created later
            rb_circ = QuantumCircuit(max_qubit, n)
            rb_circ.barrier(qubits)
            # We transpile the empty circuit to match the backend qubits
            rb_circ = transpile(
                circuits=rb_circ,
                optimization_level=1,
                basis_gates=self.transpile_options.basis_gates,
                backend=self._backend,
            )

            # composed_cliff_num is the number representing the composition of
            # all the Cliffords up to now
            composed_cliff_num = 0

            # For full_sampling, we create each circuit independently.
            for _ in range(length):
                # choose random clifford
                rb_circ, _, composed_cliff_num = self._add_random_cliff_to_circ(
                    rb_circ, composed_cliff_num, qubits, rng
                )

            rb_circ = self._add_inverse_to_circ(rb_circ, composed_cliff_num, qubits, clbits)

            rb_circ.metadata = {
                "experiment_type": "rb",
                "xval": length,
                "group": "Clifford",
                "physical_qubits": self.physical_qubits,
                "interleaved": False,
            }

            all_rb_circuits.append(rb_circ)
        return all_rb_circuits

    def _add_random_cliff_to_circ(self, circ, composed_cliff_num, qubits, rng):
        next_circ = StandardRB._clifford_utils.create_random_clifford(rng)
        circ, composed_cliff_num = self._add_cliff_to_circ(
            circ, next_circ, composed_cliff_num, qubits
        )
        return circ, next_circ, composed_cliff_num

    def _add_cliff_to_circ(
        self,
        circ: QuantumCircuit,
        next_circ: QuantumCircuit,
        composed_cliff_num: int,
        qubits: List[int],
    ):
        """Append a Clifford to the end of a circuit. Return both the updated circuit and the updated
        number representing the circuit"""
        circ.compose(next_circ, inplace=True)
        composed_cliff_num = StandardRB._clifford_utils.compose_num_with_clifford(
            composed_num=composed_cliff_num,
            qc=next_circ,
        )
        circ.barrier(qubits)
        return circ, composed_cliff_num

    def _add_inverse_to_circ(self, rb_circ, composed_num, qubits, clbits):
        """Append the inverse of a circuit to the end of the circuit"""
        inverse_cliff = StandardRB._clifford_utils.inverse_cliff(composed_num)
        rb_circ.compose(inverse_cliff, inplace=True)
        rb_circ.measure(qubits, clbits)
        return rb_circ

    # This method does a quick layout to avoid calling 'transpile()' which is
    # very costly in performance
    # We simply copy the circuit to a new circuit where we define the mapping
    # of the qubit to the single physical qubit that was requested by the user
    # This is a hack, and would be better if transpile() implemented it.
    # Something similar is done in ParallelExperiment._combined_circuits
    def _layout_for_rb(self):
        transpiled = []
        qargs_map = (
            {0: self.physical_qubits[0]}
            if self.num_qubits == 1
            else {0: self.physical_qubits[0], 1: self.physical_qubits[1]}
        )
        for circ in self.circuits():
            new_circ = QuantumCircuit(
                *circ.qregs,
                name=circ.name,
                global_phase=circ.global_phase,
                metadata=circ.metadata.copy(),
            )
            clbits = circ.num_clbits
            if clbits:
                creg = ClassicalRegister(clbits)
                new_cargs = [Clbit(creg, i) for i in range(clbits)]
                new_circ.add_register(creg)

            for inst, qargs, cargs in circ.data:
                mapped_cargs = [new_cargs[circ.find_bit(clbit).index] for clbit in cargs]
                mapped_qargs = [circ.qubits[qargs_map[circ.find_bit(i).index]] for i in qargs]
                new_circ.data.append((inst, mapped_qargs, mapped_cargs))
                # Add the calibrations
                for gate, cals in circ.calibrations.items():
                    for key, sched in cals.items():
                        new_circ.add_calibration(gate, qubits=key[0], schedule=sched, params=key[1])

            transpiled.append(new_circ)
        return transpiled

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits, transpiled."""
        if self.num_qubits in [1, 2]:
            transpiled = self._layout_for_rb()
        else:
            transpiled = super()._transpiled_circuits()
        if self.analysis.options.get("gate_error_ratio", None) is None:

            # Gate errors are not computed, then counting ops is not necessary.
            return transpiled

        # Compute average basis gate numbers per Clifford operation
        # This is probably main source of performance regression.
        # This should be integrated into transpile pass in future.
        for circ in transpiled:
            count_ops_result = defaultdict(int)
            # This is physical circuits, i.e. qargs is physical index
            for inst, qargs, _ in circ.data:
                if inst.name in ("measure", "reset", "delay", "barrier", "snapshot"):
                    continue
                qinds = [circ.find_bit(q).index for q in qargs]
                if not set(self.physical_qubits).issuperset(qinds):
                    continue
                # Not aware of multi-qubit gate direction
                formatted_key = tuple(sorted(qinds)), inst.name
                count_ops_result[formatted_key] += 1
            circ.metadata["count_ops"] = tuple(count_ops_result.items())
        return transpiled

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)

        return metadata

    def _initialize_clifford_utils(self):
        if StandardRB._clifford_utils is None or not (
            StandardRB._clifford_utils.num_qubits == self.num_qubits
            and StandardRB._clifford_utils.basis_gates == self.transpile_options.basis_gates
            and StandardRB._clifford_utils._backend == self._backend
        ):
            StandardRB._clifford_utils = CliffordUtils(
                self.num_qubits, self.transpile_options.basis_gates, backend=self._backend
            )

    def _set_basis_gates(self):
        if not hasattr(self.transpile_options, "basis_gates"):
            if not self.backend is None and self.backend.configuration().basis_gates:
                self.set_transpile_options(basis_gates=self.backend.configuration().basis_gates)
            else:
                basis_gates_option = {"basis_gates": StandardRB.default_basis_gates}
                self.transpile_options.update_options(**basis_gates_option)
