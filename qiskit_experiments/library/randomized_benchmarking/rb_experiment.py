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
from typing import Union, Iterable, Optional, List, Sequence, Tuple

import numpy as np
from numpy.random import Generator, default_rng
from numpy.random.bit_generator import BitGenerator, SeedSequence

from qiskit.circuit import QuantumCircuit, Instruction, Barrier
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend, BackendV2
from qiskit.quantum_info import Clifford
from qiskit.quantum_info.random import random_clifford
from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.framework.restless_mixin import RestlessMixin
from .clifford_utils import (
    CliffordUtils,
    _clifford_1q_int_to_instruction,
    _clifford_2q_int_to_instruction,
    _transpile_clifford_circuit,
)
from .rb_analysis import RBAnalysis

LOG = logging.getLogger(__name__)


SequenceElementType = Union[Clifford, Integral, QuantumCircuit]


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

        self._cliff_utils = None  # TODO: cleanup

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
        self._cliff_utils = CliffordUtils(
            self.num_qubits, basis_gates=self._get_basis_gates()
        )  # TODO: cleanup
        # Sample random Clifford sequences
        sequences = self._sample_sequences()
        # Convert each sequence into circuit and append the inverse to the end.
        circuits = self._sequences_to_circuits(sequences)
        # Add metadata for each circuit
        for circ, seq in zip(circuits, sequences):
            circ.metadata = {
                "experiment_type": self._type,
                "xval": len(seq),
                "group": "Clifford",
                "physical_qubits": self.physical_qubits,
            }
        return circuits

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

    def _get_basis_gates(self) -> Optional[Tuple[str]]:
        """Get sorted basis gates to use in basis transformation during circuit generation.

        Returns:
            Sorted basis gate names.
        """
        # Basis gates to use in basis transformation during circuit generation for 1Q/2Q cases
        basis_gates = self.transpile_options.get("basis_gates", None)
        if not basis_gates and self.backend:
            if isinstance(self.backend, BackendV2):
                basis_gates = self.backend.operation_names
            else:
                basis_gates = self.backend.configuration().basis_gates

        if basis_gates:
            basis_gates = tuple(sorted(basis_gates))

        return basis_gates

    def _sequences_to_circuits(
        self, sequences: List[Sequence[SequenceElementType]]
    ) -> List[QuantumCircuit]:
        """Convert a RB sequence into circuit and append the inverse to the end.

        Returns:
            A list of RB circuits.
        """
        basis_gates = self._get_basis_gates()
        # Circuit generation
        circuits = []
        for i, seq in enumerate(sequences):
            if (
                self.experiment_options.full_sampling
                or i % len(self.experiment_options.lengths) == 0
            ):
                prev_elem, prev_seq = self.__identity_clifford(), []

            circ = QuantumCircuit(self.num_qubits)
            circ.append(Barrier(self.num_qubits), circ.qubits)
            for elem in seq:
                circ.append(self._to_instruction(elem, basis_gates), circ.qubits)
                circ.append(Barrier(self.num_qubits), circ.qubits)

            # Compute inverse, compute only the difference from the previous shorter sequence
            prev_elem = self.__compose_clifford_seq(prev_elem, seq[len(prev_seq) :])
            prev_seq = seq
            inv = self.__adjoint_clifford(prev_elem)

            circ.append(self._to_instruction(inv, basis_gates), circ.qubits)
            circ.measure_all()  # includes insertion of the barrier before measurement
            circuits.append(circ)
        return circuits

    def __sample_sequence(self, length: int, rng: Generator) -> Sequence[SequenceElementType]:
        # Sample a RB sequence with the given length.
        # Return integer instead of Clifford object for 1 or 2 qubits case for speed
        if self.num_qubits == 1:
            return rng.integers(24, size=length)
        if self.num_qubits == 2:
            return rng.integers(11520, size=length)
        # Return circuit object instead of Clifford object for 3 or more qubits case for speed
        # TODO: Revisit after terra#7269, #7483, #8585
        return [random_clifford(self.num_qubits, rng).to_circuit() for _ in range(length)]

    def _to_instruction(
        self, elem: SequenceElementType, basis_gates: Optional[Tuple[str]] = None
    ) -> Instruction:
        # Switching for speed up
        if isinstance(elem, Integral):
            if self.num_qubits == 1:
                return _clifford_1q_int_to_instruction(elem, basis_gates)
            if self.num_qubits == 2:
                return _clifford_2q_int_to_instruction(elem, basis_gates)

        return elem.to_instruction()

    def __identity_clifford(self) -> SequenceElementType:
        if self.num_qubits <= 2:
            return 0
        return Clifford(np.eye(2 * self.num_qubits))

    def __compose_clifford_seq(
        self, org: SequenceElementType, seq: Sequence[SequenceElementType]
    ) -> SequenceElementType:
        if self.num_qubits <= 2:
            new = org
            for elem in seq:
                new = self.__compose_clifford(new, elem)
            return new
        # 3 or more qubits: compose Clifford from circuits for speed
        # TODO: Revisit after terra#7269, #7483, #8585
        circ = QuantumCircuit(self.num_qubits)
        for elem in seq:
            circ.compose(elem, inplace=True)
        return org.compose(Clifford.from_circuit(circ))

    def __compose_clifford(
        self, lop: SequenceElementType, rop: SequenceElementType
    ) -> SequenceElementType:
        if self.num_qubits <= 2:
            utils = self._cliff_utils
            return utils.compose_num_with_clifford(lop, utils.create_cliff_from_num(rop))

        return lop.compose(rop)

    def __adjoint_clifford(self, op: SequenceElementType) -> SequenceElementType:
        if self.num_qubits <= 2:
            return self._cliff_utils.inverse_cliff(op)

        if isinstance(op, QuantumCircuit):
            return Clifford.from_circuit(op).adjoint()

        return op.adjoint()

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits, transpiled."""
        has_custom_transpile_option = (
            any(
                opt not in {"basis_gates", "optimization_level"}
                for opt in vars(self.transpile_options)
            )
            or self.transpile_options.get("optimization_level", 0) != 0
        )
        if self.num_qubits <= 2 and not has_custom_transpile_option:
            transpiled = [
                _transpile_clifford_circuit(circ, layout=self.physical_qubits)
                for circ in self.circuits()
            ]
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
