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
import functools
from collections import defaultdict
from numbers import Integral
from typing import Union, Iterable, Optional, List, Sequence, Tuple

import numpy as np
from numpy.random import Generator, default_rng
from numpy.random.bit_generator import BitGenerator, SeedSequence

from qiskit.circuit import QuantumCircuit, Instruction, Barrier
from qiskit.exceptions import QiskitError
from qiskit.providers import BackendV2Converter
from qiskit.providers.backend import Backend, BackendV1, BackendV2
from qiskit.pulse.instruction_schedule_map import CalibrationPublisher
from qiskit.quantum_info import Clifford
from qiskit.quantum_info.random import random_clifford
from qiskit.transpiler import CouplingMap

from qiskit_experiments.warnings import deprecate_arguments
from qiskit_experiments.framework import BaseExperiment, Options
from qiskit_experiments.framework.restless_mixin import RestlessMixin

from .clifford_utils import (
    CliffordUtils,
    compose_1q,
    compose_2q,
    inverse_1q,
    inverse_2q,
    _clifford_1q_int_to_instruction,
    _clifford_2q_int_to_instruction,
    _transpile_clifford_circuit,
)
from .rb_analysis import RBAnalysis

LOG = logging.getLogger(__name__)


SequenceElementType = Union[Clifford, Integral, QuantumCircuit]


class StandardRB(BaseExperiment, RestlessMixin):
    """An experiment to characterize the error rate of a gate set on a device.

    # section: overview

    Randomized Benchmarking (RB) is an efficient and robust method
    for estimating the average error rate of a set of quantum gate operations.
    See `Qiskit Textbook
    <https://qiskit.org/textbook/ch-quantum-hardware/randomized-benchmarking.html>`_
    for an explanation on the RB method.

    A standard RB experiment generates sequences of random Cliffords
    such that the unitary computed by the sequences is the identity.
    After running the sequences on a backend, it calculates the probabilities to get back to
    the ground state, fits an exponentially decaying curve, and estimates
    the Error Per Clifford (EPC), as described in Refs. [1, 2].

    .. note::
        In 0.5.0, the default value of ``optimization_level`` in ``transpile_options`` changed
        from ``0`` to ``1`` for RB experiments. That may result in shorter RB circuits
        hence slower decay curves than before.

    # section: analysis_ref
        :class:`RBAnalysis`

    # section: reference
        .. ref_arxiv:: 1 1009.3639
        .. ref_arxiv:: 2 1109.6887
    """

    @deprecate_arguments({"qubits": "physical_qubits"}, "0.5")
    def __init__(
        self,
        physical_qubits: Sequence[int],
        lengths: Iterable[int],
        backend: Optional[Backend] = None,
        num_samples: int = 3,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        full_sampling: Optional[bool] = False,
    ):
        """Initialize a standard randomized benchmarking experiment.

        Args:
            physical_qubits: List of physical qubits for the experiment.
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
            QiskitError: If any invalid argument is supplied.
        """
        # Initialize base experiment
        super().__init__(physical_qubits, analysis=RBAnalysis(), backend=backend)

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
            full_sampling (bool): If True all Cliffords are independently sampled for
                all lengths. If False for sample of lengths longer sequences are constructed
                by appending additional Clifford samples to shorter sequences.
        """
        options = super()._default_experiment_options()
        options.update_options(
            lengths=None,
            num_samples=None,
            seed=None,
            full_sampling=None,
        )

        return options

    @classmethod
    def _default_transpile_options(cls) -> Options:
        """Default transpiler options for transpiling RB circuits."""
        return Options(optimization_level=1)

    def _set_backend(self, backend: Backend):
        """Set the backend V2 for RB experiments since RB experiments only support BackendV2
        except for simulators. If BackendV1 is provided, it is converted to V2 and stored.
        """
        if isinstance(backend, BackendV1) and "simulator" not in backend.name():
            super()._set_backend(BackendV2Converter(backend, add_delay=True))
        else:
            super()._set_backend(backend)

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of RB circuits.

        Returns:
            A list of :class:`QuantumCircuit`.
        """
        # Sample random Clifford sequences
        sequences = self._sample_sequences()
        # Convert each sequence into circuit and append the inverse to the end.
        circuits = self._sequences_to_circuits(sequences)
        # Add metadata for each circuit
        for circ, seq in zip(circuits, sequences):
            circ.metadata = {
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

    def _get_basis_gates(self) -> Optional[Tuple[str, ...]]:
        """Get sorted basis gates to use in basis transformation during circuit generation.

        - Return None if this experiment is an RB with 3 or more qubits.
        - Return None if no basis gates are supplied via ``backend`` or ``transpile_options``.
        - Return None if all 2q-gates supported on the physical qubits of the backend are one-way
        directed (e.g. cx(0, 1) is supported but cx(1, 0) is not supported).

        In all those case when None are returned, basis transformation will be skipped in the
        circuit generation step (i.e. :meth:`circuits`) and it will be done in the successive
        transpilation step (i.e. :meth:`_transpiled_circuits`) that calls :func:`transpile`.

        Returns:
            Sorted basis gate names.
        """
        # 3 or more qubits case: Return None (skip basis transformation in circuit generation)
        if self.num_qubits > 2:
            return None

        # 1 qubit case: Return all basis gates (or None if no basis gates are supplied)
        if self.num_qubits == 1:
            basis_gates = self.transpile_options.get("basis_gates", None)
            if not basis_gates and self.backend:
                if isinstance(self.backend, BackendV2):
                    basis_gates = self.backend.operation_names
                elif isinstance(self.backend, BackendV1):
                    basis_gates = self.backend.configuration().basis_gates
            return tuple(sorted(basis_gates)) if basis_gates else None

        def is_bidirectional(coupling_map):
            if coupling_map is None:
                # None for a coupling map implies all-to-all coupling
                return True
            return len(coupling_map.reduce(self.physical_qubits).get_edges()) == 2

        # 2 qubits case: Return all basis gates except for one-way directed 2q-gates.
        # Return None if there is no bidirectional 2q-gates in basis gates.
        if self.num_qubits == 2:
            basis_gates = self.transpile_options.get("basis_gates", [])
            if not basis_gates and self.backend:
                if isinstance(self.backend, BackendV2) and self.backend.target:
                    has_bidirectional_2q_gates = False
                    for op_name in self.backend.target:
                        if self.backend.target.operation_from_name(op_name).num_qubits == 2:
                            if is_bidirectional(self.backend.target.build_coupling_map(op_name)):
                                has_bidirectional_2q_gates = True
                            else:
                                continue
                        basis_gates.append(op_name)
                    if not has_bidirectional_2q_gates:
                        basis_gates = None
                elif isinstance(self.backend, BackendV1):
                    cmap = self.backend.configuration().coupling_map
                    if cmap is None or is_bidirectional(CouplingMap(cmap)):
                        basis_gates = self.backend.configuration().basis_gates
            return tuple(sorted(basis_gates)) if basis_gates else None

        return None

    def _sequences_to_circuits(
        self, sequences: List[Sequence[SequenceElementType]]
    ) -> List[QuantumCircuit]:
        """Convert an RB sequence into circuit and append the inverse to the end.

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
        # Sample an RB sequence with the given length.
        # Return integer instead of Clifford object for 1 or 2 qubits case for speed
        if self.num_qubits == 1:
            return rng.integers(CliffordUtils.NUM_CLIFFORD_1_QUBIT, size=length)
        if self.num_qubits == 2:
            return rng.integers(CliffordUtils.NUM_CLIFFORD_2_QUBIT, size=length)
        # Return circuit object instead of Clifford object for 3 or more qubits case for speed
        return [random_clifford(self.num_qubits, rng).to_circuit() for _ in range(length)]

    def _to_instruction(
        self, elem: SequenceElementType, basis_gates: Optional[Tuple[str, ...]] = None
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
        self, base_elem: SequenceElementType, elements: Sequence[SequenceElementType]
    ) -> SequenceElementType:
        if self.num_qubits <= 2:
            return functools.reduce(
                compose_1q if self.num_qubits == 1 else compose_2q, elements, base_elem
            )
        # 3 or more qubits: compose Clifford from circuits for speed
        circ = QuantumCircuit(self.num_qubits)
        for elem in elements:
            circ.compose(elem, inplace=True)
        return base_elem.compose(Clifford.from_circuit(circ))

    def __adjoint_clifford(self, op: SequenceElementType) -> SequenceElementType:
        if self.num_qubits == 1:
            return inverse_1q(op)
        if self.num_qubits == 2:
            return inverse_2q(op)
        if isinstance(op, QuantumCircuit):
            return Clifford.from_circuit(op).adjoint()
        return op.adjoint()

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits, transpiled."""
        has_custom_transpile_option = (
            not set(vars(self.transpile_options)).issubset({"basis_gates", "optimization_level"})
            or self.transpile_options.get("optimization_level", 1) != 1
        )
        has_no_undirected_2q_basis = self._get_basis_gates() is None
        if self.num_qubits > 2 or has_custom_transpile_option or has_no_undirected_2q_basis:
            transpiled = super()._transpiled_circuits()
        else:
            transpiled = [
                _transpile_clifford_circuit(circ, physical_qubits=self.physical_qubits)
                for circ in self.circuits()
            ]
            # Set custom calibrations provided in backend
            if isinstance(self.backend, BackendV2):
                qargs_patterns = [self.physical_qubits]  # for self.num_qubits == 1
                if self.num_qubits == 2:
                    qargs_patterns = [
                        (self.physical_qubits[0],),
                        (self.physical_qubits[1],),
                        self.physical_qubits,
                        (self.physical_qubits[1], self.physical_qubits[0]),
                    ]

                instructions = []  # (op_name, qargs) for each element where qargs means qubit tuple
                for qargs in qargs_patterns:
                    for op_name in self.backend.target.operation_names_for_qargs(qargs):
                        instructions.append((op_name, qargs))

                common_calibrations = defaultdict(dict)
                for op_name, qargs in instructions:
                    inst_prop = self.backend.target[op_name].get(qargs, None)
                    if inst_prop is None:
                        continue
                    try:
                        schedule = inst_prop.calibration
                    except AttributeError:
                        # TODO remove after qiskit-terra/#9681 is in stable release.
                        schedule = None
                    if schedule is None:
                        continue
                    publisher = schedule.metadata.get("publisher", CalibrationPublisher.QISKIT)
                    if publisher != CalibrationPublisher.BACKEND_PROVIDER:
                        common_calibrations[op_name][(qargs, tuple())] = schedule

                for circ in transpiled:
                    # This logic is inefficient in terms of payload size and backend compilation
                    # because this binds every custom pulse to a circuit regardless of
                    # its existence. It works but redundant calibration must be removed -- NK.
                    circ.calibrations = common_calibrations

        if self.analysis.options.get("gate_error_ratio", None) is None:
            # Gate errors are not computed, then counting ops is not necessary.
            return transpiled

        # Compute average basis gate numbers per Clifford operation
        # This is probably main source of performance regression.
        # This should be integrated into transpile pass in future.
        qubit_indices = {bit: index for index, bit in enumerate(transpiled[0].qubits)}
        for circ in transpiled:
            count_ops_result = defaultdict(int)
            # This is physical circuits, i.e. qargs is physical index
            for inst, qargs, _ in circ.data:
                if inst.name in ("measure", "reset", "delay", "barrier", "snapshot"):
                    continue
                qinds = [qubit_indices[q] for q in qargs]
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
