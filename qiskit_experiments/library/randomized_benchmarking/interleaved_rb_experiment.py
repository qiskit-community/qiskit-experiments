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
Interleaved RB Experiment class.
"""
from typing import Union, Iterable, Optional, List, Sequence, Tuple

from numpy.random import Generator
from numpy.random.bit_generator import BitGenerator, SeedSequence

from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit.quantum_info import Clifford
from .interleaved_rb_analysis import InterleavedRBAnalysis
from .rb_experiment import StandardRB, SequenceElementType


class InterleavedRB(StandardRB):
    """Interleaved randomized benchmarking experiment.

    # section: overview
        Interleaved Randomized Benchmarking (RB) is a method
        to estimate the average error-rate of a certain quantum gate.

        An interleaved RB experiment generates a standard RB sequences of random Cliffords
        and another sequence with the interleaved given gate.
        After running the two sequences on a backend, it calculates the probabilities to get back to
        the ground state, fits the two exponentially decaying curves, and estimates
        the interleaved gate error. See Ref. [1] for details.

    # section: analysis_ref
        :py:class:`InterleavedRBAnalysis`

    # section: reference
        .. ref_arxiv:: 1 1203.4550

    """

    def __init__(
        self,
        interleaved_element: Union[QuantumCircuit, Instruction, Clifford],
        qubits: Sequence[int],
        lengths: Iterable[int],
        backend: Optional[Backend] = None,
        num_samples: int = 3,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        full_sampling: bool = False,
    ):
        """Initialize an interleaved randomized benchmarking experiment.

        Args:
            interleaved_element: The element to interleave,
                    given either as a group element or as an instruction/circuit
            qubits: list of physical qubits for the experiment.
            lengths: A list of RB sequences lengths.
            backend: The backend to run the experiment on.
            num_samples: Number of samples to generate for each
                         sequence length
            seed: Optional, seed used to initialize ``numpy.random.default_rng``.
                  when generating circuits. The ``default_rng`` will be initialized
                  with this seed value everytime :meth:`circuits` is called.
            full_sampling: If True all Cliffords are independently sampled for
                           all lengths. If False for sample of lengths longer
                           sequences are constructed by appending additional
                           Clifford samples to shorter sequences.

        Raises:
            QiskitError: the interleaved_element is not convertible to Clifford object.
        """
        try:
            self._interleaved_elem = Clifford(interleaved_element)
        except QiskitError as err:
            raise QiskitError(
                f"Interleaved element {interleaved_element.name} could not be converted to Clifford."
            ) from err
        self._interleaved_op = interleaved_element
        super().__init__(
            qubits,
            lengths,
            backend=backend,
            num_samples=num_samples,
            seed=seed,
            full_sampling=full_sampling,
        )
        self.analysis = InterleavedRBAnalysis()
        self.analysis.set_options(outcome="0" * self.num_qubits)

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of RB circuits.

        Returns:
            A list of :class:`QuantumCircuit`.
        """
        # Convert interleaved element to operation and store the operation for speed
        basis_gates = self._basis_gates
        if basis_gates:
            interleaved_circ = None
            if isinstance(self._interleaved_op, QuantumCircuit):
                interleaved_circ = self._interleaved_op
            elif isinstance(self._interleaved_op, Clifford):
                interleaved_circ = self._interleaved_op.to_circuit()
            else:  # Instruction
                if self._interleaved_op.name not in basis_gates:
                    interleaved_circ = QuantumCircuit(self.num_qubits)
                    interleaved_circ.append(self._interleaved_op)
            if interleaved_circ and any(
                i.operation.name not in basis_gates for i in interleaved_circ
            ):
                interleaved_circ = transpile(
                    interleaved_circ, basis_gates=list(basis_gates), optimization_level=1
                )
                self._interleaved_op = interleaved_circ.to_instruction()
        else:
            if not isinstance(self._interleaved_op, Instruction):
                self._interleaved_op = self._interleaved_op.to_instruction()

        # Build circuits of reference sequences
        reference_sequences = self._sample_sequences()
        reference_circuits = self._sequences_to_circuits(reference_sequences)
        for circ, seq in zip(reference_circuits, reference_sequences):
            circ.metadata = {
                "experiment_type": self._type,
                "xval": len(seq),
                "group": "Clifford",
                "physical_qubits": self.physical_qubits,
                "interleaved": False,
            }
        # Build circuits of interleaved sequences
        interleaved_sequences = []
        for seq in reference_sequences:
            new_seq = []
            for elem in seq:
                new_seq.append(elem)
                new_seq.append(self._interleaved_elem)
            interleaved_sequences.append(new_seq)
        interleaved_circuits = self._sequences_to_circuits(interleaved_sequences)
        for circ, seq in zip(interleaved_circuits, reference_sequences):
            circ.metadata = {
                "experiment_type": self._type,
                "xval": len(seq),  # set length of the reference sequence
                "group": "Clifford",
                "physical_qubits": self.physical_qubits,
                "interleaved": True,
            }
        return reference_circuits + interleaved_circuits

    def _to_instruction(
        self, elem: SequenceElementType, basis_gates: Optional[Tuple[str]] = None
    ) -> Instruction:
        if elem is self._interleaved_elem:
            return self._interleaved_op

        return super()._to_instruction(elem, basis_gates)
