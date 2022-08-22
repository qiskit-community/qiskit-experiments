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
from typing import Union, Iterable, Optional, List, Sequence

from numpy.random import Generator
from numpy.random.bit_generator import BitGenerator, SeedSequence

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.quantum_info import Clifford
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend

from .rb_experiment import StandardRB, GroupOperatorType
from .interleaved_rb_analysis import InterleavedRBAnalysis


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
        """
        # TODO: Can we mitigate this condition (interleaved_element must be a Clifford group element)?
        try:
            Clifford(interleaved_element)
        except QiskitError as err:
            raise QiskitError(
                f"Interleaved element {interleaved_element.name} could not be converted to Clifford"
            ) from err
        # Convert interleaved element to operation
        self._interleaved_op = interleaved_element
        if not isinstance(interleaved_element, Instruction):
            self._interleaved_op = interleaved_element.to_instruction()
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
        # Sample random group element sequences
        sequences = self._sample_sequences()

        # Convert each sequence into circuit and append the inverse to the end.
        circuits = []
        for seq in sequences:
            # Build reference (standard) RB circuit
            srb_circ = self._sequence_to_circuit(seq)
            circuits.append(srb_circ)
            # Build interleaved RB circuit
            irb_circ = self._sequence_to_circuit(seq, self._interleaved_op)
            circuits.append(irb_circ)
        return circuits

    def _sequence_to_circuit(
        self, sequence: List[GroupOperatorType], interleaved_op: Optional[Instruction] = None
    ) -> QuantumCircuit:
        """Convert a RB sequence into circuit and append the inverse to the end.

        Returns:
            A RB circuit.
        """
        qubits = list(range(self.num_qubits))
        circ = QuantumCircuit(self.num_qubits)
        circ.barrier(qubits)
        for elem in sequence:
            circ.append(self._to_instruction(elem), qubits)
            circ.barrier(qubits)
            if interleaved_op:
                circ.append(interleaved_op, qubits)
                circ.barrier(qubits)
        # Add inverse
        # Avoid op.compose() for fast op construction TODO: revisit after terra#7483
        op = self._twirling_group.generator(circ)
        inv = op.adjoint()
        circ.append(self._to_instruction(inv), qubits)
        circ.barrier(qubits)  # TODO: Can we remove this? (measure_all inserts one more barrier)
        circ.measure_all()
        circ.metadata = {
            "experiment_type": self._type,
            "xval": len(sequence),
            "group": self._twirling_group.string,
            "physical_qubits": self.physical_qubits,
            "interleaved": bool(interleaved_op),
        }
        return circ
