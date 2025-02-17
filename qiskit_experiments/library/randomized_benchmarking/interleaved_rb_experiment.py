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
import itertools
from typing import Union, Iterable, Optional, List, Sequence, Dict, Any

from numpy.random import Generator
from numpy.random.bit_generator import BitGenerator, SeedSequence

from qiskit.circuit import QuantumCircuit, Instruction, Gate, Delay
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit.quantum_info import Clifford
from qiskit_experiments.framework import Options
from qiskit_experiments.framework.backend_timing import BackendTiming
from .clifford_utils import _synthesize_clifford
from .clifford_utils import num_from_1q_circuit, num_from_2q_circuit
from .interleaved_rb_analysis import InterleavedRBAnalysis
from .standard_rb import StandardRB, SequenceElementType


class InterleavedRB(StandardRB):
    """An experiment to characterize the error rate of a specific gate on a device.

    # section: overview
        Interleaved Randomized Benchmarking (RB) is a method
        to estimate the average error-rate of a certain quantum gate.

        An interleaved RB experiment generates a standard RB sequences of random Cliffords
        and another sequence with the interleaved given gate.
        After running the two sequences on a backend, it calculates the probabilities to get back to
        the ground state, fits the two exponentially decaying curves, and estimates
        the interleaved gate error. See Ref. [1] for details.

    # section: analysis_ref
        :class:`InterleavedRBAnalysis`

    # section: manual
        :doc:`/manuals/verification/randomized_benchmarking`

    # section: reference
        .. ref_arxiv:: 1 1203.4550

    # section: example
        .. jupyter-execute::
            :hide-code:

            # backend
            from qiskit_aer import AerSimulator
            from qiskit_ibm_runtime.fake_provider import FakePerth

            backend = AerSimulator.from_backend(FakePerth())

        .. jupyter-execute::

            import numpy as np
            from qiskit_experiments.library import StandardRB, InterleavedRB
            from qiskit_experiments.framework import ParallelExperiment, BatchExperiment
            import qiskit.circuit.library as circuits

            lengths = np.arange(1, 200, 30)
            num_samples = 10
            seed = 1010
            qubits = (1, 2)

            int_exp2 = InterleavedRB(
                circuits.CXGate(), qubits, lengths, num_samples=num_samples, seed=seed)

            int_expdata2 = int_exp2.run(backend=backend).block_for_results()
            int_results2 = int_expdata2.analysis_results()
            display(int_expdata2.figure(0))

            names = {result.name for result in int_results2}
            print(f"Available results: {names}")
    """

    def __init__(
        self,
        interleaved_element: Union[QuantumCircuit, Gate, Delay, Clifford],
        physical_qubits: Sequence[int],
        lengths: Iterable[int],
        backend: Optional[Backend] = None,
        num_samples: int = 3,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        full_sampling: bool = False,
        circuit_order: str = "RIRIRI",
    ):
        """Initialize an interleaved randomized benchmarking experiment.

        Args:
            interleaved_element: The element to interleave,
                    given either as a Clifford element, gate, delay or circuit.
                    All instructions in the element must be supported in the ``backend``(``target``).
                    If it is/contains a delay, its duration and unit must comply with
                    the timing constraints of the ``backend``
                    (:class:`~qiskit_experiments.framework.backend_timing.BackendTiming`
                    is useful to obtain valid delays).
                    Parameterized circuits/instructions are not allowed.
            physical_qubits: list of physical qubits for the experiment.
            lengths: A list of RB sequences lengths.
            backend: The backend to run the experiment on.
            num_samples: Number of samples to generate for each sequence length.
            seed: Optional, seed used to initialize ``numpy.random.default_rng``.
                  when generating circuits. The ``default_rng`` will be initialized
                  with this seed value every time :meth:`circuits` is called.
            full_sampling: If True all Cliffords are independently sampled for
                           all lengths. If False for sample of lengths longer
                           sequences are constructed by appending additional
                           Clifford samples to shorter sequences.
            circuit_order: How to order the reference and the interleaved circuits.
                ``"RIRIRI"`` (default) - Alternate a reference and an interleaved circuit. Or
                ``"RRRIII"`` - Push all reference circuits first, then all interleaved ones.

        Raises:
            QiskitError: When interleaved_element has different number of qubits
                from the physical_qubits argument.
            QiskitError: When interleaved_element is not convertible to Clifford object.
            QiskitError: When interleaved_element has an invalid delay
                (e.g. violating the timing constraints of the backend).
        """
        # Validations of interleaved_element
        # - validate number of qubits of interleaved_element
        if len(physical_qubits) != interleaved_element.num_qubits:
            raise QiskitError(
                f"Mismatch in number of qubits between qubits ({len(physical_qubits)})"
                f" and interleaved element ({interleaved_element.num_qubits})."
            )
        # - validate if interleaved_element is Clifford
        try:
            interleaved_clifford = Clifford(interleaved_element)
        except QiskitError as err:
            raise QiskitError(
                f"Interleaved element {interleaved_element.name} could not be converted to Clifford."
            ) from err
        # - validate delays in interleaved_element
        delay_ops = []
        if isinstance(interleaved_element, Delay):
            delay_ops = [interleaved_element]
        elif isinstance(interleaved_element, QuantumCircuit):
            delay_ops = [delay.operation for delay in interleaved_element.get_instructions("delay")]
        if delay_ops:
            timing = BackendTiming(backend)
        else:
            timing = None
        for delay_op in delay_ops:
            if delay_op.unit != timing.delay_unit:
                raise QiskitError(
                    f"Interleaved delay for backend {backend} must have time unit {timing.delay_unit}."
                    " Use BackendTiming to set valid duration and unit for delays."
                )
            if timing.delay_unit == "dt":
                valid_duration = timing.round_delay(samples=delay_op.duration)
                if delay_op.duration != valid_duration:
                    raise QiskitError(
                        f"Interleaved delay duration {delay_op.duration}[dt] violates the timing"
                        f" constraints of the backend {backend}. It could be {valid_duration}[dt]."
                        " Use BackendTiming to set valid duration for delays."
                    )

        super().__init__(
            physical_qubits,
            lengths,
            backend=backend,
            num_samples=num_samples,
            seed=seed,
            full_sampling=full_sampling,
        )
        # Convert interleaved element to integer for speed in 1Q or 2Q case
        if self.num_qubits == 1:
            self._interleaved_cliff = num_from_1q_circuit(interleaved_clifford.to_circuit())
        elif self.num_qubits == 2:
            self._interleaved_cliff = num_from_2q_circuit(interleaved_clifford.to_circuit())
        # Convert interleaved element to circuit for speed in 3Q or more case
        else:
            self._interleaved_cliff = interleaved_clifford.to_circuit()
        self._interleaved_element = interleaved_element  # Original interleaved element
        self._interleaved_op = None  # Transpiled interleaved element for speed
        self.set_experiment_options(circuit_order=circuit_order)
        self.analysis = InterleavedRBAnalysis()
        self.analysis.set_options(outcome="0" * self.num_qubits)

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default InterleavedRB experiment options.

        Experiment Options:
            circuit_order (str): How to order the reference and the interleaved circuits.
                ``"RIRIRI"`` (alternate a reference and an interleaved circuit) or
                ``"RRRIII"`` (push all reference circuits first, then all interleaved ones).
        """
        options = super()._default_experiment_options()
        options.update_options(
            circuit_order="RIRIRI",
        )
        return options

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of RB circuits.

        Returns:
            A list of :class:`QuantumCircuit`.

        Raises:
            QiskitError: If interleaved_element has non-supported instruction in the backend.
        """
        # Convert interleaved element to transpiled circuit operation and store it for speed
        self.__set_up_interleaved_op()

        # Build circuits of reference sequences
        reference_sequences = self._sample_sequences()
        reference_circuits = self._sequences_to_circuits(reference_sequences)
        for circ, seq in zip(reference_circuits, reference_sequences):
            circ.metadata = {
                "xval": len(seq),
                "group": "Clifford",
                "interleaved": False,
            }
        # Build circuits of interleaved sequences
        interleaved_sequences = []
        for seq in reference_sequences:
            new_seq = []
            for elem in seq:
                new_seq.append(elem)
                new_seq.append(self._interleaved_cliff)
            interleaved_sequences.append(new_seq)
        interleaved_circuits = self._sequences_to_circuits(interleaved_sequences)
        for circ, seq in zip(interleaved_circuits, reference_sequences):
            circ.metadata = {
                "xval": len(seq),  # set length of the reference sequence
                "group": "Clifford",
                "interleaved": True,
            }

        if self.experiment_options.circuit_order == "RRRIII":
            return reference_circuits + interleaved_circuits
        # Default order: RIRIRI
        return list(itertools.chain.from_iterable(zip(reference_circuits, interleaved_circuits)))

    def _to_instruction(
        self,
        elem: SequenceElementType,
        synthesis_options: Dict[str, Optional[Any]],
    ) -> Instruction:
        if elem is self._interleaved_cliff:
            return self._interleaved_op

        return super()._to_instruction(elem, synthesis_options)

    def __set_up_interleaved_op(self) -> None:
        # Convert interleaved element to transpiled circuit operation and store it for speed
        self._interleaved_op = self._interleaved_element
        # Convert interleaved element to circuit
        if isinstance(self._interleaved_op, Clifford):
            opts = self._get_synthesis_options()
            self._interleaved_op = _synthesize_clifford(self._interleaved_op, **opts)

        if isinstance(self._interleaved_op, QuantumCircuit):
            interleaved_circ = self._interleaved_op
        elif isinstance(self._interleaved_op, Gate):
            interleaved_circ = QuantumCircuit(self.num_qubits, name=self._interleaved_op.name)
            interleaved_circ.append(self._interleaved_op, list(range(self.num_qubits)))
        else:  # Delay
            interleaved_circ = []

        # Validate if all instructions in the interleaved circuit are supported in the backend
        if self.backend and hasattr(self.backend, "target"):
            for inst in interleaved_circ:
                qargs = tuple(
                    self.physical_qubits[interleaved_circ.find_bit(q).index] for q in inst.qubits
                )
                if not self.backend.target.instruction_supported(inst.operation.name, qargs):
                    raise QiskitError(
                        f"{inst.operation.name} in interleaved element is not supported"
                        f" on qubits {qargs} in the backend."
                    )

        # Store interleaved operation as Instruction
        if isinstance(self._interleaved_op, QuantumCircuit):
            if not self._interleaved_op.name.startswith("Clifford"):
                self._interleaved_op.name = f"Clifford-{self._interleaved_op.name}"
            self._interleaved_op = self._interleaved_op.to_instruction()
