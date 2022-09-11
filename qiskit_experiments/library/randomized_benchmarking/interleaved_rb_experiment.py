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

from numpy.random import Generator, default_rng
from numpy.random.bit_generator import BitGenerator, SeedSequence

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Clbit, Instruction
from qiskit.quantum_info import Clifford
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit.compiler import transpile

from .rb_experiment import StandardRB
from .interleaved_rb_analysis import InterleavedRBAnalysis
from .clifford_utils import CliffordUtils


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

        super().__init__(
            qubits,
            lengths,
            backend=backend,
            num_samples=num_samples,
            seed=seed,
            full_sampling=full_sampling,
        )
        self._set_interleaved_element(interleaved_element)
        self._transpiled_interleaved_elem = None
        self.analysis = InterleavedRBAnalysis()
        self.analysis.set_options(outcome="0" * self.num_qubits)

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of RB circuits.

        Returns:
            A list of :class:`QuantumCircuit`.

        Raises:
            QiskitError: if basis_gates is not set in transpile_options nor in backend configuration.
        """
        if self.num_qubits > 2:
            return super().circuits()
        rng = default_rng(seed=self.experiment_options.seed)
        circuits = []
        if self._clifford_utils is None:
            self._clifford_utils = CliffordUtils(
                self.num_qubits, self.transpile_options.basis_gates, backend=self.backend
            )
        if not hasattr(self.transpile_options, "basis_gates"):
            if self.backend.configuration.basis_gates:
                self.set_transpile_options(basis_gates=self.backend.configuration.basis_gates)
            else:
                self.transpile_options["basis_gates"] = self.default_basis_gates

        for _ in range(self.experiment_options.num_samples):
            self._set_transpiled_interleaved_element()
            std_circuits, int_circuits = self._build_rb_circuits(
                self.experiment_options.lengths,
                rng,
            )
            circuits += std_circuits
            circuits += int_circuits
        return circuits

    def _set_interleaved_element(self, interleaved_element):
        """Handle the various types of the interleaved element

        Args:
            interleaved_element: The element to interleave

        Raises:
            QiskitError: if there is no known conversion of interleaved_element
            to a Clifford group element
        """
        try:
            interleaved_element_op = Clifford(interleaved_element)
            self._interleaved_element = (interleaved_element, interleaved_element_op)
        except QiskitError as error:
            raise QiskitError(
                "Interleaved element {} could not be converted to Clifford element".format(
                    interleaved_element.name
                )
            ) from error

    def _set_transpiled_interleaved_element(self):
        """
        Create the transpiled interleaved element. If it is a single gate,
        create a circuit comprising this gate.
        """
        if not isinstance(self._interleaved_element, QuantumCircuit):
            if self.num_qubits == 1:
                qc_interleaved = QuantumCircuit(1, 1)
                qubits = [0]
            else:
                qc_interleaved = QuantumCircuit(2, 2)
                qubits = [0, 1]

            qc_interleaved.append(self._interleaved_element[0], qubits)
            self._transpiled_interleaved_elem = qc_interleaved
        else:
            qc_interleaved = self._interleaved_element
        if hasattr(self.transpile_options, "basis_gates"):
            basis_gates = self.transpile_options.basis_gates
        else:
            basis_gates = None
        self._transpiled_interleaved_elem = transpile(
            circuits=qc_interleaved,
            optimization_level=1,
            basis_gates=basis_gates,
            backend=self._backend,
        )

    def _build_rb_circuits(self, lengths: List[int], rng: Generator) -> List[QuantumCircuit]:
        """
        build_rb_circuits
        Args:
                lengths: A list of RB sequence lengths. We create random circuits
                         where the number of cliffords in each is defined in lengths.
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
        all_rb_interleaved_circuits = []

        # When full_sampling==False, each circuit is the prefix of the next circuit (without the
        # inverse Clifford at the end of the circuit. The variable 'circ' will contain
        # the growing circuit.
        # When each circuit reaches its length, we copy it to rb_circ, append the inverse,
        # and add it to the list of circuits.
        n = self.num_qubits
        qubits = list(range(n))
        clbits = list(range(n))

        interleaved_circ = QuantumCircuit(max_qubit, n)
        interleaved_circ.barrier(qubits)

        circ = QuantumCircuit(max_qubit, n)
        circ.barrier(qubits)
        circ = transpile(
            circuits=circ,
            optimization_level=1,
            basis_gates=self.transpile_options.basis_gates,
            backend=self._backend,
        )
        # composed_cliff_num is the number representing the composition of all the Cliffords up to now
        # composed_interleaved_num is the same for an interleaved circuit
        composed_cliff_num = 0  # 0 is the Clifford that is Id
        composed_interleaved_num = 0
        prev_length = 0

        for length in lengths:
            for i in range(prev_length, length):
                circ, next_circ, composed_cliff_num = self._add_random_cliff_to_circ(
                    circ, composed_cliff_num, qubits, rng
                )
                interleaved_circ, composed_interleaved_num = self._add_cliff_to_circ(
                    interleaved_circ, next_circ, composed_interleaved_num, qubits
                )

                # The interleaved element is appended after every Clifford
                interleaved_circ, composed_interleaved_num = self._add_cliff_to_circ(
                    interleaved_circ,
                    self._transpiled_interleaved_elem,
                    composed_interleaved_num,
                    qubits,
                )
                if i == length - 1:
                    rb_circ = circ.copy()  # circ is used as the prefix of the next circuit
                    rb_circ = self._add_inverse_to_circ(rb_circ, composed_cliff_num, qubits, clbits)

                    rb_circ.metadata = {
                        "experiment_type": "rb",
                        "xval": length,
                        "group": "Clifford",
                        "physical_qubits": self.physical_qubits,
                        "interleaved": False,
                    }
                    all_rb_circuits.append(rb_circ)

                    # interleaved_circ is used as the prefix of the next circuit
                    rb_interleaved_circ = interleaved_circ.copy()
                    rb_interleaved_circ = self._add_inverse_to_circ(
                        rb_interleaved_circ, composed_interleaved_num, qubits, clbits
                    )
                    rb_interleaved_circ.metadata = {
                        "experiment_type": "rb",
                        "xval": length,
                        "group": "Clifford",
                        "physical_qubits": self.physical_qubits,
                        "interleaved": True,
                    }
                    all_rb_interleaved_circuits.append(rb_interleaved_circ)

                prev_length = i + 1
        return all_rb_circuits, all_rb_interleaved_circuits

    def _build_rb_circuits_full_sampling(
        self, lengths: List[int], rng: Generator
    ) -> List[QuantumCircuit]:
        """
        _build_rb_circuits_full_sampling
        Args:
                lengths: A list of RB sequence lengths. We create random circuits
                    where the number of cliffords in each is defined in lengths.
                rng: Generator object for random number generation.
                    If None, default_rng will be used.
                interleaved_element: the interleaved element as a QuantumCircuit.

        Returns:
                The transpiled RB circuits.

        Additional information:
            This is similar to _build_rb_circuits for the case of full_sampling.
        """
        all_rb_circuits = []
        all_rb_interleaved_circuits = []

        n = self.num_qubits
        qubits = list(range(n))
        clbits = list(range(n))
        max_qubit = max(self.physical_qubits) + 1
        for length in lengths:
            # We define the circuit size here, for the layout that will
            # be created later
            rb_circ = QuantumCircuit(max_qubit, n)
            rb_circ.barrier(qubits)
            rb_interleaved_circ = QuantumCircuit(max_qubit, n)
            rb_interleaved_circ.barrier(qubits)
            rb_circ = transpile(
                circuits=rb_circ,
                optimization_level=1,
                basis_gates=self.transpile_options.basis_gates,
                backend=self._backend,
            )

            # composed_cliff_num is the number representing the composition of
            # all the Cliffords up to now
            # composed_interleaved_num is the same for an interleaved circuit
            composed_cliff_num = 0
            composed_interleaved_num = 0
            # For full_sampling, we create each circuit independently.
            for _ in range(length):
                rb_circ, next_circ, composed_cliff_num = self._add_random_cliff_to_circ(
                    rb_circ, composed_cliff_num, qubits, rng
                )
                rb_interleaved_circ, composed_interleaved_num = self._add_cliff_to_circ(
                    rb_interleaved_circ, next_circ, composed_interleaved_num, qubits
                )
                # The interleaved element is appended after every Clifford and its barrier
                rb_interleaved_circ, composed_interleaved_num = self._add_cliff_to_circ(
                    rb_interleaved_circ,
                    self._transpiled_interleaved_elem,
                    composed_interleaved_num,
                    qubits,
                )

            rb_circ = self._add_inverse_to_circ(rb_circ, composed_cliff_num, qubits, clbits)
            rb_circ.metadata = {
                "experiment_type": "rb",
                "xval": length,
                "group": "Clifford",
                "physical_qubits": self.physical_qubits,
                "interleaved": False,
            }

            rb_interleaved_circ = self._add_inverse_to_circ(
                rb_interleaved_circ, composed_interleaved_num, qubits, clbits
            )
            rb_interleaved_circ.metadata = {
                "experiment_type": "rb",
                "xval": length,
                "group": "Clifford",
                "physical_qubits": self.physical_qubits,
                "interleaved": True,
            }
            all_rb_circuits.append(rb_circ)
            all_rb_interleaved_circuits.append(rb_interleaved_circ)
        return all_rb_circuits, all_rb_interleaved_circuits

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
            else:
                cargs = []

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

    def _sample_circuits(self, lengths, rng):
        """ This method is called when the number of qubits is greater than 2 """
        circuits = []
        for length in lengths if self._full_sampling else [lengths[-1]]:
            elements = self._clifford_utils.random_clifford_circuits(length, rng)
            element_lengths = [len(elements)] if self._full_sampling else lengths
            std_circuits = self._generate_circuit(elements, element_lengths)
            for circuit in std_circuits:
                circuit.metadata["interleaved"] = False
            circuits += std_circuits

            int_elements = self._interleave(elements)
            int_elements_lengths = [length * 2 for length in element_lengths]
            int_circuits = self._generate_circuit(int_elements, int_elements_lengths)
            for circuit in int_circuits:
                circuit.metadata["interleaved"] = True
                circuit.metadata["xval"] = circuit.metadata["xval"] // 2
            circuits += int_circuits
        return circuits
    def _interleave(self, element_list: List) -> List:
        """Interleaving the interleaved element inside the element list.
        Args:
            element_list: The list of elements we add the interleaved element to.
        Returns:
            The new list with the element interleaved.
        """
        new_element_list = []
        for element in element_list:
            new_element_list.append(element)
            new_element_list.append(self._interleaved_element)
        return new_element_list
