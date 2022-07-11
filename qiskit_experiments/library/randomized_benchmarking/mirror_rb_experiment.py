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
from numpy.random import Generator
from numpy.random.bit_generator import BitGenerator, SeedSequence

import numpy as np

try:
    import pygsti
    from pygsti.processors import QubitProcessorSpec as QPS
    from pygsti.processors import CliffordCompilationRules as CCR
    from pygsti.baseobjs import QubitGraph as QG

    HAS_PYGSTI = True
except ImportError:
    HAS_PYGSTI = False

from qiskit import QuantumCircuit, QiskitError
from qiskit.circuit import Instruction
from qiskit.quantum_info import Clifford, random_pauli, random_clifford
from qiskit.quantum_info.operators import Pauli
from qiskit.providers.backend import Backend

from .rb_experiment import StandardRB
from .mirror_rb_analysis import MirrorRBAnalysis


class MirrorRB(StandardRB):
    """Mirror randomized benchmarking experiment.

    # section: overview
        Mirror Randomized Benchmarking (RB) is a method to estimate the average
        error-rate of quantum gates that is more scalable than other RB methods
        and can thus detect crosstalk errors.

        A mirror RB experiment generates circuits of layers of Cliffords interleaved
        with layers of Pauli gates and capped at the start and end by a layer of
        single-qubit Cliffords. The second half of the Clifford layers are the
        inverses of the first half of Clifford layers. After running the circuits on
        a backend, various quantities (success probability, adjusted success
        probability, and effective polarization) are computed and used to fit an
        exponential decay curve and calculate the EPC (error per Clifford, also
        referred to as the average gate infidelity) and entanglement infidelity. Find
        more on adjusted success probability, effective polarization, and
        entanglement infidelity in Refs. [1, 2, 3].

    # section: analysis_ref
        :py:class:`MirrorRBAnalysis`

    # section: reference
        .. ref_arxiv:: 1 2112.09853
        .. ref_arxiv:: 2 2008.11294
        .. ref_arxiv:: 3 2204.07568

    """

    def __init__(
        self,
        qubits: Sequence[int],
        lengths: Iterable[int],
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
            qubits: A list of physical qubits for the experiment.
            lengths: A list of RB sequences lengths.
            local_clifford: If True, begin the circuit with uniformly random 1-qubit
                            Cliffords and end the circuit with their inverses.
            pauli_randomize: If True, surround each inner Clifford layer with
                             uniformly random Paulis.
            two_qubit_gate_density: Expected proportion of qubits with CNOTs based on
                                    the backend coupling map.
            backend: The backend to run the experiment on.
            num_samples: Number of samples to generate for each
                         sequence length.
            seed: Optional, seed used to initialize ``numpy.random.default_rng``.
                  when generating circuits. The ``default_rng`` will be initialized
                  with this seed value everytime :meth:`circuits` is called.
            full_sampling: If True all Cliffords are independently sampled for
                           all lengths. If False for sample of lengths longer
                           sequences are constructed by appending additional
                           Clifford samples to shorter sequences.
            inverting_pauli_layer: If True, a layer of Pauli gates is appended at the
                                   end of the circuit to set all qubits to 0 (with
                                   possibly a global phase)

        Raises:
            QiskitError: if an odd length or a negative two qubit gate density is provided
        """
        # All lengths must be even
        if not all(length % 2 == 0 for length in lengths):
            raise QiskitError("All lengths must be even")

        # Two-qubit density must be non-negative
        if two_qubit_gate_density < 0:
            raise QiskitError("Two-qubit gate density must be non-negative")

        super().__init__(
            qubits,
            lengths,
            backend=backend,
            num_samples=num_samples,
            seed=seed,
            full_sampling=full_sampling,
        )

        self._local_clifford = local_clifford
        self._pauli_randomize = pauli_randomize
        self._two_qubit_gate_density = two_qubit_gate_density

        # By default, the inverting Pauli layer at the end of the circuit is not added
        self._inverting_pauli_layer = inverting_pauli_layer

        # Set analysis options
        self.analysis = MirrorRBAnalysis()
        self.analysis.set_options(outcome="0" * self.num_qubits)

    def _sample_circuits(self, lengths, rng) -> List[QuantumCircuit]:
        """Sample Mirror RB circuits.

        Steps:
        1. Sample length/2 layers of random Cliffords
        2. Compute inverse of each layer in the first half of the circuit and append to circuit
        3. Sample the random paulis and interleave them between the Clifford layers
        4. Sample the 1-qubit local Clifford and add them to the beginning and end of the circuit

        Args:
            lengths: List of lengths to run Mirror RB
            rng: Generator seed

        Returns:
            List of QuantumCircuits

        Raises:
            QiskitError: if backend without a coupling map is provided
        """

        # Backend must have a coupling map
        if not self._backend or not self._backend.configuration().coupling_map:
            raise QiskitError(
                "Must provide a backend with a coupling map or provide "
                + "coupling map if using a simulator"
            )

        circuits = []
        lengths_half = [length // 2 for length in lengths]
        for length in lengths_half if self._full_sampling else [lengths_half[-1]]:
            # Sample Clifford layer elements for first half of mirror circuit
            coupling_map = self._backend.configuration().coupling_map
            experiment_coupling_map = []
            for edge in coupling_map:
                if edge[0] in self.physical_qubits and edge[1] in self.physical_qubits:
                    experiment_coupling_map.append(edge)
            elements = self._clifford_utils.random_edgegrab_clifford_circuits(
                self.physical_qubits,
                experiment_coupling_map,
                self._two_qubit_gate_density,
                length,
                rng,
            )

            # Copy Clifford layer elements in first half of circuit
            elements_without_inv = elements[:]

            # Append inverses of Clifford elements to second half of circuit
            for element in elements_without_inv[::-1]:
                elements.append(element.inverse())
            element_lengths = [len(elements)] if self._full_sampling else lengths

            # Interleave random Paulis if set by user
            if self._pauli_randomize:
                elements = self._pauli_dress(elements, rng)
                element_lengths = [length * 2 + 1 for length in element_lengths]
                int_circuits = self._generate_mirror(elements, element_lengths)
                for circuit in int_circuits:
                    circuit.metadata["mirror"] = True
                    circuit.metadata["xval"] = (circuit.metadata["xval"] - 1) // 2

            # Add start and end local cliffords if set by user
            if self._local_clifford:
                element_lengths = [length + 2 for length in element_lengths]
                elements = self._start_end_cliffords(elements, rng)
            mirror_circuits = self._generate_mirror(elements, element_lengths)
            for circuit in mirror_circuits:
                # Use "boolean arithmetic" to calculate xval correctly for each circuit
                pauli_scale = self._pauli_randomize + 1
                clifford_const = self._local_clifford * 2
                circuit.metadata["xval"] = (
                    circuit.metadata["xval"] - self._pauli_randomize - clifford_const
                ) // pauli_scale
                circuit.metadata["mirror"] = True
            circuits += mirror_circuits

        # Append inverting Pauli layer at end of circuit if set by user
        if self._inverting_pauli_layer:
            for circuit in circuits:
                target = circuit.metadata["target"]
                label = "".join(["X" if char == "1" else "I" for char in target])
                circuit.remove_final_measurements()
                circuit.append(Pauli(label), list(range(self._num_qubits)))
                circuit.measure_all()

        return circuits

    def _pauli_dress(self, element_list: List, rng: Optional[Union[int, Generator]]) -> List:
        """Interleaving layers of random Paulis inside the element list.

        Args:
            element_list: The list of elements we add the interleaved Paulis to.
            rng: (Seed for) random number generator

        Returns:
            The new list of elements with the Paulis interleaved.
        """
        # Generate random Pauli
        rand_pauli = random_pauli(self._num_qubits, seed=rng).to_instruction()
        rand_pauli_op = Clifford(rand_pauli)
        new_element_list = [(rand_pauli, rand_pauli_op)]
        for element in element_list:
            new_element_list.append(element)
            rand_pauli = random_pauli(self._num_qubits, seed=rng).to_instruction()
            rand_pauli_op = Clifford(rand_pauli)
            new_element_list.append((rand_pauli, rand_pauli_op))
        return new_element_list

    def _start_end_cliffords(
        self, elements: Iterable[Clifford], rng: Optional[Union[int, Generator]]
    ) -> List[QuantumCircuit]:
        """Add a layer of uniformly random  1-qubit Cliffords to the beginning of the list
           and its inverse to the end of the list

        Args:
            element_list: The list of elements we add the Clifford layers to
            rng: (Seed for) random number generator

        Returns:
            The new list of elements with the start and end local (1-qubit) Cliffords.
        """
        rand_clifford = [
            self._clifford_utils.random_cliffords(num_qubits=1, rng=rng)[0]
            for _ in self.physical_qubits
        ]
        tensor_op = rand_clifford[0]
        for cliff in rand_clifford[1:]:
            tensor_op = tensor_op ^ cliff
        tensor_circ = tensor_op.to_circuit()

        rand_clifford = random_clifford(self.num_qubits, seed=rng).to_circuit()
        return [tensor_circ] + elements + [tensor_circ.inverse()]

    def _generate_mirror(
        self, elements: Iterable[Clifford], lengths: Iterable[int]
    ) -> List[QuantumCircuit]:
        """Return the RB circuits constructed from the given element list with the second
           half as the inverse of the first half

        Args:
            elements: A list of Clifford elements
            lengths: A list of RB sequences lengths.

        Returns:
            A list of :class:`QuantumCircuit`s.

        Additional information:
            The circuits are constructed iteratively; each circuit is obtained
            by extending the previous circuit (without the inversion and measurement gates)
        """
        qubits = list(range(self.num_qubits))
        circuits = []

        circs = [QuantumCircuit(self.num_qubits) for _ in range(len(lengths))]

        for current_length, group_elt_circ in enumerate(elements[: (len(elements) // 2)]):
            if isinstance(group_elt_circ, tuple):
                group_elt_gate = group_elt_circ[0]
            else:
                group_elt_gate = group_elt_circ

            if not isinstance(group_elt_gate, Instruction):
                group_elt_gate = group_elt_gate.to_instruction()
            for circ in circs:
                circ.barrier(qubits)
                circ.append(group_elt_gate, qubits)

            double_current_length = (
                (current_length + 1) * 2 + 1 if len(elements) % 2 == 1 else (current_length + 1) * 2
            )
            if double_current_length in lengths:
                rb_circ = circs.pop()
                inv_start = (
                    (-(current_length + 1) - 1) if len(elements) % 2 == 1 else -(current_length + 1)
                )
                for inv in elements[inv_start:]:
                    if isinstance(inv, tuple):
                        group_elt_gate = inv[0]
                    else:
                        group_elt_gate = inv

                    if not isinstance(group_elt_gate, Instruction):
                        group_elt_gate = group_elt_gate.to_instruction()
                    rb_circ.barrier(qubits)
                    rb_circ.append(group_elt_gate, qubits)
                rb_circ.metadata = {
                    "experiment_type": self._type,
                    "xval": double_current_length,
                    "group": "Clifford",
                    "physical_qubits": self.physical_qubits,
                    "target": self._clifford_utils.compute_target_bitstring(rb_circ),
                    "inverting_pauli_layer": self._inverting_pauli_layer,
                }
                rb_circ.measure_all()
                circuits.append(rb_circ)
        return circuits


class MirrorRBPyGSTi(MirrorRB):
    """Mirror RB experiment that uses pyGSTi's circuit generation. This subclass
    is primarily used for testing."""

    def __init__(
        self,
        qubits: Sequence[int],
        lengths: Iterable[int],
        local_clifford: bool = True,
        pauli_randomize: bool = True,
        two_qubit_gate_density: float = 0.2,
        backend: Optional[Backend] = None,
        num_samples: int = 3,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        full_sampling: bool = False,
        inverting_pauli_layer: bool = False,
    ):
        """Initialize a mirror randomized benchmarking experiment that uses
        pyGSTi's circuit generation.

        Args:
            qubits: A list of physical qubits for the experiment.
            lengths: A list of RB sequences lengths.
            local_clifford: If True, begin the circuit with uniformly random 1-qubit
                            Cliffords and end the circuit with their inverses.
            pauli_randomize: If True, surround each inner Clifford layer with
                            uniformly random Paulis.
            two_qubit_gate_density: Expected proportion of qubits with CNOTs based on
                                    the backend coupling map.
            backend: The backend to run the experiment on.
            num_samples: Number of samples to generate for each
                        sequence length.
            seed: Optional, seed used to initialize ``numpy.random.default_rng``.
                when generating circuits. The ``default_rng`` will be initialized
                with this seed value everytime :meth:`circuits` is called.
            full_sampling: If True all Cliffords are independently sampled for
                        all lengths. If False for sample of lengths longer
                        sequences are constructed by appending additional
                        Clifford samples to shorter sequences.
            inverting_pauli_layer: If True, a layer of Pauli gates is appended at the
                                end of the circuit to set all qubits to 0 (with
                                possibly a global phase)

        Raises:
            ImportError: if user does not have pyGSTi installed
        """
        if not HAS_PYGSTI:
            raise ImportError("MirrorRBPyGSTi requires pyGSTi to generate circuits.")

        super().__init__(
            qubits,
            lengths,
            local_clifford=local_clifford,
            pauli_randomize=pauli_randomize,
            two_qubit_gate_density=two_qubit_gate_density,
            backend=backend,
            num_samples=num_samples,
            seed=seed,
            full_sampling=full_sampling,
            inverting_pauli_layer=inverting_pauli_layer,
        )

        self.analysis = MirrorRBAnalysis()
        self._lengths = lengths
        self._num_samples = num_samples
        self._seed = seed
        self.analysis.set_options(outcome="0" * self.num_qubits)

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        """Return a list of experiment circuits, transpiled, with transpiled
        circuits stored as metadata."""
        transpiled = super()._transpiled_circuits()

        # Store transpiled circuits in metadata
        for circ in transpiled:
            circ.metadata["transpiled_qiskit_circ"] = circ

        return transpiled

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of RB circuits generated with PyGSTi.

        Returns:
            A list of :class:`QuantumCircuit`.
        """
        # Number of qubits to perform MRB on
        n_qubits = self._num_qubits

        # Maximum number of qubits in device
        max_qubits = self._backend.configuration().n_qubits
        qubit_labels = ["Q" + str(i) for i in range(max_qubits)]

        # List of gates to construct circuits with (CNOT and the 24 one-qubit Cliffords)
        gate_names = ["Gcnot"] + [f"Gc{i}" for i in range(24)]

        # Construct connectivity map of backend
        backend_edge_list = [list(edge) for edge in self._backend.configuration().coupling_map]
        connectivity = np.zeros((max_qubits, max_qubits))
        for edge in backend_edge_list:
            connectivity[edge[0]][edge[1]] = 1
        connectivity = np.array(connectivity, dtype=bool)

        # Define CNOT availability in backend
        availability = {"Gcnot": []}
        for i in range(max_qubits):
            for j in range(i + 1, max_qubits):
                if connectivity[i][j]:
                    availability["Gcnot"].append(("Q" + str(i), "Q" + str(j)))

        # Initialize graph and quantum processor spec
        graph = QG(qubit_labels=qubit_labels, initial_connectivity=connectivity)
        pspec = QPS(
            max_qubits,
            gate_names,
            availability=availability,
            qubit_labels=qubit_labels,
            geometry=graph,
        )

        # Compilation rules for how to combine (or not) random Pauli gates
        compilations = {
            "absolute": CCR.create_standard(
                pspec, "absolute", ("paulis", "1Qcliffords"), verbosity=0
            ),
            "paulieq": CCR.create_standard(
                pspec, "paulieq", ("1Qcliffords", "allcnots"), verbosity=0
            ),
        }

        # Depths to run MRB
        depths = self._lengths

        # Number of samples at each depth
        num_samples = self._num_samples

        # Random circuit sampler algorithm and the average density of 2Q gate per layer
        sampler = "edgegrab"
        samplerargs = [2 * self._two_qubit_gate_density]

        # Create pyGSTi experiment design
        mrb_design = pygsti.protocols.MirrorRBDesign(
            pspec,
            depths,
            num_samples,
            qubit_labels=tuple(qubit_labels[:n_qubits]),
            sampler=sampler,
            clifford_compilations=compilations,
            samplerargs=samplerargs,
            seed=self._seed,
        )

        # Create list of circuits to run and analyze using Qiskit Experiments framework
        circuits = []
        for idx, d in enumerate(depths):
            for sample in range(num_samples):
                # Convert PyGSTi circuits to qasm object and then to QuantumCircuit object
                qasm = mrb_design.all_circuits_needing_data[
                    idx * num_samples + sample
                ].convert_to_openqasm()
                rb_circ = QuantumCircuit.from_qasm_str(qasm)

                # Store metadata, such as target bitstring and experiment design, to the circuits
                rb_circ.metadata = {
                    "experiment_type": self._type,
                    "xval": d,
                    "group": "Clifford",
                    "physical_qubits": self.physical_qubits,
                    "target": mrb_design.idealout_lists[idx][sample][0][::-1],
                    "pygsti_circ": mrb_design,
                    "inverting_pauli_layer": self._inverting_pauli_layer,
                    "mirror": True,
                }

                circuits.append(rb_circ)

        # Add final layer of inverting Pauli gates if specified by user
        if self._inverting_pauli_layer:
            for i, circuit in enumerate(circuits):
                target = circuit.metadata["target"]
                label = "".join(["X" if char == "1" else "I" for char in target])
                circuit.remove_final_measurements()
                circuit.barrier(list(range(self._num_qubits)))
                circuit.append(Pauli(label=label), list(range(n_qubits)))
                circuit.measure_all()

        return circuits
