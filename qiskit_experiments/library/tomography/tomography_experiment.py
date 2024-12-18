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
Quantum Tomography experiment
"""

from typing import Union, Optional, Iterable, List, Tuple, Sequence
from itertools import product
from qiskit.circuit import QuantumCircuit, Instruction, ClassicalRegister, Clbit
from qiskit.circuit.library import PermutationGate
from qiskit.providers.backend import Backend
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit_experiments.exceptions import QiskitError
from qiskit_experiments.framework import BaseExperiment, BaseAnalysis, Options
from .basis import PreparationBasis, MeasurementBasis
from .tomography_analysis import TomographyAnalysis


class TomographyExperiment(BaseExperiment):
    """Base experiment for quantum state and process tomography.

    # section: analysis_ref
        :class:`TomographyAnalysis`

    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            basis_indices (Iterable[Tuple[List[int], List[int]]]): The basis elements to be measured.
                If None All basis elements will be measured.

        """
        options = super()._default_experiment_options()
        options.basis_indices = None
        return options

    def __init__(
        self,
        circuit: Union[QuantumCircuit, Instruction, BaseOperator],
        backend: Optional[Backend] = None,
        physical_qubits: Optional[Sequence[int]] = None,
        measurement_basis: Optional[MeasurementBasis] = None,
        measurement_indices: Optional[Sequence[int]] = None,
        preparation_basis: Optional[PreparationBasis] = None,
        preparation_indices: Optional[Sequence[int]] = None,
        conditional_circuit_clbits: Union[bool, Sequence[int], Sequence[Clbit]] = False,
        basis_indices: Optional[Iterable[Tuple[List[int], List[int]]]] = None,
        analysis: Union[BaseAnalysis, None, str] = "default",
    ):
        """Initialize a tomography experiment.

        Args:
            circuit: the quantum process circuit. If not a quantum circuit
                it must be a class that can be appended to a quantum circuit.
            backend: The backend to run the experiment on.
            physical_qubits: Optional, the physical qubits for the initial state circuit.
                If None this will be qubits [0, N) for an N-qubit circuit.
            measurement_basis: Tomography basis for measurements. If set to None
                no tomography measurements will be performed.
            measurement_indices: Optional, the `physical_qubits` indices to be
                measured as specified by the `measurement_basis`. If None all
                circuit physical qubits will be measured.
            preparation_basis: Tomography basis for measurements. If set to None
                no tomography preparations will be performed.
            preparation_indices: Optional, the `physical_qubits` indices to be
                prepared as specified by the `preparation_basis`. If None all
                circuit physical qubits will be prepared.
            basis_indices: Optional, the basis elements to be measured. If None
                All basis elements will be measured.
            conditional_circuit_clbits: Specify any clbits in the input
                circuit to treat as conditioning bits for conditional tomography.
                If set to True all circuit clbits will be treated as conditional.
                If False all circuit clbits will be marginalized over (Default: False).
            analysis: Optional, a custom analysis instance to use. If ``"default"``
                :class:`~.TomographyAnalysis` will be used. If None no analysis
                instance will be set.

        Raises:
            QiskitError: If input params are invalid.
        """
        # Initialize BaseExperiment
        if physical_qubits is None:
            physical_qubits = tuple(range(circuit.num_qubits))
        if analysis == "default":
            analysis = TomographyAnalysis()
        super().__init__(physical_qubits, analysis=analysis, backend=backend)

        # Get the target tomography circuit
        if isinstance(circuit, QuantumCircuit):
            target_circuit = circuit
        else:
            # Convert input to a circuit
            num_qubits = circuit.num_qubits
            target_circuit = QuantumCircuit(num_qubits)
            target_circuit.append(circuit, range(num_qubits))
        self._circuit = target_circuit

        self._cond_clbits = None
        if conditional_circuit_clbits is True:
            conditional_circuit_clbits = self._circuit.clbits
        if conditional_circuit_clbits:
            cond_clbits = []
            for i in conditional_circuit_clbits:
                if isinstance(i, Clbit):
                    cond_clbits.append(self._circuit.find_bit(i).index)
                elif i < self._circuit.num_clbits:
                    cond_clbits.append(i)
                else:
                    raise QiskitError(
                        f"Circuit {self._circuit.name} does not contain conditional clbit {i}"
                    )
            self._cond_clbits = cond_clbits

        # Measurement basis and qubits
        self._meas_circ_basis = measurement_basis
        if measurement_indices:
            # Convert logical qubits to physical qubits
            self._meas_indices = tuple(measurement_indices)
            self._meas_physical_qubits = tuple(self.physical_qubits[i] for i in self._meas_indices)
            for qubit in self._meas_indices:
                if qubit not in range(self.num_qubits):
                    raise QiskitError(
                        f"measurement qubit ({qubit}) is outside the range"
                        f" of circuit qubits [0, {self.num_qubits})."
                    )
        elif measurement_basis:
            self._meas_indices = tuple(range(self.num_qubits))
            self._meas_physical_qubits = self.physical_qubits
        else:
            self._meas_indices = tuple()
            self._meas_physical_qubits = tuple()

        # Preparation basis and qubits
        self._prep_circ_basis = preparation_basis
        if preparation_indices:
            self._prep_indices = tuple(preparation_indices)
            self._prep_physical_qubits = tuple(self.physical_qubits[i] for i in self._prep_indices)
            for qubit in self._prep_indices:
                if qubit not in range(self.num_qubits):
                    raise QiskitError(
                        f"preparation qubit ({qubit}) is outside the range"
                        f" of circuit qubits [0, {self.num_qubits})."
                    )
        elif preparation_basis:
            self._prep_indices = tuple(range(self.num_qubits))
            self._prep_physical_qubits = self.physical_qubits
        else:
            self._prep_indices = tuple()
            self._prep_physical_qubits = tuple()

        # Configure experiment options
        if basis_indices:
            self.set_experiment_options(basis_indices=basis_indices)

        # Configure analysis basis options
        if isinstance(self.analysis, TomographyAnalysis):
            analysis_options = {}
            if measurement_basis:
                analysis_options["measurement_basis"] = measurement_basis
                analysis_options["measurement_qubits"] = self._meas_physical_qubits
            if preparation_basis:
                analysis_options["preparation_basis"] = preparation_basis
                analysis_options["preparation_qubits"] = self._prep_physical_qubits
            if conditional_circuit_clbits:
                analysis_options["conditional_circuit_clbits"] = self._cond_clbits
            self.analysis.set_options(**analysis_options)

    def circuits(self):
        circ_qubits = self._circuit.qubits
        circ_clbits = self._circuit.clbits
        meas_creg = ClassicalRegister((len(self._meas_indices)), name="c_tomo")
        template = QuantumCircuit(
            *self._circuit.qregs, *self._circuit.cregs, meas_creg, name=f"{self._type}"
        )
        if self._circuit.metadata:
            template.metadata = self._circuit.metadata.copy()
        else:
            template.metadata = {}
        meas_clbits = [template.find_bit(i).index for i in meas_creg]

        # Build circuits
        circuits = []
        for prep_element, meas_element in self._basis_indices():
            name = template.name
            metadata = {"clbits": meas_clbits, "cond_clbits": self._cond_clbits}
            if meas_element:
                name += f"_{meas_element}"
                metadata["m_idx"] = list(meas_element)
            if prep_element:
                name += f"_{prep_element}"
                metadata["p_idx"] = list(prep_element)

            circ = template.copy(name=name)

            if prep_element:
                # Add tomography preparation
                prep_circ = self._prep_circ_basis.circuit(prep_element, self._prep_physical_qubits)
                circ.compose(prep_circ, self._prep_indices, inplace=True)
                circ.barrier(*self._prep_indices)

            # Add target circuit
            # Have to use compose since circuit.to_instruction has a bug
            # when circuit contains classical registers and conditionals
            circ.compose(self._circuit, circ_qubits, circ_clbits, inplace=True)

            # Add tomography measurement
            if meas_element:
                meas_circ = self._meas_circ_basis.circuit(meas_element, self._meas_physical_qubits)
                circ.barrier(*self._meas_indices)
                circ.compose(meas_circ, self._meas_indices, meas_clbits, inplace=True)

            # Add metadata
            circ.metadata.update(**metadata)
            circuits.append(circ)
        return circuits

    def _metadata(self):
        metadata = super()._metadata()
        if self._meas_physical_qubits:
            metadata["m_qubits"] = list(self._meas_physical_qubits)
        if self._prep_physical_qubits:
            metadata["p_qubits"] = list(self._prep_physical_qubits)
        return metadata

    def _basis_indices(self):
        """Return list of basis element indices"""
        basis_indices = self.experiment_options.basis_indices
        if basis_indices is not None:
            return basis_indices

        if self._meas_circ_basis:
            meas_shape = self._meas_circ_basis.index_shape(self._meas_physical_qubits)
            ranges = [range(i) for i in meas_shape]
            meas_elements = product(*ranges)
        else:
            meas_elements = [None]
        if self._prep_circ_basis:
            prep_shape = self._prep_circ_basis.index_shape(self._prep_physical_qubits)
            prep_elements = product(*[range(i) for i in prep_shape])
        else:
            prep_elements = [None]
        return product(prep_elements, meas_elements)

    def _permute_circuit(self) -> QuantumCircuit:
        """Permute circuit qubits.

        This permutes the circuit so that the specified preparation and measurement
        qubits correspond to input and output qubits [0, ..., N-1] and [0, ..., M-1]
        respectively for the returned circuit.
        """
        default_range = tuple(range(self.num_qubits))
        permute_meas = self._meas_indices and self._meas_indices != default_range
        permute_prep = self._prep_indices and self._prep_indices != default_range
        if not permute_meas and not permute_prep:
            return self._circuit

        total_qubits = self._circuit.num_qubits
        total_clbits = self._circuit.num_clbits
        if total_clbits:
            perm_circ = QuantumCircuit(total_qubits, total_clbits)
        else:
            perm_circ = QuantumCircuit(total_qubits)

        # Apply permutation to put prep qubits as [0, ..., M-1]
        if self._prep_indices:
            prep_qargs = list(self._prep_indices)
            if len(self._prep_indices) != total_qubits:
                prep_qargs += [i for i in range(total_qubits) if i not in self._prep_indices]
            perm_circ.append(PermutationGate(prep_qargs).inverse(), range(total_qubits))

        # Apply original circuit
        if total_clbits:
            perm_circ = perm_circ.compose(self._circuit, range(total_qubits), range(total_clbits))
        else:
            perm_circ = perm_circ.compose(self._circuit, range(total_qubits))

        # Apply permutation to put meas qubits as [0, ..., M-1]
        if self._meas_indices:
            meas_qargs = list(self._meas_indices)
            if len(self._meas_indices) != total_qubits:
                meas_qargs += [i for i in range(total_qubits) if i not in self._meas_indices]
            perm_circ.append(PermutationGate(meas_qargs), range(total_qubits))

        return perm_circ
