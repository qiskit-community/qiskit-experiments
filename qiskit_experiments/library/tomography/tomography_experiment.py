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
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.circuit.library import Permutation
from qiskit.providers.backend import Backend
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_experiments.exceptions import QiskitError
from qiskit_experiments.framework import BaseExperiment, Options
from .basis import BaseTomographyMeasurementBasis, BaseTomographyPreparationBasis
from .tomography_analysis import TomographyAnalysis


class TomographyExperiment(BaseExperiment):
    """Base experiment for quantum state and process tomography.

    # section: analysis_ref
        :py:class:`TomographyAnalysis`
    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            measurement_basis (:class:`~basis.BaseTomographyMeasurementBasis`): The
                Tomography measurement basis to use for the experiment.
                The default basis is the :class:`~basis.PauliMeasurementBasis` which
                performs measurements in the Pauli Z, X, Y bases for each qubit
                measurement.

        """
        options = super()._default_experiment_options()

        options.basis_indices = None

        return options

    def __init__(
        self,
        circuit: Union[QuantumCircuit, Instruction, BaseOperator],
        backend: Optional[Backend] = None,
        measurement_basis: Optional[BaseTomographyMeasurementBasis] = None,
        measurement_qubits: Optional[Sequence[int]] = None,
        preparation_basis: Optional[BaseTomographyPreparationBasis] = None,
        preparation_qubits: Optional[Sequence[int]] = None,
        basis_indices: Optional[Iterable[Tuple[List[int], List[int]]]] = None,
        qubits: Optional[Sequence[int]] = None,
        analysis: Optional[TomographyAnalysis] = None,
    ):
        """Initialize a tomography experiment.

        Args:
            circuit: the quantum process circuit. If not a quantum circuit
                it must be a class that can be appended to a quantum circuit.
            backend: The backend to run the experiment on.
            measurement_basis: Tomography basis for measurements.
            measurement_qubits: Optional, the qubits to be measured. These should refer
                to the logical qubits in the state circuit.
            preparation_basis: Tomography basis for measurements.
            preparation_qubits: Optional, the qubits to be prepared. These should refer
                to the logical qubits in the process circuit.
            basis_indices: Optional, the basis elements to be measured. If None
                All basis elements will be measured.
            qubits: Optional, the physical qubits for the initial state circuit.
            analysis: Optional, analysis class to use for experiment. If None the default
                tomography analysis will be used.

        Raises:
            QiskitError: if input params are invalid.
        """
        # Initialize BaseExperiment
        if qubits is None:
            qubits = range(circuit.num_qubits)
        if analysis is None:
            analysis = TomographyAnalysis()
        super().__init__(qubits, analysis=analysis, backend=backend)

        # Get the target tomography circuit
        if isinstance(circuit, QuantumCircuit):
            target_circuit = circuit
        else:
            # Convert input to a circuit
            num_qubits = circuit.num_qubits
            target_circuit = QuantumCircuit(num_qubits)
            target_circuit.append(circuit, range(num_qubits))
        self._circuit = target_circuit

        # Measurement basis and qubits
        self._meas_circ_basis = measurement_basis
        if measurement_qubits:
            self._meas_qubits = tuple(measurement_qubits)
            for qubit in self._meas_qubits:
                if qubit not in range(self.num_qubits):
                    raise QiskitError(
                        f"measurement qubit ({qubit}) is outside the range"
                        f" of circuit qubits [0, {self.num_qubits})."
                    )
        else:
            self._meas_qubits = None

        # Preparation basis and qubits
        self._prep_circ_basis = preparation_basis
        if preparation_qubits:
            self._prep_qubits = tuple(preparation_qubits)
            for qubit in self._prep_qubits:
                if qubit not in range(self.num_qubits):
                    raise QiskitError(
                        f"preparation qubit ({qubit}) is outside the range"
                        f" of circuit qubits [0, {self.num_qubits})."
                    )
        else:
            self._prep_qubits = None

        # Configure experiment options
        if basis_indices:
            self.set_experiment_options(basis_indices=basis_indices)

        # Configure analysis basis options
        analysis_options = {}
        if measurement_basis:
            analysis_options["measurement_basis"] = measurement_basis
        if preparation_basis:
            analysis_options["preparation_basis"] = preparation_basis
        self.analysis.set_options(**analysis_options)

    def circuits(self):

        # Get qubits and clbits
        meas_qubits = self._meas_qubits or range(self.num_qubits)
        total_clbits = self._circuit.num_clbits + len(meas_qubits)
        circ_qubits = list(range(self._circuit.num_qubits))
        circ_clbits = list(range(self._circuit.num_clbits))
        meas_clbits = list(range(self._circuit.num_clbits, total_clbits))

        # Build circuits
        circuits = []
        for prep_element, meas_element in self._basis_indices():
            name = f"{self._type}_{meas_element}"
            metadata = {
                "experiment_type": self._type,
                "clbits": meas_clbits,
                "m_idx": list(meas_element),
            }
            if prep_element:
                name += f"_{prep_element}"
                metadata["p_idx"] = list(prep_element)

            circ = QuantumCircuit(self.num_qubits, total_clbits, name=name)

            if prep_element:
                # Add tomography preparation
                prep_qubits = self._prep_qubits or range(self.num_qubits)
                prep_circ = self._prep_circ_basis.circuit(prep_element)
                circ.reset(prep_qubits)
                circ.append(prep_circ, prep_qubits)
                circ.barrier(prep_qubits)

            # Add target circuit
            # Have to use compose since circuit.to_instruction has a bug
            # when circuit contains classical registers and conditionals
            circ = circ.compose(self._circuit, circ_qubits, circ_clbits)

            # Add tomography measurement
            meas_circ = self._meas_circ_basis.circuit(meas_element)
            circ.barrier(meas_qubits)
            circ.append(meas_circ, meas_qubits)
            circ.measure(meas_qubits, meas_clbits)

            # Add metadata
            circ.metadata = metadata
            circuits.append(circ)
        return circuits

    def _basis_indices(self):
        """Return list of basis element indices"""
        basis_indices = self.experiment_options.basis_indices
        if basis_indices is not None:
            return basis_indices
        if self._meas_circ_basis:
            meas_size = len(self._meas_circ_basis)
            num_meas = len(self._meas_qubits) if self._meas_qubits else self.num_qubits
            meas_elements = product(range(meas_size), repeat=num_meas)
        else:
            meas_elements = [None]
        if self._prep_circ_basis:
            prep_size = len(self._prep_circ_basis)
            num_prep = len(self._prep_qubits) if self._prep_qubits else self.num_qubits
            prep_elements = product(range(prep_size), repeat=num_prep)
        else:
            prep_elements = [None]

        return product(prep_elements, meas_elements)

    def _permute_circuit(self) -> QuantumCircuit:
        """Permute circuit qubits.

        This permutes the circuit so that the specified preparation and measurement
        qubits correspond to input and output qubits [0, ..., N-1] respectively
        for the returned circuit.
        """
        if self._meas_qubits is None and self._prep_qubits is None:
            return self._circuit

        total_qubits = self._circuit.num_qubits
        total_clbits = self._circuit.num_clbits
        if total_clbits:
            perm_circ = QuantumCircuit(total_qubits, total_clbits)
        else:
            perm_circ = QuantumCircuit(total_qubits)

        # Apply permutation to put prep qubits as [0, ..., M-1]
        if self._prep_qubits:
            prep_qargs = list(self._prep_qubits)
            if len(self._prep_qubits) != total_qubits:
                prep_qargs += [i for i in range(total_qubits) if i not in self._prep_qubits]
            perm_circ.append(Permutation(total_qubits, prep_qargs).inverse(), range(total_qubits))

        # Apply original circuit
        if total_clbits:
            perm_circ = perm_circ.compose(self._circuit, range(total_qubits), range(total_clbits))
        else:
            perm_circ = perm_circ.compose(self._circuit, range(total_qubits))

        # Apply permutation to put meas qubits as [0, ..., M-1]
        if self._meas_qubits:
            meas_qargs = list(self._meas_qubits)
            if len(self._meas_qubits) != total_qubits:
                meas_qargs += [i for i in range(total_qubits) if i not in self._meas_qubits]
            perm_circ.append(Permutation(total_qubits, meas_qargs), range(total_qubits))

        return perm_circ
