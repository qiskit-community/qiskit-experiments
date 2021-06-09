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

from typing import Union, Optional, Iterable, List, Tuple
from itertools import product
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
import qiskit.quantum_info as qi
from qiskit_experiments.base_experiment import BaseExperiment, Options
from .basis import BaseTomographyMeasurementBasis, BaseTomographyPreparationBasis
from .tomography_analysis import TomographyAnalysis


class TomographyExperiment(BaseExperiment):
    """Base experiment for quantum state and process tomography"""

    __analysis_class__ = TomographyAnalysis

    @classmethod
    def _default_experiment_options(cls):
        return Options(basis_elements=None)

    def __init__(
        self,
        circuit: Union[QuantumCircuit, Instruction, BaseOperator],
        measurement_basis: Optional[BaseTomographyMeasurementBasis] = None,
        measurement_qubits: Optional[Iterable[int]] = None,
        preparation_basis: Optional[BaseTomographyPreparationBasis] = None,
        preparation_qubits: Optional[Iterable[int]] = None,
        basis_elements: Optional[Iterable[Tuple[List[int], List[int]]]] = None,
        qubits: Optional[Iterable[int]] = None,
    ):
        """Initialize a state tomography experiment.

        Args:
            circuit: the quantum process circuit. If not a quantum circuit
                it must be a class that can be appended to a quantum circuit.
            measurement_basis: Tomography basis for measurements.
            measurement_qubits: Optional, the qubits to be measured. These should refer
                to the logical qubits in the state circuit.
            preparation_basis: Tomography basis for measurements.
            preparation_qubits: Optional, the qubits to be prepared. These should refer
                to the logical qubits in the process circuit.
            basis_elements: Optional, the basis elements to be measured. If None
                All basis elements will be measured.
            qubits: Optional, the physical qubits for the initial state circuit.
        """
        # Initialize BaseExperiment
        if qubits is None:
            qubits = circuit.num_qubits
        super().__init__(qubits)

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
        else:
            self._meas_qubits = None

        # Preparation basis and qubits
        self._prep_circ_basis = preparation_basis
        if preparation_qubits:
            self._prep_qubits = tuple(preparation_qubits)
        else:
            self._prep_qubits = None

        # Configure experiment options
        if basis_elements:
            self.set_experiment_options(basis_elements=basis_elements)

        # Compute target state
        # NOTE: this is only implemented if measurement_qubits and
        # preparation_qubits init kwargs are not used.
        if not self._prep_qubits and not self._meas_qubits:
            if self._prep_circ_basis:
                try:
                    self._target_state = qi.Clifford(self._circuit)
                except QiskitError:
                    self._target_state = qi.Operator(self._circuit)
            else:
                # TODO: add support for StabilizerState for Clifford
                # circuits once state_fidelity works with that class
                self._target_state = qi.Statevector(self._circuit)

        # Configure analysis basis options
        analysis_options = {}
        if measurement_basis:
            analysis_options["measurement_basis"] = measurement_basis
        if preparation_basis:
            analysis_options["preparation_basis"] = preparation_basis
        self.set_analysis_options(**analysis_options)

    def _metadata(self):
        metadata = super()._metadata()
        metadata["target_state"] = self._target_state.copy()
        return metadata

    def circuits(self, backend=None):

        # Get qubits and clbits
        meas_qubits = self._meas_qubits or range(self.num_qubits)
        total_clbits = self._circuit.num_clbits + len(meas_qubits)
        circ_qubits = list(range(self._circuit.num_qubits))
        circ_clbits = list(range(self._circuit.num_clbits))
        meas_clbits = list(range(self._circuit.num_clbits, total_clbits))

        # Build circuits
        circuits = []
        for meas_element, prep_element in self._basis_elements():
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
            circ.append(self._circuit, circ_qubits, circ_clbits)

            # Add tomography measurement
            meas_circ = self._meas_circ_basis.circuit(meas_element)
            circ.barrier(meas_qubits)
            circ.append(meas_circ, meas_qubits)
            circ.measure(meas_qubits, meas_clbits)

            # Add metadata
            circ.metadata = metadata
            circuits.append(circ)
        return circuits

    def _basis_elements(self):
        """Return list of basis element indices"""
        basis_elements = self.experiment_options.basis_elements
        if basis_elements is not None:
            return basis_elements

        meas_size = len(self._meas_circ_basis)
        num_meas = len(self._meas_qubits) if self._meas_qubits else self.num_qubits
        meas_elements = product(range(meas_size), repeat=num_meas)
        if self._prep_circ_basis:
            prep_size = len(self._prep_circ_basis)
            num_prep = len(self._prep_qubits) if self._prep_qubits else self.num_qubits
            prep_elements = product(range(prep_size), repeat=num_prep)
        else:
            prep_elements = [None]

        return product(meas_elements, prep_elements)
