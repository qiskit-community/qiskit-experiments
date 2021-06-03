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
Quantum State Tomography experiment
"""

from typing import Union, Optional, Iterable, List, Tuple
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.quantum_info.operators.base_operator import BaseOperator
from .basis import TomographyBasis, CircuitBasis, PauliMeasurementBasis
from .tomography_experiment import TomographyExperiment, Options


class StateTomographyExperiment(TomographyExperiment):
    """Quantum state tomography experiment"""

    @classmethod
    def _default_analysis_options(cls):
        return Options(measurement_basis=PauliMeasurementBasis().matrix)

    def __init__(
        self,
        circuit: Union[QuantumCircuit, Instruction, BaseOperator],
        measurement_basis: Union[TomographyBasis, CircuitBasis] = PauliMeasurementBasis(),
        measurement_qubits: Optional[Iterable[int]] = None,
        basis_elements: Optional[Iterable[Tuple[List[int], List[int]]]] = None,
        qubits: Optional[Iterable[int]] = None,
    ):
        """Initialize a quantum process tomography experiment.

        Args:
            circuit: the quantum process circuit. If not a quantum circuit
                it must be a class that can be appended to a quantum circuit.
            measurement_basis: Tomography basis for measurements. If not specified the
                default basis is the :class:`PauliMeasurementBasis`.
            measurement_qubits: Optional, the qubits to be measured. These should refer
                to the logical qubits in the state circuit. If None all qubits
                in the state circuit will be measured.
            basis_elements: Optional, the basis elements to be measured. If None
                All basis elements will be measured. If specified each element
                is given by a list ``[m[0], m[1], ...]`` where ``m[i]`` are the
                measurement basis indices for qubit-i.
            qubits: Optional, the physical qubits for the initial state circuit.
        """
        super().__init__(
            circuit,
            measurement_basis=measurement_basis,
            measurement_qubits=measurement_qubits,
            basis_elements=basis_elements,
            qubits=qubits,
        )
