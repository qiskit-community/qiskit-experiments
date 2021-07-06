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
Circuit basis for tomography preparation and measurement circuits
"""
from typing import Iterable, Optional, List, Dict
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.quantum_info import Operator, DensityMatrix
from qiskit.exceptions import QiskitError
from .fitter_basis import FitterMeasurementBasis, FitterPreparationBasis
from .base_basis import BaseTomographyMeasurementBasis, BaseTomographyPreparationBasis


class TomographyMeasurementBasis(FitterMeasurementBasis, BaseTomographyMeasurementBasis):
    """Measurement basis for tomography experiments."""

    def __init__(
        self,
        instructions: List[Instruction],
        name: Optional[str] = None,
    ):
        """Initialize tomography measurement basis.

        Args:
            instructions: list of instructions for basis rotations.
            name: Optional, name for the basis. If None the class
                  name will be used.

        Raises:
            QiskitError: if input instructions are not valid.
        """
        self._instructions = [_convert_instruction(i) for i in instructions]
        povms = _instruction_povms(self._instructions)
        super().__init__(povms, name=name)

    def circuit(self, index: Iterable[int]) -> QuantumCircuit:
        return _product_circuit(self._instructions, index, self._name)


class TomographyPreparationBasis(FitterPreparationBasis, BaseTomographyPreparationBasis):
    """Preparation basis for tomography experiments."""

    def __init__(
        self,
        instructions: List[Instruction],
        name: Optional[str] = None,
    ):
        """Initialize tomography measurement basis.

        Args:
            instructions: list of instructions for basis rotations.
            name: Optional, name for the basis. If None the class
                  name will be used.

        Raises:
            QiskitError: if input instructions are not valid.
        """
        self._instructions = [_convert_instruction(i) for i in instructions]
        states = np.asarray(_instruction_states(self._instructions))
        super().__init__(states, name=name)

    def circuit(self, index: Iterable[int]) -> QuantumCircuit:
        return _product_circuit(self._instructions, index, self._name)


def _instruction_povms(instructions: List[Instruction]) -> List[Dict[int, np.ndarray]]:
    """Construct measurement outcome POVMs from instructions"""
    basis = []
    for inst in instructions:
        inst_inv = inst.inverse()
        basis_dict = {
            i: DensityMatrix.from_int(i, 2 ** inst.num_qubits).evolve(inst_inv).data
            for i in range(2 ** inst.num_qubits)
        }
        basis.append(basis_dict)
    return basis


def _instruction_states(instructions: List[Instruction]) -> List[np.ndarray]:
    """Construct preparation density matrices from instructions"""
    states = []
    num_qubits = instructions[0].num_qubits
    init = DensityMatrix.from_int(0, 2 ** num_qubits)
    for inst in instructions:
        states.append(init.evolve(inst).data)
    return states


def _convert_instruction(unitary):
    """Convert input to an Instruction"""
    if isinstance(unitary, Instruction):
        return unitary
    if hasattr(unitary, "to_instruction"):
        return unitary.to_instruction()
    return Operator(unitary).to_instruction()


def _product_circuit(
    instructions: List[Instruction], element: Iterable[int], name: str = ""
) -> QuantumCircuit:
    """Return a composite basis circuit."""
    num_qubits = instructions[0].num_qubits
    total_qubits = len(element) * num_qubits
    circuit = QuantumCircuit(total_qubits, name=f"{name}_{element}")
    for i, elt in enumerate(element):
        if elt >= len(instructions):
            raise QiskitError("Invalid basis element index")
        qubits = list(range(i * num_qubits, (i + 1) * num_qubits))
        circuit.append(instructions[elt], qubits)
    return circuit
