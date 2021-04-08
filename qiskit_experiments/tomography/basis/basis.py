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
from typing import Iterable, Optional, List
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.quantum_info import Operator
from qiskit.exceptions import QiskitError


class TomographyBasis:
    """Basis class for tomography experiments"""

    def __init__(
        self,
        circuit_basis: "CircuitBasis",
        matrix_basis: "FitterBasis",
        name: Optional[str] = None,
    ):
        """Initialize a tomography basis"""
        self._circuit_basis = circuit_basis
        self._matrix_basis = matrix_basis
        self._name = name
        self._hash = hash(
            (type(self), self._name, hash(self._circuit_basis), hash(self._matrix_basis))
        )

    def __hash__(self):
        return self._hash

    @property
    def name(self) -> str:
        """Return the basis name"""
        return self._name

    @property
    def circuit(self):
        """Return circuit basis"""
        return self._circuit_basis

    @property
    def matrix(self):
        """Return matrix basis"""
        return self._matrix_basis


class CircuitBasis:
    """Circuit generation for tensor-product tomography bases."""

    def __init__(
        self,
        instructions: List[Instruction],
        num_outcomes: int = 1,
        product_basis: bool = True,
        name: Optional[str] = None,
    ):
        """Initialize a circuit generator.

        Args:
            instructions: list of instructions for basis rotations.
            num_outcomes: the number of outcomes for each basis element.
            product_basis: If True the input instructions will be used as
                           a subsystem basis for a tensor product system.
                           If False the input.
            name: Optional, name for the basis. If None the class
                  name will be used.

        Raises:
            QiskitError: if input instructions are not valid.
        """
        # Convert inputs to quantum circuits
        self._instructions = [self._convert_input(i) for i in instructions]
        self._num_outcomes = num_outcomes
        self._name = name if name else type(self).__name__

        # Check number of qubits
        self._num_qubits = self._instructions[0].num_qubits
        for i in self._instructions[1:]:
            if i.num_qubits != self._num_qubits:
                raise QiskitError(
                    "Invalid input instructions. All instructions must be"
                    " defined on the same number of qubits."
                )
        # Make basis hashable
        self._hash = hash((type(self), self._name, self._num_outcomes, self._num_qubits))

    def __hash__(self):
        return self._hash

    @property
    def name(self) -> str:
        """Return the basis name"""
        return self._name

    @property
    def num_outcomes(self) -> int:
        """Return the number of outcomes"""
        return self._num_outcomes

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits of the basis"""
        return self._num_qubits

    def __len__(self) -> int:
        return len(self._instructions)

    def __call__(self, element: Iterable[int]) -> QuantumCircuit:
        """Return a composite basis rotation circuit.

        Args:
            element: a list of basis elements to tensor together.

        Returns:
            the rotation circuit for the specified basis

        Raises:
            QiskitError: if the specified elements are invalid.
        """
        total_qubits = len(element) * self.num_qubits
        circuit = QuantumCircuit(total_qubits, name=f"{self._name}_{element}")
        for i, elt in enumerate(element):
            if elt >= len(self):
                raise QiskitError("Invalid basis element index")
            qubits = list(range(i * self.num_qubits, (i + 1) * self.num_qubits))
            circuit.append(self._instructions[elt], qubits)
        return circuit

    @staticmethod
    def _convert_input(unitary):
        """Convert input to an Instruction"""
        if isinstance(unitary, Instruction):
            return unitary
        if hasattr(unitary, "to_instruction"):
            return unitary.to_instruction()
        return Operator(unitary).to_instruction()


class FitterBasis:
    """A operator basis for tomography fitters"""

    def __init__(
        self,
        elements: np.ndarray,
        num_indices: int = 1,
        num_outcomes: int = 1,
        name: Optional[str] = None,
    ):
        """Initialize a matrix basis generator.

        Args:
            elements: array of element matrices
            num_indices: Optional, custom indexing of the specified basis
                             elements.
            num_outcomes: the number of outcomes for each circuit basis element.
            name: Optional, name for the basis. If None the class
                  name will be used.

        Raises:
            QiskitError: If the number of elements and number of indices are
                         incompatible.
        """
        self._num_indices = num_indices
        self._num_outcomes = num_outcomes
        num_elts = len(elements)
        if num_indices == 1:
            self._call_fn = self._call_tensor
            self._size = num_elts
        else:
            self._call_fn = self._call_multi
            num_sub = round(num_elts ** (1 / num_indices))
            if num_sub ** num_sub != num_elts:
                raise QiskitError("Invalid number of elements for element indicies")
            self._size = num_sub
        self._name = name if name else type(self).__name__
        self._elements = np.asarray(elements, dtype=complex)

        # Make basis hashable
        self._hash = hash((type(self), self._name, self._num_indices, self._num_outcomes))

    def __hash__(self):
        return self._hash

    @property
    def name(self) -> str:
        """Return the basis name"""
        return self._name

    @property
    def num_indices(self) -> int:
        """Return the number of indices for each stored basis matrix"""
        return self._num_indices

    @property
    def num_outcomes(self) -> int:
        """Return the number of outcomes"""
        return self._num_outcomes

    def __len__(self) -> int:
        """Return element index size"""
        return self._size

    def __call__(self, element: Iterable[int]) -> np.ndarray:
        """Return a basis rotation circuit"""
        return self._call_fn(element)

    def _call_tensor(self, element: Iterable[int]) -> np.ndarray:
        """Single subsystem tensor product call function"""
        ret = self._elements[element[0]]
        for i in element[1:]:
            ret = np.kron(self._elements[i], ret)
        return ret

    def _call_multi(self, element: Iterable[int]) -> np.ndarray:
        """Multi-subsystem call function"""
        # TODO: Check order definition and see if we need to
        # reverse element
        index = 0
        for i, elt in enumerate(element):
            index += (self._size ** i) * elt
        return self._elements[index]
