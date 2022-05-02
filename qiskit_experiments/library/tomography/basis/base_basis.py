# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Fitter basis classes for tomography analysis.
"""
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Optional
import numpy as np
from qiskit import QuantumCircuit


class BaseBasis(ABC):
    """Abstract base class for a measurement and preparation bases."""

    def __init__(self, name: str):
        """Initialize a basis.

        Args:
            name: the name for the basis.
        """
        self._name = name

    def __hash__(self):
        return hash((type(self), self._name))

    def __eq__(self, value):
        tup1 = (type(self), self.name)
        tup2 = (type(value), getattr(value, "name", None))
        return tup1 == tup2

    @property
    def name(self) -> str:
        """Return the basis name"""
        return self._name

    @abstractmethod
    def index_shape(self, qubits: Sequence[int]) -> Tuple[int, ...]:
        """Return the shape for the specified number of indices.

        Args:
            qubits: the basis subsystems to return the index shape for.

        Returns:
            The shape of allowed values for the index on the specified qubits.
        """

    @abstractmethod
    def circuit(
        self, index: Sequence[int], qubits: Optional[Sequence[int]] = None
    ) -> QuantumCircuit:
        """Return the basis preparation circuit.

        Args:
            index: a list of basis elements to tensor together.
            qubits: Optional, the physical qubit subsystems for the index.
                    If None this will be set to ``(0, ..., N-1)`` for a
                    length N index.

        Returns:
            The logical basis circuit for the specified index and qubits.

        .. note::

            This returns a logical circuit on the specified number of qubits
            and should be remapped to the corresponding physical qubits
            during experiment transpilation.
        """


class PreparationBasis(BaseBasis):
    """Abstract base class for a tomography preparation basis.

    Subclasses should implement the following abstract methods to
    define a preparation basis:

    * The :meth:`circuit` method which returns the logical preparation
      :class:`.QuantumCircuit` for basis element index on the specified
      qubits. This circuit should be a logical circuit on the specified
      number of qubits and will be remapped to the corresponding physical
      qubits during transpilation.

    * The :meth:`matrix` method which returns the density matrix prepared
      by the bases element index on the specified qubits.

    * The :meth:`index_shape` method which returns the shape of allowed
      basis indices for the specified qubits, and their values.

    * The :meth:`matrix_shape` method which returns the shape of subsystem
      dimensions of the density matrix state on the specified qubits.
    """

    @abstractmethod
    def matrix_shape(self, qubits: Sequence[int]) -> Tuple[int, ...]:
        """Return the shape of subsystem dimensions of a matrix element."""

    @abstractmethod
    def matrix(self, index: Sequence[int], qubits: Optional[Sequence[int]] = None) -> np.ndarray:
        """Return the density matrix data array for the index and qubits.

        This state is used by tomography fitters for reconstruction and should
        correspond to the target state for the corresponding preparation
        :meth:`circuit`.

        Args:
            index: a list of subsystem basis indices.
            qubits: Optional, the physical qubit subsystems for the index.
                    If None this will be set to ``(0, ..., N-1)`` for a
                    length N index.

        Returns:
            The density matrix prepared by the specified index and qubits.
        """


class MeasurementBasis(BaseBasis):
    """Abstract base class for a tomography measurement basis.

    Subclasses should implement the following abstract methods to
    define a preparation basis:

    * The :meth:`circuit` method which returns the logical measurement
      :class:`.QuantumCircuit` for basis element index on the specified
      physical qubits. This circuit should be a logical circuit on the
      specified number of qubits and will be remapped to the corresponding
      physical qubits during transpilation. It should include classical
      bits and the measure instructions for the basis measurement storing
      the outcome value in these bits.

    * The :meth:`matrix` method which returns the POVM element corresponding
      to the basis element index and measurement outcome on the specified
      qubits. This should return either a :class:`.Statevector` for a PVM
      element, or :class:`.DensityMatrix` for a general POVM element.

    * The :meth:`index_shape` method which returns the shape of allowed
      basis indices for the specified qubits, and their values.

    * The :meth:`matrix_shape` method which returns the shape of subsystem
      dimensions of the POVM element matrices on the specified qubits.

    * The :meth:`outcome_shape` method which returns the shape of allowed
      outcome values for a measurement of specified qubits.
    """

    @abstractmethod
    def outcome_shape(self, qubits: Sequence[int]) -> Tuple[int, ...]:
        """Return the shape of allowed measurement outcomes on specified qubits."""

    @abstractmethod
    def matrix_shape(self, qubits: Sequence[int]) -> Tuple[int, ...]:
        """Return the shape of subsystem dimensions of a POVM matrix element."""

    @abstractmethod
    def matrix(
        self, index: Sequence[int], outcome: int, qubits: Optional[Sequence[int]] = None
    ) -> np.ndarray:
        """Return the POVM element for the basis index and outcome.

        This POVM element is used by tomography fitters for reconstruction and
        should correspond to the target measurement effect for the corresponding
        measurement :meth:`circuit` and outcome.

        Args:
            index: a list of subsystem basis indices.
            outcome: the composite system measurement outcome.
            qubits: Optional, the physical qubit subsystems for the index.
                    If None this will be set to ``(0, ..., N-1)`` for a
                    length N index.

        Returns:
            The POVM matrix for the specified index and qubits.
        """
