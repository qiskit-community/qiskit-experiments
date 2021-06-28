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
Fitter basis classes for tomography analysis.
"""
from abc import ABC, abstractmethod
from typing import Iterable, Optional
import numpy as np
from qiskit import QuantumCircuit


class BaseFitterMeasurementBasis(ABC):
    """Abstract base class for fitter measurement bases.

    This class defines the POVM element matrix for each measurement
    outcome of a basis index. It is used by tomography fitters during
    tomography analysis.
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize a fitter measurement basis.

        Args:
            name: Optional, name for the basis. If None the class
                  name will be used.
        """
        self._name = name if name else type(self).__name__

    def __hash__(self):
        return hash((type(self), self._name))

    @property
    def name(self) -> str:
        """Return the basis name"""
        return self._name

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of indices for subsystems."""

    @abstractmethod
    def num_outcomes(self, index: Iterable[int]) -> int:
        """Return the number of outcomes for basis index"""

    @abstractmethod
    def matrix(self, index: Iterable[int], outcome: int) -> np.ndarray:
        """Return the POVM matrix for the basis index and outcome.

        Args:
            index: a list of subsystem basis indices.
            outcome: the composite system count outcome.

        Returns:
            The POVM matrix for the bases and outcome.
        """


class BaseTomographyMeasurementBasis(BaseFitterMeasurementBasis):
    """Abstract base class for tomography experiment measurement bases.

    This class extends BaseFitterMeasurementBasis to include a
    circuit definition of each basis index. It can be used to construct
    tomography circuits for execution, as well as for tomography analysis
    of measurement data.
    """

    @abstractmethod
    def circuit(self, index: Iterable[int]) -> QuantumCircuit:
        """Return a composite rotation circuit to measure in basis

        Args:
            index: a list of basis elements to tensor together.

        Returns:
            the rotation circuit for the specified basis
        """


class BaseFitterPreparationBasis(ABC):
    """Abstract base class for fitter preparation bases.

    This class defines the density matrix for each preparation basis
    index. It is used by tomography fitters during tomography analysis.
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize a fitter preparation basis.

        Args:
            name: Optional, name for the basis. If None the class
                  name will be used.
        """
        self._name = name if name else type(self).__name__

    def __hash__(self):
        return hash((type(self), self._name))

    @property
    def name(self) -> str:
        """Return the basis name"""
        return self._name

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of indices for subsystems."""

    @abstractmethod
    def matrix(self, index: Iterable[int]) -> np.ndarray:
        """Return the density matrix for the basis index and outcome.

        Args:
            index: a list of subsystem basis indices.

        Returns:
            The density matrix for the bases and outcome.
        """


class BaseTomographyPreparationBasis(BaseFitterPreparationBasis):
    """Abstract base class for tomography experiment basispreparation bases.

    This class extends BaseFitterPreparationBasis to include a
    circuit definition of each basis index. It can be used to construct
    tomography circuits for execution, as well as for tomography analysis
    of measurement data.
    """

    @abstractmethod
    def circuit(self, index: Iterable[int]) -> QuantumCircuit:
        """Return the basis preparation circuit.

        Args:
            index: a list of basis elements to tensor together.

        Returns:
            the rotation circuit for the specified basis
        """
