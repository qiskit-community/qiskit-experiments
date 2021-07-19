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
from typing import Iterable, Optional, Dict
import numpy as np
from .base_basis import BaseFitterMeasurementBasis, BaseFitterPreparationBasis


class FitterMeasurementBasis(BaseFitterMeasurementBasis):
    """Measurement basis for tomography fitters"""

    def __init__(
        self,
        povms: Iterable[Dict[int, np.ndarray]],
        name: Optional[str] = None,
    ):
        """Initialize a matrix basis generator.

        Args:
            povms: array of element matrices.
            name: Optional, name for the basis. If None the class
                  name will be used.

        Raises:
            QiskitError: If the number of elements and number of indices are
                         incompatible.
        """
        super().__init__(name)
        self._size = len(povms)
        self._basis = []
        self._outcome_default = []
        self._num_outcomes = []
        for basis in povms:
            self._num_outcomes.append(len(basis))
            default_povm = basis.pop(None, None)
            self._basis.append(basis)
            self._outcome_default.append(default_povm)

    def __len__(self) -> int:
        """Return element index size"""
        return self._size

    def num_outcomes(self, index: Iterable[int]) -> int:
        num = 1
        for i in index:
            num *= self._num_outcomes[i]
        return num

    def matrix(self, index: Iterable[int], outcome: int) -> np.ndarray:
        oindex = self._outcome_index(outcome, len(index))
        ret = self._get_element(index[0], oindex[0])
        for i in range(1, len(index)):
            ret = np.kron(self._get_element(index[i], oindex[i]), ret)
        return ret

    def _get_element(self, basis_index: int, outcome_index: int):
        """Return an element for given basis and outcome index"""
        default_povm = self._outcome_default[basis_index]
        basis = self._basis[basis_index]
        return basis.get(outcome_index, default_povm)

    @staticmethod
    def _outcome_index(outcome: int, size: int):
        """Convert tensored outcome to subsystem outcomes"""
        index = [int(i) for i in reversed(bin(outcome)[2:])]
        return index + (size - len(index)) * [0]


class FitterPreparationBasis(BaseFitterPreparationBasis):
    """Preparation basis for tomography fitters"""

    def __init__(
        self,
        states: Iterable[np.ndarray],
        name: Optional[str] = None,
    ):
        """Initialize a matrix basis generator.

        Args:
            states: array of element matrices
            name: Optional, name for the basis. If None the class
                  name will be used.
        """
        super().__init__(name)
        self._size = len(states)
        self._mats = np.asarray(states, dtype=complex)

    def __len__(self) -> int:
        return self._size

    def matrix(self, index: Iterable[int]) -> np.ndarray:
        ret = self._mats[index[0]]
        for i in index[1:]:
            ret = np.kron(self._mats[i], ret)
        return ret
