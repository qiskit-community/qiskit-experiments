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

"""Results of a fit."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FitResult:
    """
    A data class that store fitting results.

    fitvals and chisq are required while the other parameters are optional.
    """

    # fit parameters
    fitvals: np.ndarray

    # chi squared value of fitting
    chisq: float

    # standard deviation of parameters
    stdevs: np.ndarray = np.array([])

    # horizontal axis data values, may be used for visualization
    xvals: np.ndarray = np.array([])

    # vertical axis data values, may be used for visualization
    yvals: np.ndarray = np.array([])

    def __repr__(self) -> str:
        return 'FitResult({})'.format(','.join(map(str, self.fitvals)))
