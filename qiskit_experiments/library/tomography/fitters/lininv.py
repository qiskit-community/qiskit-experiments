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
Linear inversion MLEtomography fitter.
"""

from typing import Dict, List, Tuple, Optional
from functools import lru_cache
import numpy as np
from qiskit_experiments.library.tomography.basis.fitter_basis import (
    FitterMeasurementBasis,
    FitterPreparationBasis,
)
from . import fitter_utils


def linear_inversion(
    outcome_data: List[np.ndarray],
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: FitterMeasurementBasis,
    preparation_basis: Optional[FitterPreparationBasis] = None,
) -> Tuple[np.ndarray, Dict]:
    r"""Linear inversion tomography fitter.

    Overview
        This fitter uses linear inversion to reconstructs the maximum-likelihood
        estimate of the least-squares log-likelihood function

        .. math::
            \hat{\rho}
                &= -\mbox{argmin }\log\mathcal{L}{\rho} \\
                &= \mbox{argmin }\sum_i (\mbox{Tr}[E_j\rho] - \hat{p}_i)^2 \\
                &= \mbox{argmin }\|Ax - y \|_2^2

        where

        * :math:`A = \sum_j |j \rangle\!\langle\!\langle E_j|` is the matrix of measured
          basis elements.
        * :math:`y = \sum_j \hat{p}_j |j\rangle` is the vector of estimated measurement
          outcome probabilites for each basis element.
        * :math:`x = |\rho\rangle\!\rangle` is the vectorized density matrix.

    Additional Details
        The linear inversion solution is given by

        .. math::
            \hat{\rho} = \sum_i \hat{p}_i D_i

        where measurement probabilities :math:`\hat{p}_i = f_i / n_i` are estimated
        from the observed count frequencies :math:`f_i` in :math:`n_i` shots for each
        basis element :math:`i`, and :math:`D_i` is the *dual basis* element constructed
        from basis :math:`\{E_i\}` via:

        .. math:

            |D_i\rangle\!\rangle = M^{-1}|E_i \rangle\!\rangle \\
            M = \sum_j |E_j\rangle\!\rangle\!\langle\!\langle E_j|

    .. note::

        Linear inversion is only possible if the input bases are a spanning set
        for the vector space of the reconstructed matrix
        (*tomographically complete*). If the basis is not tomographically complete
        the :func:`~qiskit_experiments.library.tomography.fitters.scipy_linear_lstsq`
        function can be used to solve the same objective function via
        least-squares optimization.

    Args:
        outcome_data: list of outcome frequency data.
        shot_data: basis measurement total shot data.
        measurement_data: measurement basis indice data.
        preparation_data: preparation basis indice data.
        measurement_basis: measurement matrix basis.
        preparation_basis: Optional, preparation matrix basis.

    Raises:
        AnalysisError: If the fitted vector is not a square matrix

    Returns:
        The fitted matrix rho.
    """
    # Construct dual bases
    meas_dual_basis = dual_measurement_basis(measurement_basis)
    if preparation_basis:
        prep_dual_basis = dual_preparation_basis(preparation_basis)
    else:
        prep_dual_basis = None

    if shot_data is None:
        shot_data = np.ones(len(outcome_data))

    # Construct linear inversion matrix
    rho_fit = 0.0
    for i, outcomes in enumerate(outcome_data):
        shots = shot_data[i]
        midx = measurement_data[i]
        pidx = preparation_data[i]
        for outcome, freq in outcomes:
            prob = freq / shots
            dual_op = fitter_utils.single_basis_matrix(
                midx,
                outcome,
                preparation_index=pidx,
                measurement_basis=meas_dual_basis,
                preparation_basis=prep_dual_basis,
            )
            rho_fit = rho_fit + prob * dual_op

    return rho_fit, {}


@lru_cache(2)
def dual_preparation_basis(basis: FitterPreparationBasis):
    """Construct a dual preparation basis for linear inversion"""
    return FitterPreparationBasis(fitter_utils.dual_states(basis._mats), name=f"Dual_{basis.name}")


@lru_cache(2)
def dual_measurement_basis(basis: FitterMeasurementBasis):
    """Construct a dual preparation basis for linear inversion"""
    # Vectorize basis and basis matrix of outcome projectors
    states = []
    extra = []
    num_basis = len(basis._basis)
    for i in range(num_basis):
        for outcome, povm in basis._basis[i].items():
            states.append(povm)
            extra.append([i, outcome])
        dpovm = basis._outcome_default[i]
        if dpovm is not None:
            states.append(dpovm)
            extra.append([i, None])

    # Compute dual states and convert back to dicts
    dbasis = fitter_utils.dual_states(states)
    dual_basis = [{} for i in range(num_basis)]
    for povm, (idx, outcome) in zip(dbasis, extra):
        dual_basis[idx][outcome] = povm

    return FitterMeasurementBasis(dual_basis, name=f"Dual_{basis.name}")
