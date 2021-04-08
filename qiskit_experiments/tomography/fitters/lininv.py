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
Basic linear inversion tomography fitter.
"""

from typing import Dict, Optional
import numpy as np

from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.tomography.basis import FitterBasis
from .fitter_utils import make_positive_semidefinite


def construct_dual_basis(matrix_basis: FitterBasis):
    """Construct a dual matrix basis for linear inversion"""
    mat_basis = matrix_basis._elements
    size, dim1, dim2 = np.shape(mat_basis)
    vec_basis = np.reshape(mat_basis, (size, dim1 * dim2))
    basis_mat = np.sum([np.outer(i, np.conj(i)) for i in vec_basis], axis=0)

    try:
        inv_mat = np.linalg.inv(basis_mat)
    except np.linalg.LinAlgError as ex:
        raise ValueError(
            "Cannot construct dual FitterBasis. Input FitterBasis"
            f" {matrix_basis.name} is not tomographically complete"
        ) from ex

    vec_dual = np.tensordot(inv_mat, vec_basis, axes=([1], [1])).T
    dual_elements = np.reshape(vec_dual, (size, dim1, dim2))
    # Copy basis
    return FitterBasis(
        dual_elements, num_indices=matrix_basis.num_indices, name=f"Dual_{matrix_basis.name}"
    )


def linear_inversion(
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    frequency_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_basis: Optional[FitterBasis] = None,
    preparation_basis: Optional[FitterBasis] = None,
    psd: bool = True,
    hedging_beta: float = 0.5,
    trace: Optional[float] = None,
) -> Dict:
    r"""Linear inversion tomography fitter.

    Args:
        measurement_data: measurement basis indice data.
        preparation_data: preparation basis indice data.
        frequency_data: basis measurement frequency data.
        shot_data: basis measurement total shot data.
        measurement_basis: measurement matrix basis.
        preparation_basis: preparation matrix basis.
        psd: If True rescale the eigenvalues of fitted matrix to be positive
             semidefinite (default: True)
        hedging_beta: Hedging parameter for converting frequencies to
                      probabilities. If 0 hedging is disabled.
        trace: trace constraint for the fitted matrix (default: None).

    Raises:
        AnalysisError: If the fitted vector is not a square matrix

    Returns:
        The fitted matrix rho.
    """
    # Construct probability vector
    if hedging_beta > 0:
        if hedging_beta < 0:
            raise AnalysisError("beta = {} must be non-negative.".format(hedging_beta))
        basis_outcomes = measurement_basis.num_outcomes
        num_outcomes = len(measurement_data[0]) ** basis_outcomes
        probability_data = (frequency_data + hedging_beta) / (
            shot_data + num_outcomes * hedging_beta
        )
    else:
        probability_data = frequency_data / shot_data

    meas_dual_basis = construct_dual_basis(measurement_basis)
    if preparation_basis:
        prep_dual_basis = construct_dual_basis(preparation_basis)
    else:
        prep_dual_basis = None

    # Get fitted matrix dimensions
    size, msize1 = measurement_data.shape
    mdim1, mdim2 = meas_dual_basis([0]).shape
    if prep_dual_basis:
        _, psize1 = preparation_data.shape
        pdim1, pdim2 = prep_dual_basis([0]).shape
    else:
        psize1 = 0
        pdim1, pdim2 = (1, 1)

    # Construct linear inversion matrix
    rho_fit = np.zeros(
        (mdim1 ** msize1 * pdim1 ** psize1, mdim2 ** msize1 * pdim2 ** psize1), dtype=complex
    )
    for i in range(size):
        dual_op = meas_dual_basis(measurement_data[i])
        if psize1:
            dual_op = np.kron(prep_dual_basis(preparation_data[i]).T, dual_op)
        rho_fit += probability_data[i] * dual_op

    # Rescale fitted density matrix be positive-semidefinite
    if psd is True:
        rho_fit = make_positive_semidefinite(rho_fit)

    if trace is not None:
        rho_fit *= trace / np.trace(rho_fit)

    return {"value": rho_fit}
