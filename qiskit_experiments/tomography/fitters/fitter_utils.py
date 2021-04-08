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
Common utility functions for tomography fitters.
"""

from typing import Optional, Tuple
import numpy as np
import scipy.linalg as la

from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.tomography.basis import FitterBasis


def make_positive_semidefinite(mat: np.array, epsilon: float = 0) -> np.array:
    """
    Rescale a Hermitian matrix to nearest postive semidefinite matrix.

    Args:
        mat: a hermitian matrix.
        epsilon: (default: 0) the threshold for setting
            eigenvalues to zero. If epsilon > 0 positive eigenvalues
            below epsilon will also be set to zero.
    Raises:
        AnalysisError: If epsilon is negative

    Returns:
        The input matrix rescaled to have non-negative eigenvalues.

    References:
        [1] J Smolin, JM Gambetta, G Smith, Phys. Rev. Lett. 108, 070502
            (2012). Open access: arXiv:1106.5458 [quant-ph].
    """

    if epsilon < 0:
        raise AnalysisError("epsilon must be non-negative.")

    # Get the eigenvalues and eigenvectors of rho
    # eigenvalues are sorted in increasing order
    # v[i] <= v[i+1]

    dim = len(mat)
    v, w = la.eigh(mat)
    for j in range(dim):
        if v[j] < epsilon:
            tmp = v[j]
            v[j] = 0.0
            # Rescale remaining eigenvalues
            x = 0.0
            for k in range(j + 1, dim):
                x += tmp / (dim - (j + 1))
                v[k] = v[k] + tmp / (dim - (j + 1))

    # Build positive matrix from the rescaled eigenvalues
    # and the original eigenvectors

    mat_psd = np.zeros([dim, dim], dtype=complex)
    for j in range(dim):
        mat_psd += v[j] * np.outer(w[:, j], np.conj(w[:, j]))

    return mat_psd


def stacked_basis_matrix(
    meas_basis_data: np.ndarray,
    prep_basis_data: np.ndarray,
    meas_matrix_basis: FitterBasis,
    prep_matrix_basis: FitterBasis,
) -> np.ndarray:
    """Return stacked vectorized basis matrix A for least squares."""
    size, msize1 = meas_basis_data.shape
    mdim = meas_matrix_basis([0]).size ** msize1
    if prep_matrix_basis:
        _, psize1 = prep_basis_data.shape
        pdim = prep_matrix_basis([0]).size ** psize1
    else:
        psize1 = 0
        pdim = 1
    ret = np.zeros((size, mdim * pdim), dtype=complex)
    for i in range(size):
        op = meas_matrix_basis(meas_basis_data[i])
        if psize1:
            op = np.kron(prep_matrix_basis(prep_basis_data[i]).T, op)
        ret[i] = np.ravel(op, order="F")
    return ret


def guassian_lstsq_data(
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    frequency_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_basis: Optional[FitterBasis] = None,
    preparation_basis: Optional[FitterBasis] = None,
    hedging_beta: float = 0.5,
    binomial_weights: bool = True,
    custom_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Return the matrix and vector for Gaussian least-squares fitting.

    Args:
        measurement_data: measurement basis indice data.
        preparation_data: preparation basis indice data.
        frequency_data: basis measurement frequency data.
        shot_data: basis measurement total shot data.
        measurement_basis: measurement matrix basis.
        preparation_basis: preparation matrix basis.
        hedging_beta: Hedging parameter for converting frequencies to
                      probabilities. If 0 hedging is disabled.
        binomial_weights: Compute binomial weights from frequency data.
        custom_weights: Optional, custom weights for fitter. If specified
                        binomial weights will be disabled.

    Raises:
        AnalysisError: If the fitted vector is not a square matrix

    Returns:
        The pair (A, y) where :math:`A` is the basis matrix, and :math:`y`
        is the data vector, for solving the linear least squares problem
        :math:`\argmin_x ||Ax - y||_2`.
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

    # Construct basis A matrix
    basis_matrix = stacked_basis_matrix(
        measurement_data, preparation_data, measurement_basis, preparation_basis
    )

    # Optionally apply a weights vector to the data and projectors
    if custom_weights is not None:
        weights = custom_weights
    elif binomial_weights:
        weights = np.sqrt(shot_data / (probability_data * (1 - probability_data)))
    else:
        weights = None
    if weights is not None:
        basis_matrix = weights[:, None] * basis_matrix
        probability_data = weights * probability_data

    return basis_matrix, probability_data
