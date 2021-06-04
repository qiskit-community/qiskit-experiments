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


def single_basis_matrix(
    measurement_element: np.ndarray,
    preparation_element: np.ndarray,
    measurement_basis: FitterBasis,
    preparation_basis: Optional[FitterBasis] = None,
) -> np.ndarray:
    """Return a single element basis matrix."""
    op = measurement_basis(measurement_element)
    if preparation_basis:
        op = np.kron(preparation_basis(preparation_element).T, op)
    return op


def stacked_basis_matrix(
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: FitterBasis,
    preparation_basis: FitterBasis,
) -> np.ndarray:
    """Return stacked vectorized basis matrix A for least squares."""
    size, msize1 = measurement_data.shape
    mdim = measurement_basis([0]).size ** msize1
    if preparation_basis:
        _, psize1 = preparation_data.shape
        pdim = preparation_basis([0]).size ** psize1
    else:
        psize1 = 0
        pdim = 1
    ret = np.zeros((size, mdim * pdim), dtype=complex)
    for i in range(size):
        op = np.conj(single_basis_matrix(
            measurement_data[i], preparation_data[i], measurement_basis, preparation_basis
        ))
        ret[i] = np.ravel(op, order="F")
    return ret


def binomial_weights(
    frequency_data: np.ndarray, shot_data: np.ndarray, num_outcomes: int = 2, beta: float = 0
) -> np.ndarray:
    r"""Compute weights vector from the binomial distribution.

    The returned weights are given by :math:`w_i 1 / \sigma_i` where
    the standard deviation :math:`\sigma_i` is estimated as
    :math:`\sigma_i = \sqrt{p_i(1-p_i) / n_i}`. To avoid dividing
    by zero the probabilities are hedged using the *add-beta* rule

    .. math:
        p_i = \frac{f_i + \beta}{n_i + K \beta}

    where :math:`f_i` is the observed frequency, :math:`n_i` is the
    number of shots, and :math:`K` is the number of possible measurement
    outcomes.

    Args:
        frequency_data: basis measurement frequency data.
        shot_data: basis measurement total shot data.
        num_outcomes: the number of measuremement outcomes.
        beta: Hedging parameter for converting frequencies to
              probabilities. If 0 hedging is disabled.

    Returns:
        The weight vector.
    """
    # Compute hedged probabilities where the "add-beta" rule ensures
    # there are no zero or 1 values so we don't have any zero variance
    probs = (frequency_data + beta) / (shot_data + num_outcomes * beta)
    variance = probs * (1 - probs)
    return np.sqrt(shot_data / variance)
