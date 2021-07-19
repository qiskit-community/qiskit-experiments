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

from typing import Optional, List, Tuple, Iterable
import numpy as np
import scipy.linalg as la

from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.library.tomography.basis import (
    FitterMeasurementBasis,
    FitterPreparationBasis,
)


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
    measurement_index: np.ndarray,
    outcome: int,
    measurement_basis: FitterMeasurementBasis,
    preparation_index: Optional[np.ndarray] = None,
    preparation_basis: Optional[FitterPreparationBasis] = None,
) -> np.ndarray:
    """Return a single element basis matrix.

    Args:
        measurement_index: measurement basis indices for each
            subsystem
        outcome: measurement outcome in the specified basis.
        measurement_basis: fitter measurement basis object.
        preparation_index: Optional, preparation basis indices
            for each subsystem.
        preparation_basis: fitter preparation basis object.

    Returns:
        The corresponding basis matrix for tomography fitter.
    """
    op = measurement_basis.matrix(measurement_index, outcome)
    if preparation_basis:
        pmat = preparation_basis.matrix(preparation_index)
        op = np.kron(pmat.T, op)
    return op


def lstsq_data(
    outcome_data: List[np.ndarray],
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: FitterMeasurementBasis,
    preparation_basis: Optional[FitterPreparationBasis] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return stacked vectorized basis matrix A for least squares."""
    # Get leading dimension of returned matrix
    size = 0
    for outcome in outcome_data:
        size += len(outcome)

    # Get measurement basis dimensions
    bsize, msize = measurement_data.shape
    mdim = measurement_basis.matrix([0], 0).size ** msize
    if preparation_basis:
        # Get preparation basis dimensions
        _, psize = preparation_data.shape
        pdim = preparation_basis.matrix([0]).size ** psize
    else:
        psize = 0
        pdim = 1

    # Construct basis matrix
    basis_mat = np.zeros((size, mdim * pdim), dtype=complex)
    probs = np.zeros(size, dtype=float)
    idx = 0
    for i in range(bsize):
        midx = measurement_data[i]
        pidx = preparation_data[i]
        shots = shot_data[i]
        odata = outcome_data[i]
        for outcome, freq in odata:
            op = single_basis_matrix(
                midx,
                outcome,
                measurement_basis=measurement_basis,
                preparation_index=pidx,
                preparation_basis=preparation_basis,
            )
            basis_mat[idx] = np.conj(np.ravel(op, order="F"))
            probs[idx] = freq / shots
            idx += 1
    return basis_mat, probs


def binomial_weights(
    outcome_data: List[np.ndarray],
    shot_data: np.ndarray,
    num_outcomes: Optional[np.ndarray] = None,
    beta: float = 0,
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
        outcome_data: list of outcome frequency data.
        shot_data: basis measurement total shot data.
        num_outcomes: the number of measurement outcomes for
                      each outcome data set.
        beta: Hedging parameter for converting frequencies to
              probabilities. If 0 hedging is disabled.

    Returns:
        The weight vector.
    """
    num_data = len(outcome_data)
    if num_outcomes is None:
        num_outcomes = 2 * np.ones(num_data, dtype=int)
    # Get leading dimension of returned matrix
    size = 0
    for outcome in outcome_data:
        size += len(outcome)

    # Compute hedged probabilities where the "add-beta" rule ensures
    # there are no zero or 1 values so we don't have any zero variance
    probs = np.zeros(size, dtype=float)
    prob_shots = np.zeros(size, dtype=int)
    idx = 0
    for i in range(num_data):
        shots = shot_data[i]
        denom = shots + num_outcomes[i] * beta
        odata = outcome_data[i]
        for outcome, freq in odata:
            probs[idx] = (freq + beta) / denom
            prob_shots[idx] = shots
            idx += 1
    variance = probs * (1 - probs)
    return np.sqrt(prob_shots / variance)


def dual_states(states: Iterable[np.ndarray]):
    """Construct a dual preparation basis for linear inversion"""
    mats = np.asarray(states)
    size, dim1, dim2 = np.shape(mats)
    vec_basis = np.reshape(mats, (size, dim1 * dim2))
    basis_mat = np.sum([np.outer(i, np.conj(i)) for i in vec_basis], axis=0)

    try:
        inv_mat = np.linalg.inv(basis_mat)
    except np.linalg.LinAlgError as ex:
        raise ValueError(
            "Cannot construct dual basis states. Input states" " are not tomographically complete"
        ) from ex

    vec_dual = np.tensordot(inv_mat, vec_basis, axes=([1], [1])).T
    dual_mats = np.reshape(vec_dual, (size, dim1, dim2))
    return dual_mats
