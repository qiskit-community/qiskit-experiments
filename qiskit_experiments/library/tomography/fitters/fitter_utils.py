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

from qiskit_experiments.library.tomography.basis import (
    FitterMeasurementBasis,
    FitterPreparationBasis,
)


def lstsq_data(
    outcome_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: Optional[FitterMeasurementBasis] = None,
    preparation_basis: Optional[FitterPreparationBasis] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return stacked vectorized basis matrix A for least squares."""
    # Get leading dimension of returned matrix
    size = outcome_data.size

    # Get measurement basis dimensions
    if measurement_basis:
        bsize, msize = measurement_data.shape
        mdim = measurement_basis.matrix([0], 0).size ** msize
    else:
        msize = 0
        mdim = 1
    # Get preparation basis dimensions
    if preparation_basis:
        bsize, psize = preparation_data.shape
        pdim = preparation_basis.matrix([0]).size ** psize
    else:
        psize = 0
        pdim = 1

    # Allocate empty stacked basis matrix and prob vector
    basis_mat = np.zeros((size, mdim * pdim), dtype=complex)
    probs = np.zeros(size, dtype=float)
    idx = 0
    for i in range(bsize):
        midx = measurement_data[i]
        pidx = preparation_data[i]
        shots = shot_data[i]
        odata = outcome_data[i]

        # Get prep basis component
        if preparation_basis:
            p_mat = np.transpose(preparation_basis.matrix(pidx))
        else:
            p_mat = None

        # Get probabilities and optional measurement basis component
        for outcome in range(odata.size):
            if measurement_basis:
                op = measurement_basis.matrix(midx, outcome)
                if preparation_basis:
                    op = np.kron(p_mat, op)
            else:
                op = p_mat

            # Add vectorized op to stacked basis matrix
            basis_mat[idx] = np.conj(np.ravel(op, order="F"))
            probs[idx] = odata[outcome] / shots
            idx += 1
    return basis_mat, probs


def binomial_weights(
    outcome_data: np.ndarray,
    shot_data: np.ndarray,
    num_outcomes: Optional[np.ndarray] = None,
    beta: float = 0,
) -> np.ndarray:
    r"""Compute weights vector from the binomial distribution.

    The returned weights are given by :math:`w_i = 1 / \sigma_i` where
    the standard deviation :math:`\sigma_i` is estimated as
    :math:`\sigma_i = \sqrt{p_i(1-p_i) / n_i}`. To avoid dividing
    by zero the probabilities are hedged using the *add-beta* rule

    .. math:
        p_i = \frac{f_i + \beta}{n_i + K \beta}

    where :math:`f_i` is the observed frequency, :math:`n_i` is the
    number of shots, and :math:`K` is the number of possible measurement
    outcomes.

    Args:
        outcome_data: measurement outcome frequency data.
        shot_data: basis measurement total shot data.
        num_outcomes: the number of measurement outcomes for
                      each outcome data set.
        beta: Hedging parameter for converting frequencies to
              probabilities. If 0 hedging is disabled.

    Returns:
        The weight vector.
    """
    size = outcome_data.size
    num_data, num_outcomes = outcome_data.shape

    # Compute hedged probabilities where the "add-beta" rule ensures
    # there are no zero or 1 values so we don't have any zero variance
    probs = np.zeros(size, dtype=float)
    prob_shots = np.zeros(size, dtype=int)
    idx = 0
    for i in range(num_data):
        shots = shot_data[i]
        denom = shots + num_outcomes * beta
        freqs = outcome_data[i]
        for outcome in range(num_outcomes):
            probs[idx] = (freqs[outcome] + beta) / denom
            prob_shots[idx] = shots
            idx += 1
    variance = probs * (1 - probs)
    return np.sqrt(prob_shots / variance)
