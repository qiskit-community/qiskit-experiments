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

from typing import Optional, Tuple, Callable, Sequence, Union
import functools
import numpy as np

from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.library.tomography.basis import (
    MeasurementBasis,
    PreparationBasis,
)


def lstsq_data(
    outcome_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: Optional[MeasurementBasis] = None,
    preparation_basis: Optional[PreparationBasis] = None,
    measurement_qubits: Optional[Tuple[int, ...]] = None,
    preparation_qubits: Optional[Tuple[int, ...]] = None,
    conditional_measurement_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return stacked vectorized basis matrix A for least squares."""
    if measurement_basis is None and preparation_basis is None:
        raise AnalysisError("`measurement_basis` and `preparation_basis` cannot both be None")

    # Get leading dimension of returned matrix
    size = outcome_data.size
    mdim = 1
    pdim = 1
    circ_cdim = outcome_data.shape[0]
    cdim = 1
    num_cond = 0

    # Get full and conditional measurement basis dimensions
    if measurement_basis:
        bsize, num_meas = measurement_data.shape
        if not measurement_qubits:
            measurement_qubits = tuple(range(num_meas))

        # Partition measurement qubits into conditional measurement qubits and
        # regular measurement qubits
        if conditional_measurement_indices is not None:
            conditional_measurement_indices = tuple(conditional_measurement_indices)
            conditional_qubits = tuple(
                measurement_qubits[i] for i in conditional_measurement_indices
            )
            measurement_qubits = tuple(
                qubit
                for i, qubit in enumerate(measurement_qubits)
                if i not in conditional_measurement_indices
            )
            num_cond = len(conditional_measurement_indices)
            cdim = np.prod(measurement_basis.outcome_shape(conditional_qubits), dtype=int)
        if measurement_qubits:
            mdim = np.prod(measurement_basis.matrix_shape(measurement_qubits), dtype=int)

    # Get preparation basis dimensions
    if preparation_basis:
        bsize, num_prep = preparation_data.shape
        if not preparation_qubits:
            preparation_qubits = tuple(range(num_prep))
        if preparation_qubits:
            pdim = np.prod(preparation_basis.matrix_shape(preparation_qubits), dtype=int)

    # Reduced outcome functions
    # Set measurement indices to an array so we can use for array indexing later
    if num_cond:
        f_cond_outcome = _partial_outcome_function(conditional_measurement_indices)
        if measurement_qubits:
            measurement_indices = np.array(
                [i for i in range(num_meas) if i not in conditional_measurement_indices], dtype=int
            )
            f_meas_outcome = _partial_outcome_function(tuple(measurement_indices))
        else:
            measurement_indices = []
            f_meas_outcome = lambda x: 0
    else:
        measurement_indices = None
        f_meas_outcome = lambda x: x
        f_cond_outcome = lambda x: 0

    # Allocate empty stacked basis matrix and prob vector
    reduced_size = size // circ_cdim // cdim
    basis_mat = np.zeros((reduced_size, mdim * mdim * pdim * pdim), dtype=complex)
    probs = np.zeros((circ_cdim, cdim, reduced_size), dtype=float)

    # Probably scale is 1/shots
    # To avoid NaNs from dividing by 0 for conditional data with no shots
    # we manually set these scale values to 0.For 0 shot conditional state data
    # we manually set probs to zero
    pscale_data = np.zeros(shot_data.shape, dtype=float)
    mask = shot_data != 0
    pscale_data[mask] = 1 / shot_data[mask]

    # Fill matrices
    for cond_circ_idx in range(outcome_data.shape[0]):
        cond_idxs = {i: 0 for i in range(cdim)}
        for i in range(bsize):
            midx = measurement_data[i]
            midx_meas = midx[measurement_indices]
            pidx = preparation_data[i]
            odata = outcome_data[cond_circ_idx][i]
            pscale = pscale_data[cond_circ_idx][i]

            # Get prep basis component
            if preparation_qubits:
                p_mat = np.transpose(preparation_basis.matrix(pidx, preparation_qubits))
            else:
                p_mat = None

            # Get probabilities and optional measurement basis component
            midx_meas = midx[measurement_indices] if num_cond else midx
            meas_cache = set()
            for outcome in range(odata.size):
                # Get conditional and measurement outcome values
                outcome_cond = f_cond_outcome(outcome)
                outcome_meas = f_meas_outcome(outcome)
                idx = cond_idxs[outcome_cond]

                # Store probability
                probs[cond_circ_idx, outcome_cond, idx] = odata[outcome] * pscale

                # Check if new meas basis element and construct basis matrix
                store_mat = True
                if measurement_qubits:
                    if outcome_meas not in meas_cache:
                        meas_cache.add(outcome_meas)
                        mat = measurement_basis.matrix(midx_meas, outcome_meas, measurement_qubits)
                        if preparation_basis:
                            mat = np.kron(p_mat, mat)
                    else:
                        store_mat = False
                else:
                    mat = p_mat
                if store_mat:
                    basis_mat[idx] = np.conj(np.ravel(mat, order="F"))

                # Increase counter
                cond_idxs[outcome_cond] += 1

    return basis_mat, probs


def dirichlet_mean_and_var(
    outcome_data: np.ndarray,
    shot_data: Optional[Union[np.ndarray, int]] = None,
    outcome_prior: Union[np.ndarray, int] = 0.5,
    conditional_measurement_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute mean probabilities and variance from outcome data.

    This is computed via a Bayesian update of a Dirichlet distribution
    with observed outcome data frequences :math:`f_i(s)`, and Dirichlet
    prior :math:`\alpha_i(s)` for tomography basis index `i` and
    measurement outcome `s`.

    The mean posterior probabilities are computed as

    .. math:
        p_i(s) &= \frac{f_i(s) + \alpha_i(s)}{\bar{\alpha}_i + N_i} \\
        Var[p_i(s)] &= \frac{p_i(s)(1-p_i(s))}{\bar{\alpha}_i + N_i + 1}

    where :math:`N_i = \sum_s f_i(s)` is the total number of shots, and
    :math:`\bar{\alpha}_i = \sum_s \alpha_i(s)` is the norm of the prior
    vector for basis index `i`.

    Args:
        outcome_data: measurement outcome frequency data.
        shot_data: Optional, basis measurement total shot data. If not
            provided this will be inferred from the sum of outcome data
            for each basis index.
        outcome_prior: measurement outcome Dirichlet distribution prior.
        conditional_measurement_indices: outcome indices of conditional
            outcome data.

    Returns:
        The mean probabilities and variances for Bayesian update
        with the given outcome data and prior.
    """
    size = outcome_data.size
    dim_circ_cond, num_data, tomo_outcomes = outcome_data.shape
    if conditional_measurement_indices:
        dim_cond = 2 ** len(conditional_measurement_indices)
        f_cond_outcome = _partial_outcome_function(tuple(conditional_measurement_indices))
    else:
        dim_cond = 1
        f_cond_outcome = lambda x: 0

    # Compute hedged probabilities where the "add-beta" rule ensures
    # there are no zero or 1 values so we don't have any zero variance
    reduced_size = size // dim_cond // dim_circ_cond
    mean_probs = np.zeros((dim_circ_cond, dim_cond, reduced_size), dtype=float)
    var_denom = np.zeros((dim_circ_cond, dim_cond, reduced_size), dtype=int)

    # Bayesian update
    posterior = outcome_data + outcome_prior

    # Total shots for computing probabilities
    # If shot data is not provided it is inferred from outcome data
    # assuming shots is the sum of all observed counts for each basis
    if shot_data is None:
        posterior_total = np.sum(posterior, axis=(0, -1))
    else:
        if isinstance(shot_data, np.ndarray):
            total_shots = np.sum(shot_data, axis=0)
        else:
            total_shots = shot_data
        outcome_shots = np.sum(outcome_data, axis=(0, -1))
        posterior_shots = np.sum(posterior, axis=(0, -1))
        posterior_total = posterior_shots + total_shots - outcome_shots

    # Fill matrices
    for i in range(dim_circ_cond):
        cond_idxs = {i: 0 for i in range(dim_cond)}
        for j in range(num_data):
            freqs = posterior[i, j]
            for outcome in range(tomo_outcomes):
                shots = posterior_total[outcome]
                outcome_cond = f_cond_outcome(outcome)
                idx = cond_idxs[outcome_cond]
                mean_probs[i, outcome_cond, idx] = freqs[outcome] / shots
                var_denom[i, outcome_cond, idx] = shots + 1
                cond_idxs[outcome_cond] += 1
    variance = mean_probs * (1 - mean_probs) / var_denom
    return mean_probs, variance


def binomial_weights(
    outcome_data: np.ndarray,
    shot_data: Optional[np.ndarray] = None,
    beta: float = 0,
    conditional_measurement_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    r"""Compute weights vector from the multinomial distribution.

    The returned weights are given by :math:`w_i = 1 / \sigma_i` where
    the standard deviation :math:`\sigma_i` is estimated via Bayesian
    update of the Dirichlet distribution.

    Args:
        outcome_data: measurement outcome frequency data.
        shot_data: Optional, basis measurement total shot data. If not
            provided this will be inferred from the sum of outcome data
            for each basis index.
        beta: Hedging parameter for converting frequencies to
              probabilities. If 0 hedging is disabled.
        conditional_measurement_indices: outcome indices of conditional
            outcome data.

    Returns:
        The weight vector.
    """
    _, variance = dirichlet_mean_and_var(
        outcome_data,
        shot_data=shot_data,
        outcome_prior=beta,
        conditional_measurement_indices=conditional_measurement_indices,
    )
    return 1 / np.sqrt(variance)


@functools.lru_cache(None)
def _partial_outcome_function(indices: Tuple[int]) -> Callable:
    """Return function for computing partial outcome of specified indices"""
    # NOTE: This function only works for 2-outcome subsystem measurements
    ind_array = np.asarray(indices, dtype=int)
    mask_array = 1 << ind_array
    bit_array = 1 << np.arange(ind_array.size, dtype=int)

    @functools.lru_cache(None)
    def partial_outcome(outcome: int) -> int:
        return np.dot(bit_array, (mask_array & outcome) >> ind_array)

    return partial_outcome
