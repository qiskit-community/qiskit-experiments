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

from typing import Dict, Tuple, Optional, Sequence, List
from functools import lru_cache
import time
import numpy as np
from qiskit_experiments.library.tomography.basis import (
    MeasurementBasis,
    PreparationBasis,
    LocalMeasurementBasis,
    LocalPreparationBasis,
)
from .lstsq_utils import _partial_outcome_function
from .fitter_data import _basis_dimensions


def linear_inversion(
    outcome_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: Optional[MeasurementBasis] = None,
    preparation_basis: Optional[PreparationBasis] = None,
    measurement_qubits: Optional[Tuple[int, ...]] = None,
    preparation_qubits: Optional[Tuple[int, ...]] = None,
    conditional_measurement_indices: Optional[np.ndarray] = None,
    conditional_preparation_indices: Optional[np.ndarray] = None,
    atol: float = 1e-8,
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
          outcome probabilities for each basis element.
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

        The Linear inversion fitter treats the input measurement and preparation
        bases as local bases and constructs separate 1-qubit dual basis for each
        individual qubit.

        Linear inversion is only possible if the input bases are local and a spanning
        set for the vector space of the reconstructed matrix
        (*tomographically complete*). If the basis is not tomographically complete
        the :func:`~qiskit_experiments.library.tomography.fitters.scipy_linear_lstsq`
        or :func:`~qiskit_experiments.library.tomography.fitters.cvxpy_linear_lstsq`
        function can be used to solve the same objective function via
        least-squares optimization.

    Args:
        outcome_data: basis outcome frequency data.
        shot_data: basis outcome total shot data.
        measurement_data: measurement basis indice data.
        preparation_data: preparation basis indice data.
        measurement_basis: the tomography measurement basis.
        preparation_basis: the tomography preparation basis.
        measurement_qubits: Optional, the physical qubits that were measured.
             If None they are assumed to be [0, ..., M-1] for M measured qubits.
        preparation_qubits: Optional, the physical qubits that were prepared.
            If None they are assumed to be [0, ..., N-1] forN prepared qubits.
        conditional_measurement_indices: Optional, conditional measurement data
            indices. If set this will return a list of fitted states conditioned
            on a fixed basis measurement of these qubits.
        conditional_preparation_indices: Optional, conditional preparation data
            indices. If set this will return a list of fitted states conditioned
            on a fixed basis preparation of these qubits.
        atol: truncate any probabilities below this value to zero.

    Raises:
        AnalysisError: If the fitted vector is not a square matrix

    Returns:
        The fitted matrix rho.
    """
    t_start = time.time()

    if measurement_basis and measurement_qubits is None:
        measurement_qubits = tuple(range(measurement_data.shape[1]))
    if preparation_basis and preparation_qubits is None:
        preparation_qubits = tuple(range(preparation_data.shape[1]))

    input_dims = _basis_dimensions(
        basis=preparation_basis,
        qubits=preparation_qubits,
        conditional_indices=conditional_preparation_indices,
    )
    output_dims = _basis_dimensions(
        basis=measurement_basis,
        qubits=measurement_qubits,
        conditional_indices=conditional_measurement_indices,
    )

    metadata = {
        "fitter": "linear_inversion",
        "input_dims": input_dims,
        "output_dims": output_dims,
    }

    if conditional_preparation_indices:
        # Split measurement qubits into conditional and non-conditional qubits
        f_prep_qubits = []
        prep_indices = []
        for i, qubit in enumerate(preparation_qubits):
            if i not in conditional_preparation_indices:
                f_prep_qubits.append(qubit)
                prep_indices.append(i)
        # Indexing array for fully tomo measured qubits
        f_prep_indices = np.array(prep_indices, dtype=int)
        f_cond_prep_indices = np.array(conditional_preparation_indices, dtype=int)
    else:
        f_prep_qubits = preparation_qubits
        f_prep_indices = slice(None)
        f_cond_prep_indices = slice(0, 0)

    if conditional_measurement_indices:
        # Split measurement qubits into conditional and non-conditional qubits
        f_cond_meas_qubits = []
        f_meas_qubits = []
        meas_indices = []
        for i, qubit in enumerate(measurement_qubits):
            if i in conditional_measurement_indices:
                f_cond_meas_qubits.append(qubit)
            else:
                f_meas_qubits.append(qubit)
                meas_indices.append(i)

        cond_meas_size = np.prod(measurement_basis.outcome_shape(f_cond_meas_qubits), dtype=int)

        # Indexing array for fully tomo measured qubits
        f_meas_indices = np.array(meas_indices, dtype=int)
        f_cond_meas_indices = np.array(conditional_measurement_indices, dtype=int)

        # Reduced outcome functions
        f_meas_outcome = _partial_outcome_function(tuple(f_meas_indices))
        f_cond_meas_outcome = _partial_outcome_function(tuple(conditional_measurement_indices))
    else:
        cond_meas_size = 1
        f_meas_qubits = measurement_qubits
        f_meas_indices = slice(None)
        f_cond_meas_indices = slice(0, 0)

        def f_meas_outcome(x):
            return x

        def f_cond_meas_outcome(_):
            return 0

    # Construct dual bases
    meas_dual_basis = None
    if measurement_basis and f_meas_qubits:
        meas_duals = {i: _dual_povms(measurement_basis, i) for i in f_meas_qubits}
        meas_dual_basis = LocalMeasurementBasis(
            f"Dual_{measurement_basis.name}", qubit_povms=meas_duals
        )

    prep_dual_basis = None
    if preparation_basis and f_prep_qubits:
        prep_duals = {i: _dual_states(preparation_basis, i) for i in f_prep_qubits}
        prep_dual_basis = LocalPreparationBasis(
            f"Dual_{preparation_basis.name}", qubit_states=prep_duals
        )

    if shot_data is None:
        # Define shots by sum of all outcome frequencies for each basis
        shot_data = np.sum(outcome_data, axis=(0, -1))

    # Calculate shape of matrix to be fitted
    if prep_dual_basis:
        pdim = np.prod(prep_dual_basis.matrix_shape(f_prep_qubits), dtype=int)
    else:
        pdim = 1
    if meas_dual_basis:
        mdim = np.prod(meas_dual_basis.matrix_shape(f_meas_qubits), dtype=int)
    else:
        mdim = 1
    shape = (pdim * mdim, pdim * mdim)

    # Construct linear inversion matrix
    cond_circ_size = outcome_data.shape[0]
    cond_fits = []
    if cond_circ_size > 1:
        metadata["conditional_circuit_outcome"] = []
    if conditional_measurement_indices:
        metadata["conditional_measurement_index"] = []
        metadata["conditional_measurement_outcome"] = []
    if conditional_preparation_indices:
        metadata["conditional_preparation_index"] = []
    for circ_idx in range(cond_circ_size):
        fits = {}
        for i, outcomes in enumerate(outcome_data[circ_idx]):
            shots = shot_data[i]
            pidx = preparation_data[i][f_prep_indices]
            midx = measurement_data[i][f_meas_indices]
            cond_prep_key = tuple(preparation_data[i][f_cond_prep_indices])
            cond_meas_key = tuple(measurement_data[i][f_cond_meas_indices])
            key = (cond_prep_key, cond_meas_key)
            if key not in fits:
                fits[key] = [np.zeros(shape, dtype=complex) for _ in range(cond_meas_size)]

            # Get prep basis component
            if prep_dual_basis:
                p_mat = np.transpose(prep_dual_basis.matrix(pidx, f_prep_qubits))
            else:
                p_mat = 1

            # Get probabilities and optional measurement basis component
            for outcome, freq in enumerate(outcomes):
                prob = freq / shots
                if np.isclose(prob, 0, atol=atol):
                    # Skip component with zero probability
                    continue

                # Get component on non-conditional bits
                outcome_meas = f_meas_outcome(outcome)
                if meas_dual_basis:
                    dual_op = meas_dual_basis.matrix(midx, outcome_meas, f_meas_qubits)
                    if prep_dual_basis:
                        dual_op = np.kron(p_mat, dual_op)
                else:
                    dual_op = p_mat

                # Add component to correct conditional
                outcome_cond = f_cond_meas_outcome(outcome)
                fits[key][outcome_cond] += prob * dual_op

        # Append conditional fit metadata
        for (prep_key, meas_key), c_fits in fits.items():
            cond_fits += c_fits
            if cond_circ_size > 1:
                metadata["conditional_circuit_outcome"] += len(c_fits) * [circ_idx]
            if conditional_measurement_indices:
                metadata["conditional_measurement_index"] += len(c_fits) * [meas_key]
                metadata["conditional_measurement_outcome"] += list(range(cond_meas_size))
            if conditional_preparation_indices:
                metadata["conditional_preparation_index"] += len(c_fits) * [prep_key]

    t_stop = time.time()
    metadata["fitter_time"] = t_stop - t_start

    if len(cond_fits) == 1:
        return cond_fits[0], metadata
    return cond_fits, metadata


@lru_cache(None)
def _dual_states(basis: PreparationBasis, qubit: int) -> np.ndarray:
    """Construct a dual preparation basis for linear inversion"""
    size = basis.index_shape((qubit,))[0]
    states = np.asarray([basis.matrix((i,), (qubit,)) for i in range(size)])
    return _construct_dual_states(states)


@lru_cache(None)
def _dual_povms(basis: MeasurementBasis, qubit: int) -> List[List[np.ndarray]]:
    """Construct dual POVM states for linear inversion"""
    size = basis.index_shape((qubit,))[0]
    num_outcomes = basis.outcome_shape((qubit,))[0]

    # Concatenate all POVM effects to treat as states for linear inversion
    states = []
    for index in range(size):
        for outcome in range(num_outcomes):
            states.append(basis.matrix((index,), outcome, (qubit,)))

    dual_basis = _construct_dual_states(states)

    # Organize back into nested lists of dual POVM effects
    dual_povms = []
    idx = 0
    for _ in range(size):
        dual_povms.append([dual_basis[idx + i] for i in range(num_outcomes)])
        idx += num_outcomes
    return dual_povms


def _construct_dual_states(states: Sequence[np.ndarray]):
    """Construct a dual preparation basis for linear inversion"""
    mats = np.asarray(states)
    size, dim1, dim2 = np.shape(mats)
    vec_basis = np.reshape(mats, (size, dim1 * dim2))
    basis_mat = np.sum([np.outer(i, np.conj(i)) for i in vec_basis], axis=0)

    try:
        inv_mat = np.linalg.inv(basis_mat)
    except np.linalg.LinAlgError as ex:
        raise ValueError(
            "Cannot construct dual basis states. Input states are not tomographically complete"
        ) from ex

    vec_dual = np.tensordot(inv_mat, vec_basis, axes=([1], [1])).T
    dual_mats = np.reshape(vec_dual, (size, dim1, dim2)).round(15)
    return dual_mats
