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

from typing import Dict, Optional
from functools import lru_cache
import numpy as np

from qiskit_experiments.tomography.basis import FitterBasis
from .fitter_utils import make_positive_semidefinite, single_basis_matrix

# LRU cache size of 2 is used to allow caching 1 measurement and
# 1 preparation basis when calling the linear_inversion fitter function


@lru_cache(2)
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
    trace: Optional[float] = None,
) -> Dict:
    r"""Linear inversion tomography fitter.

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

        Linear inversion does not support constraints directly, but the following
        constraints can be applied to the fitted matrix

        - *Positive-semidefinite* (``psd=True``): The eigenvalues of the fitted matrix
          are  rescaled using the method from [1] to remove any negative eigenvalues.
        - *Trace* (``trace=float``): If the trace constraint is applied , the fitted
          matrix is rescaled to have the specified trace.

        1. J Smolin, JM Gambetta, G Smith, Phys. Rev. Lett. 108, 070502 (2012).
           Open access: https://arxiv.org/abs/arXiv:1106.5458

    .. note::

        Linear inversion is only possible if the input bases are a spaning set
        for the vector space of the reconstructed matrix
        (*tomographically complete*). If the basis is not tomographically complete
        the :func:`~qiskit_experiments.tomography.fitters.scipy_linear_lstsq`
        function can be used to solve the same objective function via
        least-squares optimization.

    Args:
        measurement_data: measurement basis indice data.
        preparation_data: preparation basis indice data.
        frequency_data: basis measurement frequency data.
        shot_data: basis measurement total shot data.
        measurement_basis: measurement matrix basis.
        preparation_basis: preparation matrix basis.
        psd: If True rescale the eigenvalues of fitted matrix to be positive
             semidefinite (default: True)
        trace: trace constraint for the fitted matrix (default: None).

    Raises:
        AnalysisError: If the fitted vector is not a square matrix

    Returns:
        The fitted matrix rho.
    """
    probability_data = frequency_data / shot_data

    meas_dual_basis = construct_dual_basis(measurement_basis)
    if preparation_basis:
        prep_dual_basis = construct_dual_basis(preparation_basis)
    else:
        prep_dual_basis = None

    # Construct linear inversion matrix
    rho_fit = 0.0
    for i in range(probability_data.size):
        dual_op = single_basis_matrix(
            measurement_data[i], preparation_data[i], meas_dual_basis, prep_dual_basis
        )
        rho_fit = rho_fit + probability_data[i] * dual_op

    # Rescale fitted density matrix be positive-semidefinite
    if psd is True:
        rho_fit = make_positive_semidefinite(rho_fit)

    # Rescale trace
    if trace is not None:
        rho_fit *= trace / np.trace(rho_fit)

    return {"value": rho_fit}
