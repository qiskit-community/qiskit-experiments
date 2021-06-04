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
Linear least-square MLE tomography fitter.
"""

from typing import Optional, Dict
import numpy as np
import scipy.linalg as la

from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.tomography.basis import FitterBasis
from . import fitter_utils


def scipy_linear_lstsq(
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    frequency_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_basis: Optional[FitterBasis] = None,
    preparation_basis: Optional[FitterBasis] = None,
    psd: bool = True,
    trace: Optional[float] = None,
    weights: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict:
    r"""Weighted linear least-squares tomography fitter.

    This fitter reconstructs the maximum-likelihood estimate by using
    :func:`scipy.linalg.lstsq` to minimize the least-squares negative log
    likelihood function

    .. math::
        \hat{\rho}
            &= -\mbox{argmin }\log\mathcal{L}{\rho} \\
            &= \mbox{argmin }\sum_i w_i^2(\mbox{Tr}[E_j\rho] - \hat{p}_i)^2 \\
            &= \mbox{argmin }\|W(Ax - y) \|_2^2

    where

    * :math:`A = \sum_j |j \rangle\!\langle\!\langle E_j|` is the matrix of measured
      basis elements.
    * :math:`W = \sum_j w_j|j\rangle\!\langle j|` is an optional diagonal weights
      matrix if an optional weights vector is supplied.
    * :math:`y = \sum_j \hat{p}_j |j\langle` is the vector of estimated measurement
      outcome probabilites for each basis element.
    * :math:`x = |\rho\rangle\!\rangle` is the vectorized density matrix.

    .. note::

        Linear least squares does not support constraints directly, but the following
        constraints can be applied to the fitted matrix

        - *Positive-semidefinite* (``psd=True``): The eigenvalues of the fitted matrix
          are  rescaled using the method from [1] to remove any negative eigenvalues.
        - *Trace* (``trace=float``): If the trace constraint is applied , the fitted
          matrix is rescaled to have the specified trace.

        1. J Smolin, JM Gambetta, G Smith, Phys. Rev. Lett. 108, 070502 (2012).
           Open access: https://arxiv.org/abs/arXiv:1106.5458

    .. note::

        Linear least-squares constructs the full basis matrix :math:`A` as a dense
        numpy array so should not be used for than 5 or 6 qubits. For larger number
        of qubits try the
        :func:`~qiskit_experiments.tomography.fitters.linear_inversion`
        fitter function.

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
        weights: Optional array of weights for least squares objective.
        kwargs: additional kwargs for :func:`scipy.linalg.lstsq`.

    Raises:
        AnalysisError: If the fitted vector is not a square matrix

    Returns:
        The fitted matrix rho that maximizes the least-squares likelihood function.
    """
    # Probability vector y
    probability_data = frequency_data / shot_data

    # Basis matrix A
    basis_matrix = fitter_utils.stacked_basis_matrix(
        measurement_data, preparation_data, measurement_basis, preparation_basis
    )

    if weights is not None:
        basis_matrix = weights[:, None] * basis_matrix
        probability_data = weights * probability_data

    # Perform least squares fit using Scipy.linalg lstsq function
    lstsq_options = {"check_finite": False, "lapack_driver": "gelsy"}
    for key, val in kwargs.items():
        lstsq_options[key] = val
    sol, residues, rank, svals = la.lstsq(basis_matrix, probability_data, **lstsq_options)

    # Reshape fit to a density matrix
    size = len(sol)
    dim = int(np.sqrt(size))
    if dim * dim != size:
        raise AnalysisError("Least-squares fitter: invalid result shape.")
    rho_fit = np.reshape(sol, (dim, dim), order="F")

    # Rescale fitted density matrix be positive-semidefinite
    if psd is True:
        rho_fit = fitter_utils.make_positive_semidefinite(rho_fit)

    if trace is not None:
        rho_fit *= trace / np.trace(rho_fit)

    return {"value": rho_fit, "fit": {"residues": residues, "rank": rank, "singular_values": svals}}


def scipy_gaussian_lstsq(
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    frequency_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_basis: Optional[FitterBasis] = None,
    preparation_basis: Optional[FitterBasis] = None,
    psd: bool = True,
    trace: Optional[float] = None,
    **kwargs,
) -> Dict:
    r"""Gaussian linear least-squares tomography fitter.

    .. note::

        This function calls :func:`scipy_linear_lstsq` with a Gaussian weights
        vector. Refer to its documentation for additional details.

    This fitter uses the :func:`scipy_linear_lstsq` fitter to reconstructs
    the maximum-likelihood estimate of the Gaussian weighted least-squares
    log-likelihood function

    .. math::
        \hat{rho} &= \mbox{argmin} -\log\mathcal{L}{\rho} \\
        -\log\mathcal{L}(\rho)
            &= \sum_i \frac{1}{\sigma_i^2}(\mbox{Tr}[E_j\rho] - \hat{p}_i)^2
             = \|W(Ax -y) \|_2^2

    The Gaussian weights are estimated from the observed frequency and shot data
    using

    .. math::

        \sigma_i &= \sqrt{\frac{q_i(1 - q_i)}{n_i}} \\
        q_i &= \frac{f_i + \beta}{n_i + K \beta}

    where :math:`q_i` are hedged probabilities which are rescaled to avoid
    0 and 1 values using the "add-beta" rule, with :math:`\beta=0.5`, and
    :math:`K=2^m` the number of measurement outcomes for each basis measurement.

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
        kwargs: additional kwargs for :func:`scipy.linalg.lstsq`.

    Raises:
        AnalysisError: If the fitted vector is not a square matrix

    Returns:
        The fitted matrix rho that maximizes the least-squares likelihood function.
    """
    num_outcomes = measurement_basis.num_outcomes ** len(measurement_data[0])
    weights = fitter_utils.binomial_weights(frequency_data, shot_data, num_outcomes, beta=0.5)
    return scipy_linear_lstsq(
        measurement_data,
        preparation_data,
        frequency_data,
        shot_data,
        measurement_basis=measurement_basis,
        preparation_basis=preparation_basis,
        psd=psd,
        trace=trace,
        weights=weights,
        **kwargs,
    )
