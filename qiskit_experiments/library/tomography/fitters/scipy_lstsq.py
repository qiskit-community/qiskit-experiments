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

from typing import Optional, Dict, Tuple
import numpy as np
import scipy.linalg as la
from qiskit.utils import deprecate_function
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.library.tomography.basis import (
    MeasurementBasis,
    PreparationBasis,
)
from . import lstsq_utils

# Note this warning doesnt show up when run in analysis so we
# also add a warning when setting the option value that calls this function

# pylint: disable = bad-docstring-quotes
@deprecate_function(
    "The scipy lstsq tomography fitters are deprecated as of 0.4 and will "
    "be removed after the 0.5 release. Use the `linear_lstsq`, "
    "`cvxpy_linear_lstsq`, or `cvxpy_gaussian_lstsq` fitters instead."
)
def scipy_linear_lstsq(
    outcome_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: Optional[MeasurementBasis] = None,
    preparation_basis: Optional[PreparationBasis] = None,
    measurement_qubits: Optional[Tuple[int, ...]] = None,
    preparation_qubits: Optional[Tuple[int, ...]] = None,
    weights: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[np.ndarray, Dict]:
    r"""Weighted linear least-squares tomography fitter.

    Overview
        This fitter reconstructs the maximum-likelihood estimate by using
        :func:`scipy.linalg.lstsq` to minimize the least-squares negative log
        likelihood function

        .. math::
            \hat{\rho}
                &= -\mbox{argmin }\log\mathcal{L}{\rho} \\
                &= \mbox{argmin }\sum_i w_i^2(\mbox{Tr}[E_j\rho] - \hat{p}_i)^2 \\
                &= \mbox{argmin }\|W(Ax - y) \|_2^2

        where

        - :math:`A = \sum_j |j \rangle\!\langle\!\langle E_j|` is the matrix of measured
          basis elements.
        - :math:`W = \sum_j w_j|j\rangle\!\langle j|` is an optional diagonal weights
          matrix if an optional weights vector is supplied.
        - :math:`y = \sum_j \hat{p}_j |j\langle` is the vector of estimated measurement
          outcome probabilites for each basis element.
        - :math:`x = |\rho\rangle\!\rangle` is the vectorized density matrix.

    .. note::

        Linear least-squares constructs the full basis matrix :math:`A` as a dense
        numpy array so should not be used for than 5 or 6 qubits. For larger number
        of qubits try the
        :func:`~qiskit_experiments.library.tomography.fitters.linear_inversion`
        fitter function.

    Args:
        outcome_data: measurement outcome frequency data.
        shot_data: basis measurement total shot data.
        measurement_data: measurement basis indice data.
        preparation_data: preparation basis indice data.
        measurement_basis: Optional, measurement matrix basis.
        preparation_basis: Optional, preparation matrix basis.
        measurement_qubits: Optional, the physical qubits that were measured.
                            If None they are assumed to be ``[0, ..., M-1]`` for
                            M measured qubits.
        preparation_qubits: Optional, the physical qubits that were prepared.
                            If None they are assumed to be ``[0, ..., N-1]`` for
                            N prepared qubits.
        weights: Optional array of weights for least squares objective.
        kwargs: additional kwargs for :func:`scipy.linalg.lstsq`.

    Raises:
        AnalysisError: If the fitted vector is not a square matrix

    Returns:
        The fitted matrix rho that maximizes the least-squares likelihood function.
    """
    basis_matrix, probability_data = lstsq_utils.lstsq_data(
        outcome_data,
        shot_data,
        measurement_data,
        preparation_data,
        measurement_basis=measurement_basis,
        preparation_basis=preparation_basis,
        measurement_qubits=measurement_qubits,
        preparation_qubits=preparation_qubits,
    )

    if weights is not None:
        basis_matrix = weights[:, None] * basis_matrix
        probability_data = weights * probability_data

    # Perform least squares fit using Scipy.linalg lstsq function
    lstsq_options = {"check_finite": False, "lapack_driver": "gelsy"}
    for key, val in kwargs.items():
        lstsq_options[key] = val
    sol, _, _, _ = la.lstsq(basis_matrix, probability_data, **lstsq_options)

    # Reshape fit to a density matrix
    size = len(sol)
    dim = int(np.sqrt(size))
    if dim * dim != size:
        raise AnalysisError("Least-squares fitter: invalid result shape.")
    rho_fit = np.reshape(sol, (dim, dim), order="F")

    return rho_fit, {}


def scipy_gaussian_lstsq(
    outcome_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: Optional[MeasurementBasis] = None,
    preparation_basis: Optional[PreparationBasis] = None,
    measurement_qubits: Optional[Tuple[int, ...]] = None,
    preparation_qubits: Optional[Tuple[int, ...]] = None,
    **kwargs,
) -> Dict:
    r"""Gaussian linear least-squares tomography fitter.

    .. note::

        This function calls :func:`scipy_linear_lstsq` with a Gaussian weights
        vector. Refer to its documentation for additional details.

    Overview
        This fitter uses the :func:`scipy_linear_lstsq` fitter to reconstructs
        the maximum-likelihood estimate of the Gaussian weighted least-squares
        log-likelihood function

        .. math::
            \hat{rho} &= \mbox{argmin} -\log\mathcal{L}{\rho} \\
            -\log\mathcal{L}(\rho)
                &= \sum_i \frac{1}{\sigma_i^2}(\mbox{Tr}[E_j\rho] - \hat{p}_i)^2
                = \|W(Ax -y) \|_2^2

    Additional Details
        The Gaussian weights are estimated from the observed frequency and shot data
        using

        .. math::

            \sigma_i &= \sqrt{\frac{q_i(1 - q_i)}{n_i}} \\
            q_i &= \frac{f_i + \beta}{n_i + K \beta}

        where :math:`q_i` are hedged probabilities which are rescaled to avoid
        0 and 1 values using the "add-beta" rule, with :math:`\beta=0.5`, and
        :math:`K=2^m` the number of measurement outcomes for each basis measurement.

    Args:
        outcome_data: measurement outcome frequency data.
        shot_data: basis measurement total shot data.
        measurement_data: measurement basis indice data.
        preparation_data: preparation basis indice data.
        measurement_basis: Optional, measurement matrix basis.
        preparation_basis: Optional, preparation matrix basis.
        measurement_qubits: Optional, the physical qubits that were measured.
                            If None they are assumed to be ``[0, ..., M-1]`` for
                            M measured qubits.
        preparation_qubits: Optional, the physical qubits that were prepared.
                            If None they are assumed to be ``[0, ..., N-1]`` for
                            N prepared qubits.
        kwargs: additional kwargs for :func:`scipy.linalg.lstsq`.

    Raises:
        AnalysisError: If the fitted vector is not a square matrix

    Returns:
        The fitted matrix rho that maximizes the least-squares likelihood function.
    """
    weights = lstsq_utils.binomial_weights(outcome_data, shot_data, beta=0.5)
    return scipy_linear_lstsq(
        outcome_data,
        shot_data,
        measurement_data,
        preparation_data,
        measurement_basis=measurement_basis,
        preparation_basis=preparation_basis,
        measurement_qubits=measurement_qubits,
        preparation_qubits=preparation_qubits,
        weights=weights,
        **kwargs,
    )
