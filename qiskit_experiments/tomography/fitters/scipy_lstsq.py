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
Data processing for linear least square tomography fitters
"""

from typing import Optional, Dict
import numpy as np
import scipy.linalg as la

from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.tomography.basis import FitterBasis
from .fitter_utils import guassian_lstsq_data, make_positive_semidefinite


def scipy_guassian_lstsq(
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    frequency_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_basis: Optional[FitterBasis] = None,
    preparation_basis: Optional[FitterBasis] = None,
    psd: bool = True,
    trace: Optional[float] = None,
    hedging_beta: float = 0.5,
    binomial_weights: bool = True,
    custom_weights: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict:
    r"""Weighted least-squares tomography fitter.

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
        hedging_beta: Hedging parameter for converting frequencies to
                      probabilities. If 0 hedging is disabled.
        binomial_weights: Compute binomial weights from frequency data.
        custom_weights: Optional, custom weights for fitter. If specified
                        binomial weights will be disabled.
        kwargs: additional kwargs for scipy.linalg.lstsq

    Raises:
        AnalysisError: If the fitted vector is not a square matrix

    Returns:
        The fitted matrix rho that minimizes
        :math:`||A \cdot \text{vec}(\text{rho}) - \text{data}||_2`.

    .. note::

        **Objective function**

        This fitter solves the least-squares minimization problem

        .. math::

            x = \mbox{argmin} ||A \cdot x - y ||_2

        where

        * :math:`A` is the matrix of measurement operators
          :math:`A = \sum_i |i\rangle\!\langle\!\langl M_i|`
        * :math:`y` is the vector of expectation value data for each projector
          corresponding to estimates of :math:`b_i = Tr[M_i \cdot x]`.
        * :math:`x` is the vectorized density matrix (or Choi-matrix) to be fitted
          :math:`x = |\rho\rangle\\!\rangle`.

        **PSD Constraint**

        Since this minimization problem is unconstrained the returned fitted
        matrix may not be postive semidefinite (PSD). To enforce the PSD
        constraint the fitted matrix is rescaled using the method proposed in
        Reference [1].

        **Trace constraint**

        In general the trace of the fitted matrix will be determined by the
        input data. If a trace constraint is specified the fitted matrix
        will be rescaled to have this trace by
        :math:`\text{rho} = \frac{\text{trace}\cdot\text{rho}}{\text{trace}(\text{rho})}`

        **Hedging beta**

        Hedged probabilities are used when the number of outcomes exceeds the number
        of measurement shots so there is a high probability 0 frequency outcomes
        are due to the limited number of shots, not because the true probability is
        zero. In this case probabilities are biased away from extreme values of 0
        or 1 via ``p[i] = (frequencies[i] + beta) / (shots[i] + num_outcomes * beta)``.
        See reference [2] for more details.

    References:
        1. J Smolin, JM Gambetta, G Smith, Phys. Rev. Lett. 108, 070502 (2012).
           Open access: https://arxiv.org/abs/arXiv:1106.5458
        2. R Blume-Kohout, Phys. Rev. Lett. 105, 200504 (2010).
           Open access: https://arxiv.org/abs/1001.2029
    """
    # Linear least squares data
    basis_matrix, probability_data = guassian_lstsq_data(
        measurement_data,
        preparation_data,
        frequency_data,
        shot_data,
        measurement_basis,
        preparation_basis,
        hedging_beta=hedging_beta,
        binomial_weights=binomial_weights,
        custom_weights=custom_weights,
    )

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
        rho_fit = make_positive_semidefinite(rho_fit)

    if trace is not None:
        rho_fit *= trace / np.trace(rho_fit)

    return {"value": rho_fit, "fit": {"residues": residues, "rank": rank, "singular_values": svals}}
