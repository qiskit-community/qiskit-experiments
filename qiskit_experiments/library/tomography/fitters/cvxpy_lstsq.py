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
Constrained convex least-squares tomography fitter.
"""

from typing import Optional, Dict, Tuple
import numpy as np

from qiskit_experiments.library.tomography.basis import (
    MeasurementBasis,
    PreparationBasis,
)
from . import cvxpy_utils
from .cvxpy_utils import cvxpy
from . import lstsq_utils


@cvxpy_utils.requires_cvxpy
def cvxpy_linear_lstsq(
    outcome_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: Optional[MeasurementBasis] = None,
    preparation_basis: Optional[PreparationBasis] = None,
    measurement_qubits: Optional[Tuple[int, ...]] = None,
    preparation_qubits: Optional[Tuple[int, ...]] = None,
    psd: bool = True,
    trace_preserving: bool = False,
    trace: Optional[float] = None,
    weights: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[np.ndarray, Dict]:
    r"""Constrained weighted linear least-squares tomography fitter.

    Overview
        This fitter reconstructs the maximum-likelihood estimate by using
        ``cvxpy`` to minimize the constrained least-squares negative log
        likelihood function

        .. math::
            \hat{\rho}
                &= -\mbox{argmin }\log\mathcal{L}{\rho} \\
                &= \mbox{argmin }\sum_i w_i^2(\mbox{Tr}[E_j\rho] - \hat{p}_i)^2 \\
                &= \mbox{argmin }\|W(Ax - y) \|_2^2

        subject to

        - *Positive-semidefinite* (``psd=True``): :math:`\rho \gg 0` is constrained
          to be a positive-semidefinite matrix.
        - *Trace* (``trace=t``): :math:`\mbox{Tr}(\rho) = t` is constrained to have
          the specified trace.
        - *Trace preserving* (``trace_preserving=True``): When performing process
          tomography the Choi-state :math:`\rho` represents is constrained to be
          trace preserving.

        where

        - :math:`A` is the matrix of measurement operators
          :math:`A = \sum_i |i\rangle\!\langle\!\langle M_i|`
        - :math:`y` is the vector of expectation value data for each projector
          corresponding to estimates of :math:`b_i = Tr[M_i \cdot x]`.
        - :math:`x` is the vectorized density matrix (or Choi-matrix) to be fitted
          :math:`x = |\rho\rangle\\!\rangle`.

    .. note:

        Various solvers can be called in CVXPY using the `solver` keyword
        argument. When ``psd=True`` the optimization problem is a case of a
        *semidefinite program* (SDP) and requires a SDP compatible solver
        for CVXPY. CVXPY includes an SDP compatible solver `SCS`` but it
        is recommended to install the the open-source ``CVXOPT`` solver
        or one of the supported commercial solvers. See the `CVXPY
        documentation
        <https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options>`_
        for more information on solvers.

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
        psd: If True rescale the eigenvalues of fitted matrix to be positive
             semidefinite (default: True)
        trace_preserving: Enforce the fitted matrix to be
            trace preserving when fitting a Choi-matrix in quantum process
            tomography (default: False).
        trace: trace constraint for the fitted matrix (default: None).
        weights: Optional array of weights for least squares objective.
        kwargs: kwargs for cvxpy solver.

    Raises:
        QiskitError: If CVXPy is not installed on the current system.
        AnalysisError: If analysis fails.

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
        weights = weights / np.sqrt(np.sum(weights**2))
        basis_matrix = weights[:, None] * basis_matrix
        probability_data = weights * probability_data

    # Since CVXPY only works with real variables we must specify the real
    # and imaginary parts of rho separately: rho = rho_r + 1j * rho_i

    dim = int(np.sqrt(basis_matrix.shape[1]))
    rho_r, rho_i, cons = cvxpy_utils.complex_matrix_variable(
        dim, hermitian=True, psd=psd, trace=trace
    )

    # Trace preserving constraint when fitting Choi-matrices for
    # quantum process tomography. Note that this adds an implicitly
    # trace constraint of trace(rho) = sqrt(len(rho)) = dim
    # if a different trace constraint is specified above this will
    # cause the fitter to fail.
    if trace_preserving:
        cons += cvxpy_utils.trace_preserving_constraint(rho_r, rho_i)

    # OBJECTIVE FUNCTION

    # The function we wish to minimize is || arg ||_2 where
    #   arg =  bm * vec(rho) - data
    # Since we are working with real matrices in CVXPY we expand this as
    #   bm * vec(rho) = (bm_r + 1j * bm_i) * vec(rho_r + 1j * rho_i)
    #                 = bm_r * vec(rho_r) - bm_i * vec(rho_i)
    #                   + 1j * (bm_r * vec(rho_i) + bm_i * vec(rho_r))
    #                 = bm_r * vec(rho_r) - bm_i * vec(rho_i)
    # where we drop the imaginary part since the expectation value is real
    bm_r = np.real(basis_matrix)
    bm_i = np.imag(basis_matrix)
    arg = bm_r @ cvxpy.vec(rho_r) - bm_i @ cvxpy.vec(rho_i) - probability_data
    obj = cvxpy.Minimize(cvxpy.norm(arg, p=2))
    prob = cvxpy.Problem(obj, cons)

    # Solve SDP
    cvxpy_utils.set_default_sdp_solver(kwargs)
    cvxpy_utils.solve_iteratively(prob, 5000, **kwargs)

    # Return optimal values and problem metadata
    rho_fit = rho_r.value + 1j * rho_i.value
    metadata = {
        "cvxpy_solver": prob.solver_stats.solver_name,
        "cvxpy_status": prob.status,
    }
    return rho_fit, metadata


@cvxpy_utils.requires_cvxpy
def cvxpy_gaussian_lstsq(
    outcome_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: Optional[MeasurementBasis] = None,
    preparation_basis: Optional[PreparationBasis] = None,
    measurement_qubits: Optional[Tuple[int, ...]] = None,
    preparation_qubits: Optional[Tuple[int, ...]] = None,
    psd: bool = True,
    trace_preserving: bool = False,
    trace: Optional[float] = None,
    **kwargs,
) -> Dict:
    r"""Constrained Gaussian linear least-squares tomography fitter.

    .. note::

        This function calls :func:`cvxpy_linear_lstsq` with a Gaussian weights
        vector. Refer to its documentation for additional details.

    Overview
        This fitter reconstructs the maximum-likelihood estimate by using
        ``cvxpy`` to minimize the constrained least-squares negative log
        likelihood function

        .. math::
            \hat{\rho}
                &= \mbox{argmin} (-\log\mathcal{L}{\rho}) \\
                &= \mbox{argmin }\|W(Ax - y) \|_2^2 \\
            -\log\mathcal{L}(\rho)
                &= |W(Ax -y) \|_2^2 \\
                &= \sum_i \frac{1}{\sigma_i^2}(\mbox{Tr}[E_j\rho] - \hat{p}_i)^2

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
        psd: If True rescale the eigenvalues of fitted matrix to be positive
             semidefinite (default: True)
        trace_preserving: Enforce the fitted matrix to be
            trace preserving when fitting a Choi-matrix in quantum process
            tomography (default: False).
        trace: trace constraint for the fitted matrix (default: None).
        kwargs: kwargs for cvxpy solver.

    Raises:
        QiskitError: If CVXPY is not installed on the current system.
        AnalysisError: If analysis fails.

    Returns:
        The fitted matrix rho that maximizes the least-squares likelihood function.
    """
    weights = lstsq_utils.binomial_weights(outcome_data, shot_data, beta=0.5)
    return cvxpy_linear_lstsq(
        outcome_data,
        shot_data,
        measurement_data,
        preparation_data,
        measurement_basis=measurement_basis,
        preparation_basis=preparation_basis,
        measurement_qubits=measurement_qubits,
        preparation_qubits=preparation_qubits,
        psd=psd,
        trace=trace,
        trace_preserving=trace_preserving,
        weights=weights,
        **kwargs,
    )
