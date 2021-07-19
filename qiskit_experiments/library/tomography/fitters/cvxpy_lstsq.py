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
Contrained convex least-squares tomography fitter.
"""

from typing import Optional, Dict, List, Tuple
import numpy as np
from scipy import sparse as sps

from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.library.tomography.basis import (
    BaseFitterMeasurementBasis,
    BaseFitterPreparationBasis,
)
from .cvxpy_utils import requires_cvxpy, cvxpy
from . import fitter_utils


@requires_cvxpy
def cvxpy_linear_lstsq(
    outcome_data: List[np.ndarray],
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: BaseFitterMeasurementBasis,
    preparation_basis: Optional[BaseFitterPreparationBasis] = None,
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
          to be a postive-semidefinite matrix.
        - *Trace* (``trace=t``): :math:`\mbox{Tr}(\rho) = t` is constained to have
          the specified trace.
        - *Trace preserving* (``trace_preserving=True``): When performing process
          tomography the Choi-state :math:`\rho` represents is contstained to be
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
        outcome_data: list of outcome frequency data.
        shot_data: basis measurement total shot data.
        measurement_data: measurement basis indice data.
        preparation_data: preparation basis indice data.
        measurement_basis: measurement matrix basis.
        preparation_basis: Optional, preparation matrix basis.
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
    basis_matrix, probability_data = fitter_utils.lstsq_data(
        outcome_data,
        shot_data,
        measurement_data,
        preparation_data,
        measurement_basis,
        preparation_basis=preparation_basis,
    )

    if weights is not None:
        basis_matrix = weights[:, None] * basis_matrix
        probability_data = weights * probability_data

    # SDP VARIABLES

    # Since CVXPY only works with real variables we must specify the real
    # and imaginary parts of rho seperately: rho = rho_r + 1j * rho_i

    dim = int(np.sqrt(basis_matrix.shape[1]))
    rho_r = cvxpy.Variable((dim, dim), symmetric=True)
    rho_i = cvxpy.Variable((dim, dim))

    # CONSTRAINTS

    # The constraint that rho is Hermitian (rho.H = rho)
    # transforms to the two constraints
    #   1. rho_r.T = rho_r.T  (real part is symmetric)
    #   2. rho_i.T = -rho_i.T  (imaginary part is anti-symmetric)

    cons = [rho_i == -rho_i.T]

    # Trace constraint: note this should not be used at the same
    # time as the trace preserving constraint.
    if trace is not None:
        cons.append(cvxpy.trace(rho_r) == trace)

    # Since we can only work with real matrices in CVXPY we can specify
    # a complex PSD constraint as
    #   rho >> 0 iff [[rho_r, -rho_i], [rho_i, rho_r]] >> 0

    if psd is True:
        rho = cvxpy.bmat([[rho_r, -rho_i], [rho_i, rho_r]])
        cons.append(rho >> 0)

    # Trace preserving constraint when fitting Choi-matrices for
    # quantum process tomography. Note that this adds an implicity
    # trace constraint of trace(rho) = sqrt(len(rho)) = dim
    # if a different trace constraint is specified above this will
    # cause the fitter to fail.

    if trace_preserving is True:
        sdim = int(np.sqrt(dim))
        ptr = partial_trace_super(sdim, sdim)
        cons.append(ptr @ cvxpy.vec(rho_r) == np.identity(sdim).ravel())
        cons.append(ptr @ cvxpy.vec(rho_i) == np.zeros(sdim * sdim))

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

    # SDP objective function
    arg = bm_r @ cvxpy.vec(rho_r) - bm_i @ cvxpy.vec(rho_i) - probability_data
    obj = cvxpy.Minimize(cvxpy.norm(arg, p=2))

    # Solve SDP
    prob = cvxpy.Problem(obj, cons)
    iters = 5000
    max_iters = kwargs.get("max_iters", 20000)
    # Set default solver if none is specified
    if "solver" not in kwargs:
        if "CVXOPT" in cvxpy.installed_solvers():
            kwargs["solver"] = "CVXOPT"
        elif "MOSEK" in cvxpy.installed_solvers():
            kwargs["solver"] = "MOSEK"

    problem_solved = False
    while not problem_solved:
        kwargs["max_iters"] = iters
        prob.solve(**kwargs)
        if prob.status in ["optimal_inaccurate", "optimal"]:
            problem_solved = True
        elif prob.status == "unbounded_inaccurate":
            if iters < max_iters:
                iters *= 2
            else:
                raise AnalysisError(
                    "CVXPY fit failed, probably not enough iterations for the " "solver"
                )
        elif prob.status in ["infeasible", "unbounded"]:
            raise AnalysisError(
                "CVXPY fit failed, problem status {} which should not " "happen".format(prob.status)
            )
        else:
            raise AnalysisError("CVXPY fit failed, reason unknown")

    rho_fit = rho_r.value + 1j * rho_i.value
    metadata = {
        "cvxpy_solver": prob.solver_stats.solver_name,
        "cvxpy_status": prob.status,
    }
    return rho_fit, metadata


@requires_cvxpy
def cvxpy_gaussian_lstsq(
    outcome_data: List[np.ndarray],
    shot_data: np.ndarray,
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    measurement_basis: BaseFitterMeasurementBasis,
    preparation_basis: Optional[BaseFitterPreparationBasis] = None,
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
        outcome_data: list of outcome frequency data.
        shot_data: basis measurement total shot data.
        measurement_data: measurement basis indice data.
        preparation_data: preparation basis indice data.
        measurement_basis: measurement matrix basis.
        preparation_basis: Optional, preparation matrix basis.
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
    num_outcomes = num_outcomes = [measurement_basis.num_outcomes(i) for i in measurement_data]
    weights = fitter_utils.binomial_weights(outcome_data, shot_data, num_outcomes, beta=0.5)
    return cvxpy_linear_lstsq(
        outcome_data,
        shot_data,
        measurement_data,
        preparation_data,
        measurement_basis,
        preparation_basis=preparation_basis,
        psd=psd,
        trace=trace,
        trace_preserving=trace_preserving,
        weights=weights,
        **kwargs,
    )


def partial_trace_super(dim1: int, dim2: int) -> np.array:
    """
    Return the partial trace superoperator in the column-major basis.

    This returns the superoperator S_TrB such that:
        S_TrB * vec(rho_AB) = vec(rho_A)
    for rho_AB = kron(rho_A, rho_B)

    Args:
        dim1: the dimension of the system not being traced
        dim2: the dimension of the system being traced over

    Returns:
        A Numpy array of the partial trace superoperator S_TrB.
    """

    iden = sps.identity(dim1)
    ptr = sps.csr_matrix((dim1 * dim1, dim1 * dim2 * dim1 * dim2))

    for j in range(dim2):
        v_j = sps.coo_matrix(([1], ([0], [j])), shape=(1, dim2))
        tmp = sps.kron(iden, v_j.tocsr())
        ptr += sps.kron(tmp, tmp)

    return ptr
