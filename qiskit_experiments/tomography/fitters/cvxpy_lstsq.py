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

from typing import Optional, Dict
import numpy as np
from scipy import sparse as sps

from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.tomography.basis import FitterBasis
from .cvxpy_utils import requires_cvxpy, cvxpy
from .fitter_utils import guassian_lstsq_data


@requires_cvxpy
def cvxpy_guassian_lstsq(
    measurement_data: np.ndarray,
    preparation_data: np.ndarray,
    frequency_data: np.ndarray,
    shot_data: np.ndarray,
    measurement_basis: Optional[FitterBasis] = None,
    preparation_basis: Optional[FitterBasis] = None,
    psd: bool = True,
    trace_preserving: bool = False,
    trace: Optional[float] = None,
    hedging_beta: float = 0.5,
    binomial_weights: bool = True,
    custom_weights: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict:
    r"""
    Reconstruct a quantum state using CVXPY convex optimization.

    **Objective function**

    This fitter solves the least-squares minimization problem

        .. math::

            x = \mbox{argmin} ||A \cdot x - y ||_2

    subject to

    * :math:`x >> 0` (PSD, optional)
    * :math:`\text{trace}(x) = t` (trace, optional)
    * :math:`\text{partial_trace}(x)` = identity (trace_preserving, optional)

    where

    * :math:`A` is the matrix of measurement operators
      :math:`A = \sum_i |i\rangle\!\langle\!\langl M_i|`
    * :math:`y` is the vector of expectation value data for each projector
      corresponding to estimates of :math:`b_i = Tr[M_i \cdot x]`.
    * :math:`x` is the vectorized density matrix (or Choi-matrix) to be fitted
      :math:`x = |\rho\rangle\\!\rangle`.

    **PSD constraint**

    The PSD keyword constrains the fitted matrix to be
    postive-semidefinite, which makes the optimization problem a SDP. If
    PSD=False the fitted matrix will still be constrained to be Hermitian,
    but not PSD. In this case the optimization problem becomes a SOCP.

    **Trace constraint**

    The trace keyword constrains the trace of the fitted matrix. If
    trace=None there will be no trace constraint on the fitted matrix.
    This constraint should not be used for process tomography and the
    trace preserving constraint should be used instead.

    **Trace preserving (TP) constraint**

    The trace_preserving keyword constrains the fitted matrix to be TP.
    This should only be used for process tomography, not state tomography.
    Note that the TP constraint implicitly enforces the trace of the fitted
    matrix to be equal to the square-root of the matrix dimension. If a
    trace constraint is also specified that differs from this value the fit
    will likely fail.

    **CVXPY Solvers**

    Various solvers can be called in CVXPY using the `solver` keyword
    argument. See the `CVXPY documentation
    <https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options>`_
    for more information on solvers.

    Args:
        measurement_data: measurement basis indice data.
        preparation_data: preparation basis indice data.
        frequency_data: basis measurement frequency data.
        shot_data: basis measurement total shot data.
        measurement_basis: measurement matrix basis.
        preparation_basis: preparation matrix basis.
        psd: If True rescale the eigenvalues of fitted matrix to be positive
             semidefinite (default: True)
        trace_preserving: Enforce the fitted matrix to be
            trace preserving when fitting a Choi-matrix in quantum process
            tomography (default: False).
        trace: trace constraint for the fitted matrix (default: None).
        hedging_beta: Hedging parameter for converting frequencies to
                      probabilities. If 0 hedging is disabled.
        binomial_weights: Compute binomial weights from frequency data.
        custom_weights: Optional, custom weights for fitter. If specified
                        binomial weights will be disabled.
        kwargs: kwargs for cvxpy solver.

    Raises:
        QiskitError: If CVXPy is not installed on the current system.
        AnalysisError: If analysis fails.

    Returns:
        The fitted matrix rho that minimizes :math:`||basis_matrix * vec(rho) - data||_2`.
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

    analysis_result = {"value": rho_fit}
    return analysis_result


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
