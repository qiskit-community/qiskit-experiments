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
Utility functions for CVXPy module
"""

from typing import Callable, List, Tuple, Optional
import functools
import numpy as np
import scipy.sparse as sps

from qiskit_experiments.exceptions import AnalysisError

# Check if CVXPY package is installed
try:
    import cvxpy
    from cvxpy import Problem, Variable
    from cvxpy.constraints.constraint import Constraint

    HAS_CVXPY = True

except ImportError:
    cvxpy = None

    HAS_CVXPY = False

    # Used for type hints
    Problem = None
    Variable = None
    Constraint = None


def requires_cvxpy(func: Callable) -> Callable:
    """Function decorator for functions requiring CVXPy.

    Args:
        func: a function requiring CVXPy.

    Returns:
        The decorated function.

    Raises:
        QiskitError: If CVXPy is not installed.
    """

    @functools.wraps(func)
    def decorated_func(*args, **kwargs):
        if not HAS_CVXPY:
            raise ImportError(
                f"The CVXPY package is required to for {func}."
                "You can install it with 'pip install cvxpy'."
            )
        return func(*args, **kwargs)

    return decorated_func


def solve_iteratively(problem: Problem, initial_iters: int, scale: int = 2, **solve_kwargs) -> None:
    """Solve a CVXPY problem increasing iterations if solution is inaccurate.

    If the problem is not solved with the ``initial_iters`` value of
    iterations the number of iterations will be doubled up to the
    specified ``"max_iters"`` value in the solve_kwargs. If no max
    iters is specified this will be set to 4 times the initial iters
    values.

    Args:
        problem: A CVXPY Problem to solve
        initial_iters: The initial number of max iterations to use
                       when solving the problem
        scale: Scale factor for increasing the initial_iters up to
               max_iters at each step (Default: 2).
        solve_kwargs: kwargs for problem.solve method.

    Raises:
        AnalysisError: If the CVXPY solve fails to return an optimal or
                       optimal_inaccurate solution.
    """
    current_max_iters = initial_iters
    final_max_iters = solve_kwargs.get("max_iters", 2 * scale * initial_iters)
    problem_solved = False
    while not problem_solved:
        solve_kwargs["max_iters"] = current_max_iters
        problem.solve(**solve_kwargs)
        if problem.status in ["optimal_inaccurate", "optimal"]:
            problem_solved = True
        elif problem.status == "unbounded_inaccurate":
            if scale > 1 and current_max_iters < final_max_iters:
                current_max_iters = int(scale * current_max_iters)
            else:
                raise AnalysisError(
                    "CVXPY solver failed to find an optimal solution in "
                    "the given number of iterations. Try setting a larger "
                    "value for 'max_iters' solver option."
                )
        else:
            raise AnalysisError(
                "CVXPY solver failed with problem status '{}'.".format(problem.status)
            )


def set_default_sdp_solver(solver_kwargs: dict):
    """Set default SDP solver from installed solvers."""
    if "solver" in solver_kwargs:
        return
    if "CVXOPT" in cvxpy.installed_solvers():
        solver_kwargs["solver"] = "CVXOPT"
    elif "MOSEK" in cvxpy.installed_solvers():
        solver_kwargs["solver"] = "MOSEK"


def complex_matrix_variable(
    dim: int, hermitian: bool = False, psd: bool = False, trace: Optional[complex] = None
) -> Tuple[Variable, Variable, List[Constraint]]:
    """Construct a pair of real variables and constraints for a Hermitian matrix

    Args:
        dim: The dimension of the complex square matrix.
        hermitian: If True add constraint that the matrix is Hermitian.
                   (Default: False).
        psd: If True add a constraint that the matrix is positive
             semidefinite (Default: False).
        trace: Optional, add a constraint that the trace of the matrix is
               the specified value.

    Returns:
        A tuple ``(mat.real, mat.imag, constraints)`` of two real CVXPY
        matrix variables, and constraints.
    """
    mat_r = cvxpy.Variable((dim, dim))
    mat_i = cvxpy.Variable((dim, dim))
    cons = []

    if hermitian:
        cons += hermitian_constraint(mat_r, mat_i)
    if trace is not None:
        cons += trace_constraint(mat_r, mat_i, trace)
    if psd:
        cons += psd_constraint(mat_r, mat_i)
    return mat_r, mat_i, cons


def hermitian_constraint(mat_r: Variable, mat_i: Variable) -> List[Constraint]:
    """Return CVXPY constraint for a Hermitian matrix variable.

    Args:
        mat_r: The CVXPY variable for the real part of the matrix.
        mat_i: The CVXPY variable for the complex part of the matrix.

    Returns:
        A list of constraints on the real and imaginary parts.
    """
    return [mat_r == mat_r.T, mat_i == -mat_i.T]


def psd_constraint(mat_r: Variable, mat_i: Variable) -> List[Constraint]:
    """Return CVXPY Hermitian constraints for a complex matrix.

    Args:
        mat_r: The CVXPY variable for the real part of the matrix.
        mat_i: The CVXPY variable for the complex part of the matrix.

    Returns:
        A list of constraints on the real and imaginary parts.
    """
    bmat = cvxpy.bmat([[mat_r, -mat_i], [mat_i, mat_r]])
    return [bmat >> 0]


def trace_constraint(mat_r: Variable, mat_i: Variable, trace: complex) -> List[Constraint]:
    """Return CVXPY trace constraints for a complex matrix.

    Args:
        mat_r: The CVXPY variable for the real part of the matrix.
        mat_i: The CVXPY variable for the complex part of the matrix.
        trace: The value for the trace constraint.

    Returns:
        A list of constraints on the real and imaginary parts.
    """
    return [cvxpy.trace(mat_r) == cvxpy.real(trace), cvxpy.trace(mat_i) == cvxpy.imag(trace)]


def trace_preserving_constraint(mat_r: Variable, mat_i: Variable) -> List[Constraint]:
    """Return CVXPY trace preserving constraints for a complex matrix.

    Args:
        mat_r: The CVXPY variable for the real part of the matrix.
        mat_i: The CVXPY variable for the complex part of the matrix.

    Returns:
        A list of constraints on the real and imaginary parts.
    """
    sdim = int(np.sqrt(mat_r.shape[0]))
    ptr = partial_trace_super(sdim, sdim)
    return [
        ptr @ cvxpy.vec(mat_r) == np.identity(sdim).ravel(),
        ptr @ cvxpy.vec(mat_i) == np.zeros(sdim * sdim),
    ]


@functools.lru_cache(3)
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
