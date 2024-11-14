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

from typing import Callable, List, Tuple, Optional, Union
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


def solve_iteratively(
    problem: Problem, initial_iters: int, scale: int = 2, solver: str = "SCS", **solve_kwargs
) -> None:
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
        solver: The solver to use. Defaults to the Splitting Conic Solver.
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
        problem.solve(solver=solver, **solve_kwargs)
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
            raise AnalysisError(f"CVXPY solver failed with problem status '{problem.status}'.")


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


def trace_constraint(
    mat_r: Union[Variable, List[Variable]],
    mat_i: Union[Variable, List[Variable]],
    trace: complex,
    hermitian: bool = False,
) -> List[Constraint]:
    """Return CVXPY trace constraints for a complex matrix.

    Args:
        mat_r: The CVXPY variable for the real part of the matrix.
        mat_i: The CVXPY variable for the complex part of the matrix.
        trace: The value for the trace constraint.
        hermitian: If the input variables are Hermitian, only the real trace constraint
                   is required.
    Returns:
        A list of constraints on the real and imaginary parts.

    Raises:
        TypeError: If input variables are not valid.
    """
    if isinstance(mat_r, (list, tuple)):
        arg_r = cvxpy.sum(mat_r)
    elif isinstance(mat_r, Variable):
        arg_r = mat_r
    else:
        raise TypeError("Input must be a cvxpy variable or list of variables")
    cons = [cvxpy.trace(arg_r) == np.real(trace)]

    if hermitian:
        return cons

    # If not hermitian add imaginary trace constraint
    if isinstance(mat_i, (list, tuple)):
        arg_i = cvxpy.sum(mat_i)
    elif isinstance(mat_i, Variable):
        arg_i = mat_i
    else:
        raise TypeError("Input must be a cvxpy variable or list of variables")
    cons.append(cvxpy.trace(arg_i) == np.imag(trace))

    return cons


def partial_trace_constaint(
    mat_r: Variable,
    mat_i: Variable,
    constraint: np.ndarray,
) -> List[Constraint]:
    """Return CVXPY partial trace constraints for a complex matrix.

    Args:
        mat_r: The CVXPY variable for the real part of the matrix.
        mat_i: The CVXPY variable for the complex part of the matrix.
        constraint: The constraint matrix to set the partial trace to.

    Returns:
        A list of constraints on the real and imaginary parts.

    Raises:
        TypeError: If input variables are not valid.
    """
    sdim = mat_r.shape[0]
    output_dim = constraint.shape[0]
    input_dim = sdim // output_dim
    ptr = partial_trace_super(input_dim, output_dim)
    vec_cons = np.ravel(constraint, order="F")
    return [
        ptr @ cvxpy.vec(mat_r, order="F") == vec_cons.real.round(12),
        ptr @ cvxpy.vec(mat_i, order="F") == vec_cons.imag.round(12),
    ]


def trace_preserving_constaint(
    mat_r: Union[Variable, List[Variable]],
    mat_i: Union[Variable, List[Variable]],
    input_dim: Optional[int] = None,
    hermitian: bool = False,
) -> List[Constraint]:
    """Return CVXPY trace preserving constraints for a complex matrix.

    Args:
        mat_r: The CVXPY variable for the real part of the matrix.
        mat_i: The CVXPY variable for the complex part of the matrix.
        input_dim: Optional, the input dimension for the system channel if the input
                   and output dimensions are not equal.
        hermitian: If the input variables are Hermitian, only the real trace constraint
                   is required.

    Returns:
        A list of constraints on the real and imaginary parts.

    Raises:
        TypeError: If input variables are not valid.
    """
    if isinstance(mat_r, (tuple, list)):
        sdim = mat_r[0].shape[0]
        arg_r = cvxpy.sum(mat_r)
    elif isinstance(mat_r, Variable):
        sdim = mat_r.shape[0]
        arg_r = mat_r
    else:
        raise TypeError("Input must be a cvxpy variable or list of variables")
    if input_dim is None:
        input_dim = int(np.sqrt(sdim))
    output_dim = sdim // input_dim

    ptr = partial_trace_super(input_dim, output_dim)
    cons = [ptr @ cvxpy.vec(arg_r, order="F") == np.identity(input_dim).ravel()]

    if hermitian:
        return cons

    # If not hermitian add imaginary partial trace constraint
    if isinstance(mat_i, (tuple, list)):
        arg_i = cvxpy.sum(mat_i)
    elif isinstance(mat_i, Variable):
        arg_i = mat_i
    else:
        raise TypeError("Input must be a cvxpy variable or list of variables")
    cons.append(ptr @ cvxpy.vec(arg_i, order="F") == np.zeros(input_dim**2))
    return cons


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
