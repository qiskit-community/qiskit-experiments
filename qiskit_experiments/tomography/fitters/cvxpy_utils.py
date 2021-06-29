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

from typing import Callable
import functools

# Check if CVXPY package is installed
try:
    import cvxpy

    HAS_CVXPY = True
except ImportError:
    cvxpy = None
    HAS_CVXPY = False


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


class SDPSolverChecker:
    """Class for checking installed CVXPy SDP solvers"""

    _HAS_SDP_SOLVER = None
    _HAS_SDP_SOLVER_NOT_SCS = False
    _SDP_SOLVERS = set()

    def __init__(self):
        self._check_for_sdp_solver()

    @property
    def has_sdp_solver(self) -> bool:
        """Return True if CVXPy is installed with an SDP solver"""
        return SDPSolverChecker._HAS_SDP_SOLVER

    @property
    def has_sdp_solver_not_scs(self) -> bool:
        """Return True if CVXPy is installed with an SDP solver"""
        return SDPSolverChecker._HAS_SDP_SOLVER_NOT_SCS

    @property
    def sdp_solvers(self):
        """Return True if CVXPy is installed with an SDP solver other than SCS"""
        return self._SDP_SOLVERS

    @classmethod
    def _check_for_sdp_solver(cls):
        """Check if CVXPy solver is available"""
        if cls._HAS_SDP_SOLVER is None:
            cls._HAS_SDP_SOLVER = False
            if HAS_CVXPY:
                # pylint:disable=import-error
                solvers = cvxpy.installed_solvers()
                # Check for other SDP solvers cvxpy supports
                for solver in ["CVXOPT", "MOSEK"]:
                    if solver in solvers:
                        cls._SDP_SOLVERS.add(solver)
                        cls._HAS_SDP_SOLVER = True
                        cls._HAS_SDP_SOLVER_NOT_SCS = True
                if "SCS" in solvers:
                    # Try example problem to see if built with BLAS
                    # SCS solver cannot solver larger than 2x2 matrix
                    # problems without BLAS
                    try:
                        var = cvxpy.Variable((5, 5), PSD=True)
                        obj = cvxpy.Minimize(cvxpy.norm(var))
                        cvxpy.Problem(obj).solve(solver="SCS")
                        cls._SDP_SOLVERS.add("SCS")
                        cls._HAS_SDP_SOLVER = True
                    except cvxpy.error.SolverError:
                        pass
