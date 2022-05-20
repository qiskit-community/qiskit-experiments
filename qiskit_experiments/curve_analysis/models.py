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

# pylint: disable=abstract-method, no-member

"""Fit model wrapper of LMFIT to support multi-objective optimization."""

import time
import warnings
from typing import Callable, List, Dict, Tuple, Sequence, Iterator

import numpy as np
import uncertainties
from asteval import Interpreter, get_ast_names
from lmfit.minimizer import Minimizer
from lmfit.model import Model
from lmfit.parameter import Parameters

from qiskit_experiments.curve_analysis.curve_data import SeriesDef, SolverResult
from qiskit_experiments.exceptions import AnalysisError


class CurveModel(Model):
    """A model that describes the curve fitting function.

    The model manages generation of the fitting function, computation of
    residual value against given parameters as a cost function of optimization,
    and preparation of parameter objects passed to the solver instance.
    The :class:`CurveModel` is implemented on top of the `LMFIT`_ ``Model`` class
    with some extension for the multi objective optimization
    described in :ref:`curve_analysis_overview`.

    The CurveModel is instantiated with a list of :class:`.SeriesDef` instances,
    that may provide a string representation of the fitting function
    along with some metadata to manage data processing and visualization.

    The fitting function description is parsed with the `ASTEVAL`_ parser, which
    evaluates mathematical expressions. Note that mathematical functions are
    imported only from Python's math module and numpy for the safety on code execution.
    This mechanism allows us to easily serialize an analysis subclass instance
    that is initialized with fitting functions, i.e. serialization of callables is not necessary.
    In addition, the function description is now human-readable string,
    and thus it can be directly copied to analysis result for helping experimentalists to
    investigate the fitting outcome.

    Fitting parameters are also automatically extracted from the function description,
    and thus an analysis class author just needs to provide a function description
    when one wants to define new analysis class.

    Here is an example of acceptable function descriptions.

    .. parsed-literal::

        amp * sin(2 * pi * freq * x + phi) + base
        amp * cos(2 * pi * freq * x + phi) + base

    The parameter name ``x`` is reserved by the model to represent a parameter to be scanned.
    Thus, a composite model comprising two functions above consists of four independent parameters
    ``amp``, ``freq``, ``phi``, and ``base`` that will be evaluated by the curve fitter.
    Other names ``cos``, ``sin``, and ``pi`` are universal functions and
    a constant value imported from the Python's math module in this example.
    Note that when multiple series definitions are provided like this example,
    the parameters with the same name are shared among the fit functions.

    To compute the residual value, input and output values of each function are concatenated.
    Thus, a special parameter ``allocation`` is also reserved to map each element in ``x``
    to one of fitting functions with index ``i`` consisting the composite model
    :math:`F = F_0 \\oplus F_1 \\oplus ... F_i`. See :ref:`curve_analysis_overview` for details.
    This mapping is implicitly managed by the model.

    .. _LMFIT: https://lmfit.github.io/lmfit-py/model.html
    .. _ASTEVAL: https://asteval.readthedocs.io/en/latest/
    """

    def __init__(self, name: str, series_defs: Sequence[SeriesDef]):
        """Create new model.

        Args:
            name: Name of this model.
            series_defs: List of series definitions.
        """
        composite_func, unite_params = self._compose_functions(
            exprs=[series_def.fit_func for series_def in series_defs]
        )
        super().__init__(func=composite_func, name=name, param_names=unite_params)
        self._series_defs = tuple(series_defs)

    @staticmethod
    def _compose_functions(exprs: List[str]) -> Tuple[Callable, List[str]]:
        """A helper method to generate composite function.

        Args:
            exprs: List of model expressions.

        Returns:
            A tuple of composite fit function and its parameters.
        """
        curve_models = []
        curve_params = []
        for expr in exprs:
            if callable(expr):
                import inspect

                warnings.warn(
                    "Providing a callable to SeriesDef has been deprecated. "
                    "Now '.fit_func' is a string representation of model. "
                    "This warning will be removed and support for callable is dropped "
                    "in Qiskit Experiments 0.5.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                # Use callable as-is with parsing function signature
                expr_callable = expr
                signature = list(inspect.signature(expr_callable).parameters.keys())
                params = signature[1:]
            else:
                # Generate function from string
                expr_callable, params = _compile_expr(expr)
            curve_models.append(expr_callable)
            curve_params.append(params)

        def _composite(x, allocation, **kwargs):
            y = np.zeros(x.shape)
            for f_idx, (func, params) in enumerate(zip(curve_models, curve_params)):
                locs = allocation == f_idx
                y[locs] = func(x=x[locs], **{p: kwargs[p] for p in params})
            return y

        unite_params = []
        for params in curve_params:
            unite_params.extend(p for p in params if p not in unite_params)

        return _composite, unite_params

    @property
    def definitions(self) -> Iterator[SeriesDef]:
        """Return series definitions."""
        yield from self._series_defs

    def eval_with_uncertainties(
        self,
        x: np.ndarray,
        params: Dict[str, uncertainties.UFloat],
        model_index: int,
    ) -> np.ndarray:
        """Compute Y values with error propagation.

        Args:
            x: X values.
            params: Fitter parameters with uncertainties.
            model_index: Index of model to compute.

        Returns:
            Y values with uncertainty (uarray).
        """
        model_expr = self._series_defs[model_index].fit_func

        # Create function that performs error propagation
        func, this_params = _compile_expr(model_expr)
        sub_params = {pname: params[pname] for pname in this_params}
        wrapfunc = np.vectorize(uncertainties.wrap(func))

        return wrapfunc(x=x, **sub_params)

    def _parse_params(self):
        self.independent_vars = ["x", "allocation"]
        self._param_names = self._param_root_names
        self._func_haskeywords = True
        self._func_allargs = self.independent_vars + self.param_names
        self.def_vals = {}

    def __getstate__(self):
        # Local function cannot be pickled
        state = self.__dict__.copy()
        del state["func"]

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # Initialize non-pickled callable from string expressions
        composite_func, _ = self._compose_functions(
            exprs=[series_def.fit_func for series_def in state["_series_defs"]]
        )
        self.func = composite_func


class CurveSolver(Minimizer):
    """A solver class that performs curve fitting with :class:`CurveModel`.

    This class is a wrapper of the `LMFIT`_ ``Minimizer`` class that implements
    several curve fitting algorithms. By default, this class uses
    the scipy's Trust Region Reflective algorithm (`least squares`_) for
    the bounded optimization problem, i.e. parameters may have boundaries.

    This wrapper class overrides :meth:`.fit` method so that it
    returns a lightweight dataclass :class:`.SolverResult` that may be added to
    the analysis results. The original ``Minimizer`` class overrides ``self``
    with the fitting status and outcomes, but this object is too heavy
    to be stored in the analysis result payload because the minimizer instance keeps
    full fitting functionality for reanalyzing the result.
    The :class:`.SolverResult` dataclass drastically reduces the serialization overhead
    and the data volume saved in the cloud database, since it only keeps
    the output of the curve fitting solver which is necessary by the curve analysis
    post-process to finalize the analysis outcome.

    .. _least_squares: https://docs.scipy.org/doc/scipy/reference/generated/\
        scipy.optimize.least_squares.html
    .. _LMFIT: https://lmfit.github.io/lmfit-py/model.html
    """

    def __init__(
        self,
        model: CurveModel,
        params: Parameters,
        # We typically choose "leastsq" or "least_squares" based method, provided by scipy.
        # The former only supports the Levenberg-Marquardt, while the latter supports
        # the Trust Region Reflective algorithm for bounded optimization problem.
        # Use of "leastsq" sometimes results in the poor correlation evaluation, i.e.
        # covariance matrix is not computed for complicated multi objective models.
        method: str = "least_squares",
        **options,
    ):
        """Create new solver.

        Args:
            model: Curve fitting model.
            params: Initial parameters.
            method: Method. Default to scipy ``least_squares``-based method.
            **options: Extra options being passed to the LMFIT minimizer.
        """
        fitter_opts = {
            "nan_policy": "omit",
            "max_nfev": None,
        }
        fitter_opts.update(options)

        super().__init__(
            userfcn=model._residual,
            params=params,
            calc_covar=True,
            **fitter_opts,
        )
        self.model_repr = {series_def.name: series_def.fit_func for series_def in model.definitions}
        self.method = method

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sigma: np.ndarray,
        allocation: np.ndarray,
    ) -> SolverResult:
        """Perform fitting. Output is stored within the instance.

        Args:
            x: X values.
            y: Y values.
            sigma: Standard error of Y values.
            allocation: Mapping of each vector element to fitting function.

        Returns:
            A dataclass of fitting outcome.
        """
        if np.all(np.isfinite(sigma)):
            # Same for setting "absolute_sigma" in scipy curve_fit.
            # Covariance matrix is not scaled by reduced chi-squared.
            self.scale_covar = False
            weights = 1.0 / sigma
        else:
            self.scale_covar = True
            weights = None

        self.userargs = (y, weights)
        self.userkws = {"x": x, "allocation": allocation}

        try:
            minimizer_result = self.minimize(method=self.method)
        except ValueError as ex:
            # We assume the solver is scipy least_squares.
            # This raises ValueError when it fails due to wrong guesses or bounds.
            return SolverResult(
                method=self.method,
                model_repr=self.model_repr,
                success=False,
                message=str(ex),
                x_data=x,
                y_data=y,
            )

        if hasattr(minimizer_result, "covar") and any(np.diag(minimizer_result.covar) < 0):
            # Diagonal element of covariance matrix should be positive.
            # However, when residual is significant, i.e. bad quality,
            # sometimes it computes ill-covariance matrix with negative diagonals.
            delattr(minimizer_result, "covar")

        outcome = SolverResult(
            method=self.method,
            model_repr=self.model_repr,
            success=minimizer_result.success,
            nfev=minimizer_result.nfev,
            message=minimizer_result.message,
            dof=minimizer_result.nfree,
            chisq=minimizer_result.chisqr,
            reduced_chisq=minimizer_result.redchi,
            aic=minimizer_result.aic,
            bic=minimizer_result.bic,
            params={name: param.value for name, param in minimizer_result.params.items()},
            var_names=minimizer_result.var_names,
            x_data=x,
            y_data=y,
            covar=getattr(minimizer_result, "covar", None),
        )

        return outcome


def _compile_expr(expr: str) -> Tuple[Callable, Tuple[str, ...]]:
    # DO NOT LRU cache this function.
    # Seems like it's more efficient, but it also caches the local interpreter symtable
    # which may induce unexpected fit parameter confusion.
    # Specifically, the fit function is evaluated by looking-up the symtable,
    # thus if the table is shared among multiple fitter instance,
    # parameter values in the table might be updated from different instance.
    interpreter = Interpreter()
    astcode = interpreter.parse(expr.strip())
    parsed_params = get_ast_names(astcode)
    if "x" not in parsed_params:
        raise AnalysisError("Model must be a function of scan parameter 'x'.")

    params = []
    for param in parsed_params:
        if param == "x" or param in interpreter.symtable:
            # remove "x" or names from module, e.g. "exp" is not a parameter, it is a ufunc
            continue
        if param not in params:
            params.append(param)

    def expr_callable(**kwargs):
        interpreter.symtable.update(kwargs)
        interpreter.start_time = time.time()  # for time-out
        return interpreter.run(astcode)

    return expr_callable, tuple(params)
