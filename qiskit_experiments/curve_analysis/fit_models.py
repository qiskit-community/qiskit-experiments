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
"""Fit models that are used for curve fitting."""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import numpy as np

from qiskit_experiments.exceptions import AnalysisError


class FitModel(ABC):
    """Base class of fit models.

    This is a function-like object that implements a fit model as a ``__call__`` magic method,
    thus it behaves like a Python function that the SciPy curve_fit solver accepts.
    Note that the fit function there only accepts variadic arguments.

    This class ties together the fit function and associated parameter names to
    perform correct parameter mapping among multiple objective functions with different signatures,
    in which some parameters may be excluded from the fitting when they are fixed.
    """

    def __init__(
        self,
        fit_functions: List[Callable],
        signatures: List[List[str]],
        fit_models: Optional[List[str]] = None,
        fixed_parameters: Optional[List[str]] = None,
    ):
        """Create new fit model.

        Args:
            fit_functions: List of callables that defines fit function of a single series.
            signatures: List of parameter names of a single series.
            fit_models: List of string representation of fit functions.
            fixed_parameters: List of parameter names that are fixed in the fit.

        Raises:
            AnalysisError: When ``fit_functions`` and ``signatures`` don't match.
        """
        if len(fit_functions) != len(signatures):
            raise AnalysisError("Different numbers of fit_functions and signatures are given.")

        self._fit_functions = fit_functions
        self._signatures = signatures
        self._fit_models = fit_models or [None for _ in range(len(fit_functions))]

        if not fixed_parameters:
            fixed_parameters = []
        self._fixed_params = {p: None for p in fixed_parameters}

        # Create signature of this fit model, i.e. this will be signature of scipy fit function.
        # The curves comprising this model may have different signature.
        # The signature of this fit model is union of parameters in all curves.
        union_params = []
        for signature in signatures:
            for parameter in signature:
                if parameter not in union_params and parameter not in fixed_parameters:
                    union_params.append(parameter)
        self._uniton_params = union_params

    @abstractmethod
    def __call__(self, x: np.ndarray, *params) -> np.ndarray:
        pass

    def bind_parameters(self, **kwparams):
        """Assign values to the fixed parameters."""
        bind_dict = {k: kwparams[k] for k in self._fixed_params.keys() if k in kwparams}
        self._fixed_params.update(bind_dict)

    @property
    def signature(self) -> List[str]:
        """Return signature of this fit model."""
        return self._uniton_params

    @property
    def fit_model(self) -> str:
        """Return fit models."""
        if any(f is None for f in self._fit_models):
            return "not defined"
        return ",".join(self._fit_models)

    def copy(self):
        """Return copy of this function."""
        return self.__class__(
            fit_functions=self._fit_functions,
            signatures=self._signatures,
            fit_models=self._fit_models,
            fixed_parameters=list(self._fixed_params.keys()),
        )

    def __repr__(self):
        sigrepr = ", ".join(self.signature)
        if self._fixed_params:
            fixrepr = ", ".join(self._fixed_params.keys())
            return f"{self.__class__.__name__}(x, {sigrepr}; @ Fixed {fixrepr})"
        return f"{self.__class__.__name__}(x, {sigrepr})"


class SingleFitFunction(FitModel):
    r"""Fit model consisting of a single curve.

    This model is created when only single curve exist in the fit model.

    .. math::

        F(x, \Theta) = f(x, \vec{p})

    The parameter :math:`\vec{p} = \Theta \cup \Theta_{\rm fix}` which is a union of
    the fit parameters and the fixed parameters :math:`\Theta_{\rm fix}`.
    The function :math:`f` is usually set by :attr:`SeriesDef.fit_func` which is
    a standard python function.
    """

    def __call__(self, x: np.ndarray, *params) -> np.ndarray:
        """Compute values of fit functions.

        Args:
            x: Composite X values array.
            *params: Variadic argument provided from the fitter.

        Returns:
            Computed Y values array.
        """
        kwparams = dict(zip(self._uniton_params, params))
        kwparams.update(self._fixed_params)

        return self._fit_functions[0](x, **{p: kwparams[p] for p in self._signatures[0]})


class CompositeFitFunction(FitModel):
    r"""Fit model consisting of multiple curves sharing fit parameters.

    This model is created when multiple curves form a fit model.

    .. math::

        F(x, \Theta) = f_0(x_0, \vec{p}_0) \oplus f_1(x_1, \vec{p}_1) \oplus ...

    The parameter :math:`\vec{p}_i = \theta_i \cup \Theta_{\rm fix}` is a union of the
    fit parameter for the function :math:`f_i` and the fixed parameters :math:`\Theta_{\rm fix}`.
    The composite function :math:`F` consists of multiple fit functions :math:`f_i`
    taking independent data points :math:`x_i` with partly shared fit parameters :math:`\theta_i`,
    where :math:`\Theta = \theta_0 \cup \theta_1 \cup ...` and the composite data vector
    :math:`x = x_0 \oplus x_1 \oplus ...`

    In the NumPy array data, this can be represented
    by a single array together with the array specifying location, which is provided as a
    :attr:`CompositeFitFunction.data_allocation`. For example:

    .. parsed-literal::

        data_allocation = array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, ...])

    This data represents the location where the function with index ``i``
    is returned and where the x values :math:`x_i` comes from.
    One must set this data indices before calling the composite fit function.
    """

    def __init__(
        self,
        fit_functions: List[Callable],
        signatures: List[List[str]],
        fit_models: Optional[List[str]] = None,
        fixed_parameters: Optional[List[str]] = None,
    ):
        super().__init__(fit_functions, signatures, fit_models, fixed_parameters)
        self.data_allocation = None

    def __call__(self, x: np.ndarray, *params) -> np.ndarray:
        """Compute values of fit functions.

        Args:
            x: Composite X values array.
            *params: Variadic argument provided from the fitter.

        Returns:
            Computed Y values array.
        """
        kwparams = dict(zip(self._uniton_params, params))
        kwparams.update(self._fixed_params)

        y = np.zeros(x.size, dtype=float)
        for i, (func, sig) in enumerate(zip(self._fit_functions, self._signatures)):
            inds = self.data_allocation == i
            y[inds] = func(x[inds], **{p: kwparams[p] for p in sig})

        return y
