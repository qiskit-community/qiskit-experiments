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
from typing import Callable, List, Optional, Union

import numpy as np

from qiskit_experiments.exceptions import AnalysisError


class FitModel(ABC):
    r"""Base class of fit models.

    This is a function-like object that implements a fit model as a ``__call__`` magic method,
    thus it behaves like a Python function that the SciPy curve_fit solver accepts.
    Note that the fit function there only accepts variadic arguments.

    This class ties together the fit function and associated parameter names to
    perform correct parameter mapping among multiple objective functions with different signatures,
    in which some parameters may be excluded from the fitting when their values are fixed.

    Examples:

        Given we have two functions :math:`F_1(x_1, p_0, p_1, p_2)` and :math:`F_2(x_2, p_0, p_3)`.
        During the fit, we assign :math:`p_1=2` and exclude it from the fitting.
        This is formulated with set operation as follows:

        .. math::

            \Theta_1 = \{ p_0, p_1, p_2 \}, \Theta_2 = \{p_0, p_3\}, \Theta_{\rm fix} = \{p_1\}

        Note that :class:`FitModel` subclass is instantiated with a list of
        :math:`F_1` and :math:`F_2` (``fit_functions``) together with
        a list of :math:`\Theta_1` and :math:`\Theta_2` (``signatures``) and
        :math:`\Theta_{\rm fix}` (set via :meth:`bind_parameters`).
        The signature of new fit model instance will be
        :math:`\Theta = (\Theta_1 \cup \Theta_2) - \Theta_{\rm fix} = \{ p_0, p_2, p_3\}`.
        The fit function that this model provides is accordingly

        .. math::

            F(x, \Theta) = F_1(x_0, \Theta_1) \oplus F_2(x_1, \Theta_2) \
                = F(x_0 \oplus x_1, p_0, p_2, p_3).

        This function might be called from the scipy curve fit algorithm
        which only takes variadic arguments (i.e. agnostic to parameter names).

        .. math::

            F(x, {\rm *args}) = F(x,\bar{p}_0, \bar{p}_1, \bar{p}_2)

        The fit model internally maps :math:`\bar{p}_0 \rightarrow p_0`,
        :math:`\bar{p}_1 \rightarrow p_2`, and :math:`\bar{p}_2 \rightarrow p_3`
        while assigning :math:`p_1=2` when its called from the curve fitting algorithm.
        Note that this mapping is performed in the ``__call__`` method.
        The function signature :math:`\Theta` is provided with the property :attr:`signature`.

    Notes:

        This class is usually instantiated with the :class:`SeriesDef` in the
        ``__init_subclass__`` method of :class:`CurveAnalysis` subclasses.
        User doesn't need to take care of how to initialize this class
        unless one manually create the instance for debugging purposes.
    """

    def __init__(
        self,
        fit_functions: List[Callable],
        signatures: List[List[str]],
        fit_models: Optional[Union[List[str], str]] = None,
    ):
        """Create new fit model.

        Args:
            fit_functions: List of callables that forms the fit model for a
                particular curve analysis class. It may consist of multiple curves
                which are defined in :attr:`CurveAnalysis.__series__`.
            signatures: List of argument names that each fit function callable takes.
                The length of the list should be identical to the ``fit_functions``.
            fit_models: String representation of fit functions.
                Because this is just a metadata, the format of input value doesn't matter.
                It may be a single string description for the entire fit model, or
                list of descriptions for each fit function. If not provided,
                "not defined" is stored in the experiment result metadata.

        Raises:
            AnalysisError: When ``fit_functions`` and ``signatures`` have a different length.
        """
        if len(fit_functions) != len(signatures):
            raise AnalysisError("Different numbers of fit_functions and signatures are given.")

        self._fit_functions = fit_functions
        self._signatures = signatures

        # String representation of the fit model. This is stored as a list of string.
        if not fit_models or isinstance(fit_models, str):
            fit_models = [fit_models]
        self._fit_models = fit_models

        # Create signature of this fit model, i.e. this will be signature of scipy fit function.
        # The curves comprising this model may have different signatures.
        # The signature of this fit model is union of parameters in all curves.
        # This is order preserving since this affects the index of ``popt`` that scipy fitter
        # returns, which appears as @Parameters entry of curve analysis as-is.
        union_params = []
        for signature in signatures:
            for parameter in signature:
                if parameter not in union_params:
                    union_params.append(parameter)
        self._union_params = union_params

        # This is set by users.
        self._fixed_params = {}

    @abstractmethod
    def __call__(self, x: np.ndarray, *params) -> np.ndarray:
        """Compute values of fit functions.

        Args:
            x: Input X values array.
            *params: Variadic argument provided from the fitter.

        Returns:
            Computed Y values array.
        """
        pass

    def bind_parameters(self, **kwparams):
        """Assign values to the fixed parameters.

        Args:
            kwparams: Dictionary of parameters that are excluded from the fitting.
                Every parameter, i.e. dictionary key, should be defined in the fit model.

        Raises:
            AnalysisError: When parameter name is not defined in the fit model.
        """
        if any(k not in self._union_params for k in kwparams):
            raise AnalysisError(
                f"Fixed parameters {', '.join(kwparams.keys())} are not all defined in the "
                f"fit model {', '.join(self._union_params)}."
            )
        self._fixed_params = kwparams

    @property
    def signature(self) -> List[str]:
        """Return signature of this fit model."""
        return [p for p in self._union_params if p not in self._fixed_params]

    @property
    def fit_model(self) -> str:
        """Return fit models."""
        if any(f is None for f in self._fit_models):
            return "not defined"
        return ",".join(self._fit_models)

    def copy(self):
        """Return copy of this function."""
        instance = self.__class__(
            fit_functions=self._fit_functions,
            signatures=self._signatures,
            fit_models=self._fit_models,
        )

        if self._fixed_params:
            instance.bind_parameters(**self._fixed_params.copy())

        return instance

    def __repr__(self):
        sigrepr = ", ".join(self.signature)
        if self._fixed_params:
            fixrepr = ", ".join(self._fixed_params.keys())
            return f"{self.__class__.__name__}(x, {sigrepr}; @ Fixed {fixrepr})"
        return f"{self.__class__.__name__}(x, {sigrepr})"


class SingleFitFunction(FitModel):
    r"""Fit model consisting of a single curve.

    This model is created when only a single curve exists in the fit model.

    .. math::

        F(x, \Theta) = f(x, \vec{p})

    The parameter :math:`\vec{p} = \Theta \cup \Theta_{\rm fix}` which is a union of
    the fit parameters and the fixed parameters :math:`\Theta_{\rm fix}`.
    The function :math:`f` is usually set by :attr:`SeriesDef.fit_func` which is
    a standard python function.

    .. seealso::

        Class :class:`FitModel`.
    """

    def __call__(self, x: np.ndarray, *params) -> np.ndarray:
        kwparams = dict(zip(self.signature, params))
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

    .. seealso::

        Class :class:`FitModel`.
    """

    def __init__(
        self,
        fit_functions: List[Callable],
        signatures: List[List[str]],
        fit_models: Optional[List[str]] = None,
    ):
        super().__init__(fit_functions, signatures, fit_models)

        # This attribute is set by users or another function that calls this model.
        # The existence of this value is not checked within the __call__ for performance,
        # but one must guarantee this is assigned before the model is called.
        self.data_allocation = None

    def __call__(self, x: np.ndarray, *params) -> np.ndarray:
        kwparams = dict(zip(self.signature, params))
        kwparams.update(self._fixed_params)

        y = np.zeros(x.size, dtype=float)
        for i, (func, sig) in enumerate(zip(self._fit_functions, self._signatures)):
            inds = self.data_allocation == i
            y[inds] = func(x[inds], **{p: kwparams[p] for p in sig})

        return y
