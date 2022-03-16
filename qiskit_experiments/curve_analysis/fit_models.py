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
from qiskit_experiments.curve_analysis.curve_data import SeriesDef


class FitModel(ABC):
    r"""Base class of fit models.

    This is a function-like object that implements a fit model as a ``__call__`` magic method,
    thus it behaves like a Python function that the SciPy curve_fit solver accepts.
    Note that the fit function there only accepts variadic arguments.

    This class ties together the fit function and associated parameter names to
    perform correct parameter mapping among multiple objective functions with different signatures,
    in which some parameters may be excluded from the fitting when their values are fixed.

    Examples:

        We assume a model with two functions :math:`F_1(x_1, p_0, p_1, p_2)` and
        :math:`F_2(x_2, p_0, p_3)`.
        During the fit, we assign :math:`p_1=2` and exclude it from the fitting.
        The parameters of this model are described by the sets

        .. math::

            \Theta_1 = \{ p_0, p_1, p_2 \}, \Theta_2 = \{p_0, p_3\}, \Theta_{\rm fix} = \{p_1\}

        The corresponding :class:`FitModel` subclass is instantiated with a list ``fit_functions``
        containing the :math:`F_1` and :math:`F_2` functions together with
        a list ``signatures`` containing :math:`\Theta_1` and :math:`\Theta_2`. The parameters
        with fixed values :math:`\Theta_{\rm fix}` are removed from the signature using the
        :meth:`bind_parameters` method. The signature of new fit model instance will be

        .. math::

            \Theta = (\Theta_1 \cup \Theta_2) - \Theta_{\rm fix} = \{ p_0, p_2, p_3\}.

        The fit function that this model provides is therefore

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
        Users do not need to know how to initialize this class
        unless they manually create the instance for debugging purposes.
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

        # Create the signature of the fit model, i.e. the signature of ``__call__`` for scipy.
        # The individual curves comprising this model may have different signatures.
        # The signature of this fit model is the union of the parameters in all curves.
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

    @classmethod
    def from_definitions(
        cls, series_defs: List[SeriesDef]
    ) -> Union["SingleFitFunction", "CompositeFitFunction"]:
        """Create fit model from series definitions.

        Args:
            series_defs: Series definitions that define a set of fit functions.

        Returns:
            Fit model.
        """
        fit_functions = []
        signatures = []
        fit_models = []
        for series_def in series_defs:
            fit_functions.append(series_def.fit_func)
            signatures.append(series_def.signature)
            fit_models.append(series_def.model_description)

        if len(series_defs) == 1:
            return SingleFitFunction(fit_functions, signatures, fit_models)
        return CompositeFitFunction(fit_functions, signatures, fit_models)

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

        F(X, \Theta) = f_0(\vec{x}_0, \vec{p}_0) \oplus f_1(\vec{x}_1, \vec{p}_1) \oplus ...,

    here the function :math:`f_i(\vec{x}_i, \vec{p}_i)` is applied to the data with the sequence
    of x-values :math:`\vec{x}_i \in \Re^{N_i}`, which are provided by the :math:`i`-th subset
    of experiments, :math:`E_i(\vec{x}_i) = \{E_i(x_0), E_i(x_1), ... E_i(x_{N_i-1})\}`,
    together with the measured outcomes :math:`\vec{y}_i \in \Re^{N_i}` that are fit by this model.
    The size of vector :math:`N_i` may depend on the configuration of experiment :math:`i`.

    The parameter :math:`\vec{p}_i = \theta_i \cup \Theta_{\rm fix}` is a union of the
    fit parameter for the function :math:`f_i` and the fixed parameters :math:`\Theta_{\rm fix}`.
    The composite function :math:`F` consists of multiple fit functions :math:`f_i`
    taking independent data points :math:`\vec{x}_i` with
    partly shared fit parameters :math:`\theta_i`,
    where :math:`\Theta = \theta_0 \cup \theta_1 \cup ...`.
    The composite scan values is :math:`X =\vec{x}_0 \oplus \vec{x}_1 \oplus ...` and
    the corresponding outcome is :math:`Y =\vec{y}_0 \oplus \vec{y}_1 \oplus ...`.

    In the Qiskit Experiments, these data sources are represented by
    a single 1D array ``vec[k]``, rather than a 2D array ``mat[i, j]``.
    To keep the mapping of the datum at index :math:`k` to the original series :math:`i`,
    an extra index vector ``data_allocation`` :math:`I` must be
    provided with :math:`X` and :math:`Y`.

    For example, we assume following data are obtained with two experiments :math:`E_0, E_1`.

    .. code-block:: python3

        # From E0
        x_0 = array([1, 2, 3])
        y_0 = array([4, 5, 6])

        # From E1
        x_1 = array([4, 5])
        y_1 = array([7, 8])

    The composite data :math:`(X, Y, I)` might take the form:

    .. code-block:: python3

        X = array([1, 4, 2, 5, 3])
        Y = array([4, 7, 5, 8, 6])
        I = array([0, 1, 0, 1, 0])

    With this data representation, we can reconstruct each subset as

    .. code-block:: python3

        # To E0 subset
        assert all(X[I == 0] == x_0)
        assert all(Y[I == 0] == y_0)

        # To E1 subset
        assert all(X[I == 1] == x_1)
        assert all(Y[I == 1] == y_1)

    The caller of this model must set this data indices before calling the function.
    With this data representation, we can reuse the fitting algorithm for the
    single objective function, where only 1-D arrays are accepted,
    for the multi-objective optimization model consisting of multiple data set.

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
