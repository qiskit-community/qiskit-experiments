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
Curve data classes.
"""
import dataclasses
import itertools
import inspect
from typing import Any, Dict, Union, List, Tuple, Optional, Iterable, Callable

import numpy as np
import uncertainties
from uncertainties.unumpy import uarray

from qiskit.utils.deprecation import deprecate_func

from qiskit_experiments.exceptions import AnalysisError


@dataclasses.dataclass(frozen=True)
class SeriesDef:
    """A dataclass to describe the definition of the curve.

    Attributes:
        fit_func: A callable that defines the fit model of this curve. The argument names
            in the callable are parsed to create the fit parameter list, which will appear
            in the analysis results. The first argument should be ``x`` that represents
            X-values that the experiment sweeps.
        filter_kwargs: Optional. Dictionary of properties that uniquely identifies this series.
            This dictionary is used for data processing.
            This must be provided when the curve analysis consists of multiple series.
        name: Optional. Name of this series.
        plot_color: Optional. String representation of the color that is used to draw fit data
            and data points in the output figure. This depends on the drawer class
            being set to the curve analysis options. Usually this conforms to the
            Matplotlib color names.
        plot_symbol: Optional. String representation of the marker shape that is used to draw
            data points in the output figure. This depends on the drawer class
            being set to the curve analysis options. Usually this conforms to the
            Matplotlib symbol names.
        canvas: Optional. Index of sub-axis in the output figure that draws this curve.
            This option is valid only when the drawer instance provides multi-axis drawing.
        model_description: Optional. Arbitrary string representation of this fit model.
            This string will appear in the analysis results as a part of metadata.
    """

    fit_func: Callable
    filter_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    name: str = "Series-0"
    plot_color: str = "black"
    plot_symbol: str = "o"
    canvas: Optional[int] = None
    model_description: Optional[str] = None
    signature: Tuple[str, ...] = dataclasses.field(init=False)

    @deprecate_func(
        since="0.5",
        additional_msg="SeriesDef has been replaced by the LMFIT module.",
        removal_timeline="after 0.6",
        package_name="qiskit-experiments",
    )
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __post_init__(self):
        """Parse the fit function signature to extract the names of the variables.
        Fit functions take arguments F(x, p0, p1, p2, ...) thus the first value should be excluded.
        """
        signature = list(inspect.signature(self.fit_func).parameters.keys())
        fitparams = tuple(signature[1:])

        # Note that this dataclass is frozen
        object.__setattr__(self, "signature", fitparams)


@dataclasses.dataclass(frozen=True)
class CurveData:
    """A dataclass that manages the multiple arrays comprising the dataset for fitting.

    This dataset can consist of X, Y values from multiple series.
    To extract curve data of the particular series, :meth:`get_subset_of` can be used.

    Attributes:
        x: X-values that experiment sweeps.
        y: Y-values that observed and processed by the data processor.
        y_err: Uncertainty of the Y-values which is created by the data processor.
            Usually this assumes standard error.
        shots: Number of shots used in the experiment to obtain the Y-values.
        data_allocation: List with identical size with other arrays.
            The value indicates the series index of the corresponding element.
            This is classified based upon the matching of :attr:`SeriesDef.filter_kwargs`
            with the circuit metadata of the corresponding data index.
            If metadata doesn't match with any series definition, element is filled with ``-1``.
        labels: List of curve labels. The list index corresponds to the series index.
    """

    x: np.ndarray
    y: np.ndarray
    y_err: np.ndarray
    shots: np.ndarray
    data_allocation: np.ndarray
    labels: List[str]

    @deprecate_func(
        since="0.6",
        additional_msg="CurveData is replaced by `ScatterTable`'s DataFrame representation.",
        removal_timeline="after 0.7",
        package_name="qiskit-experiments",
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_subset_of(self, index: Union[str, int]) -> "CurveData":
        """Filter data by series name or index.

        Args:
            index: Series index of name.

        Returns:
            A subset of data corresponding to a particular series.
        """
        if isinstance(index, int):
            _index = index
            _name = self.labels[index]
        else:
            _index = self.labels.index(index)
            _name = index

        locs = self.data_allocation == _index
        return CurveData(
            x=self.x[locs],
            y=self.y[locs],
            y_err=self.y_err[locs],
            shots=self.shots[locs],
            data_allocation=np.full(np.count_nonzero(locs), _index),
            labels=[_name],
        )


class CurveFitResult:
    """Result of Qiskit Experiment curve analysis."""

    def __init__(
        self,
        method: Optional[str] = None,
        model_repr: Optional[Dict[str, str]] = None,
        success: Optional[bool] = True,
        nfev: Optional[int] = None,
        message: Optional[str] = "",
        dof: Optional[float] = None,
        init_params: Optional[Dict[str, float]] = None,
        chisq: Optional[float] = None,
        reduced_chisq: Optional[float] = None,
        aic: Optional[float] = None,
        bic: Optional[float] = None,
        params: Optional[Dict[str, float]] = None,
        var_names: Optional[List[str]] = None,
        x_data: Optional[np.ndarray] = None,
        y_data: Optional[np.ndarray] = None,
        weighted_residuals: Optional[np.ndarray] = None,
        residuals: Optional[np.ndarray] = None,
        covar: Optional[np.ndarray] = None,
    ):
        """Create new Qiskit curve analysis result object.

        Args:
            method: A name of fitting algorithm used for the curve fitting.
            model_repr: String representation of fit functions of each curve.
            success: True when the fitting is successfully performed.
            nfev: Number of fit function evaluation until the solution is obtained.
            message: Any message from the fitting software.
            dof: Degree of freedom in this fitting, i.e. number of free parameters.
            init_params: Initial parameters provided to the fitter.
            chisq: Chi-squared value.
            reduced_chisq: Reduced Chi-squared value.
            aic: Akaike's information criterion.
            bic: Bayesian information criterion.
            params: Estimated fitting parameters keyed on the parameter names in the fit function.
            var_names: Name of variables, i.e. fixed parameters are excluded from the list.
            x_data: X values used for the fitting.
            y_data: Y values used for the fitting.
            weighted_residuals: The residuals from the fitting after assigning weights for each ydata.
            residuals: residuals of the fitted model.
            covar: Covariance matrix of fitting variables.
        """
        self.method = method
        self.model_repr = model_repr
        self.success = success
        self.nfev = nfev
        self.message = message
        self.dof = dof
        self.init_params = init_params
        self.chisq = chisq
        self.reduced_chisq = reduced_chisq
        self.aic = aic
        self.bic = bic
        self.params = params
        self.var_names = var_names
        self.x_data = x_data
        self.y_data = y_data
        self.weighted_residuals = weighted_residuals
        self.residuals = residuals
        self.covar = covar

    @property
    def x_range(self) -> Tuple[float, float]:
        """Range of x_data values."""
        return min(self.x_data), max(self.x_data)

    @property
    def y_range(self) -> Tuple[float, float]:
        """Range of y_data values."""
        return min(self.y_data), max(self.y_data)

    @property
    def ufloat_params(self) -> Dict[str, uncertainties.UFloat]:
        """UFloat representation of fit parameters."""
        if hasattr(self, "_ufloat_params"):
            # Return cache
            return getattr(self, "_ufloat_params")

        if self.params is None:
            ufloat_params = None
        else:
            if self.covar is not None:
                ufloat_fitvals = uncertainties.correlated_values(
                    nom_values=[self.params[name] for name in self.var_names],
                    covariance_mat=self.covar,
                    tags=self.var_names,
                )
            else:
                # Invalid covariance matrix. Std dev is set to nan, i.e. not computed.
                with np.errstate(invalid="ignore"):
                    # Setting std_devs to NaN will trigger floating point exceptions
                    # which we can ignore. See https://stackoverflow.com/q/75656026
                    ufloat_fitvals = uarray(
                        nominal_values=[self.params[name] for name in self.var_names],
                        std_devs=np.full(len(self.var_names), np.nan),
                    )
            # Combine fixed params and fitting variables into a single dictionary
            # Fixed parameter has zero std_dev
            ufloat_params = {}
            for name in self.params.keys():
                try:
                    uind = self.var_names.index(name)
                    ufloat_params[name] = ufloat_fitvals[uind]
                except ValueError:
                    ufloat_params[name] = uncertainties.ufloat(self.params[name], std_dev=0.0)

        setattr(self, "_ufloat_params", ufloat_params)
        return ufloat_params

    @property
    def correl(self):
        """Correlation matrix of fit parameters."""
        if hasattr(self, "_correl"):
            # Return cache
            return getattr(self, "_correl")

        if self.covar is not None:
            # This is how uncertainties computes correlation matrix
            stdevs = np.sqrt(np.diag(self.covar))
            correl = self.covar / stdevs / stdevs[:, np.newaxis]
        else:
            correl = None

        setattr(self, "_correl", correl)
        return correl

    def __str__(self):
        ret = "CurveFitResult:"
        ret += f"\n - fitting method: {self.method}"
        ret += f"\n - number of sub-models: {len(self.model_repr)}"
        for model_name, model_expr in self.model_repr.items():
            if len(model_expr) > 60:
                model_expr = f"{model_expr[:60]}..."
            ret += f"\n  * F_{model_name}(x) = {model_expr}"
        ret += f"\n - success: {self.success}"
        ret += f"\n - number of function evals: {self.nfev}"
        ret += f"\n - degree of freedom: {self.dof}"
        ret += f"\n - chi-square: {self.chisq}"
        ret += f"\n - reduced chi-square: {self.reduced_chisq}"
        ret += f"\n - Akaike info crit.: {self.aic}"
        ret += f"\n - Bayesian info crit.: {self.bic}"
        if self.init_params is not None:
            ret += "\n - init params:"
            for name, value in self.init_params.items():
                ret += f"\n  * {name} = {value}"
        if self.ufloat_params is not None:
            ret += "\n - fit params:"
            for name, param in self.ufloat_params.items():
                if np.isfinite(param.std_dev):
                    ret += f"\n  * {name} = {param.nominal_value} ± {param.std_dev}"
                else:
                    ret += f"\n  * {name} = {param.nominal_value}"
        if self.correl is not None:
            ret += "\n - correlations:"
            correlated = {}
            for pi, pj in itertools.combinations(range(len(self.var_names)), 2):
                correlated[(pi, pj)] = self.correl[pi, pj]
            for (pi, pj), corr in sorted(correlated.items(), key=lambda item: item[1]):
                ret += f"\n  * ({self.var_names[pi]}, {self.var_names[pj]}) = {corr}"

        return ret

    def __copy__(self):
        instance = CurveFitResult(**self.__json_encode__())
        # Copying ufloat invalidate parameter correlation.
        # Note that ufloat object has `self._linear_part.linear_combo` dictionary
        # to store parameter correlation keyed on the ufloat objects.
        # Copying the ufloat object may change object id, which is the identifier
        # of ufloat value, thus it invalidates the `linear_combo` dictionary.
        # To avoid missing correlation, the copy invalidate ufloat parameter object cache.
        return instance

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __json_encode__(self):
        return {
            "method": self.method,
            "model_repr": self.model_repr,
            "success": self.success,
            "nfev": self.nfev,
            "message": self.message,
            "dof": self.dof,
            "init_params": self.init_params,
            "chisq": self.chisq,
            "reduced_chisq": self.reduced_chisq,
            "aic": self.aic,
            "bic": self.bic,
            "params": self.params,
            "var_names": self.var_names,
            "x_data": self.x_data,
            "y_data": self.y_data,
            "covar": self.covar,
        }

    @classmethod
    def __json_decode__(cls, value):
        return cls(**value)


@dataclasses.dataclass(frozen=True)
class FitData:
    """A dataclass to store the outcome of the fitting.

    Attributes:
        popt: List of optimal parameter values with uncertainties if available.
        popt_keys: List of parameter names being fit.
        pcov: Covariance matrix from the least square fitting.
        reduced_chisq: Reduced Chi-squared value for the fit curve.
        dof: Degree of freedom in this fit model.
        x_data: X-values provided to the fitter.
        y_data: Y-values provided to the fitter.
    """

    popt: List[uncertainties.UFloat]
    popt_keys: List[str]
    pcov: np.ndarray
    reduced_chisq: float
    dof: int
    x_data: np.ndarray
    y_data: np.ndarray

    @deprecate_func(
        since="0.5",
        additional_msg="Fit data is replaced with 'CurveFitResult' based on LMFIT minimizer result.",
        removal_timeline="after 0.6",
        package_name="qiskit-experiments",
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def x_range(self) -> Tuple[float, float]:
        """Range of x values."""
        return np.min(self.x_data), np.max(self.x_data)

    @property
    def y_range(self) -> Tuple[float, float]:
        """Range of y values."""
        return np.min(self.y_data), np.max(self.y_data)

    def fitval(self, key: str) -> uncertainties.UFloat:
        """A helper method to get fit value object from parameter key name.

        Args:
            key: Name of parameters to extract.

        Returns:
            A UFloat object which functions as a standard Python float object
            but with automatic error propagation.

        Raises:
            ValueError: When specified parameter is not defined.
        """
        try:
            index = self.popt_keys.index(key)
            return self.popt[index]
        except ValueError as ex:
            raise ValueError(f"Parameter {key} is not defined.") from ex


@dataclasses.dataclass
class ParameterRepr:
    """Detailed description of fitting parameter.

    Attributes:
        name: Original name of the fit parameter being defined in the fit model.
        repr: Optional. Human-readable parameter name shown in the analysis result and in the figure.
        unit: Optional. Physical unit of this parameter if applicable.
    """

    # Fitter argument name
    name: str

    # Unicode representation
    repr: Optional[str] = None

    # Unit
    unit: Optional[str] = None


class OptionsDict(dict):
    """General extended dictionary for fit options.

    This dictionary provides several extra features.

    - A value setting method which validates the dict key and value.
    - Dictionary keys are limited to those specified in the constructor as ``parameters``.
    """

    def __init__(
        self,
        parameters: List[str],
        defaults: Optional[Union[Iterable[Any], Dict[str, Any]]] = None,
    ):
        """Create new dictionary.

        Args:
            parameters: List of parameter names used in the fit model.
            defaults: Default values.

        Raises:
            AnalysisError: When defaults is provided as array-like but the number of
                element doesn't match with the number of fit parameters.
        """
        if defaults is not None:
            if not isinstance(defaults, dict):
                if len(defaults) != len(parameters):
                    raise AnalysisError(
                        f"Default parameter {defaults} is provided with array-like "
                        "but the number of element doesn't match. "
                        f"This fit requires {len(parameters)} parameters."
                    )
                defaults = dict(zip(parameters, defaults))

            full_options = {p: self.format(defaults.get(p, None)) for p in parameters}
        else:
            full_options = {p: None for p in parameters}

        super().__init__(**full_options)

    def __setitem__(self, key, value):
        """Set value with validations.

        Raises:
            AnalysisError: When key is not previously defined.
        """
        if key not in self:
            raise AnalysisError(f"Parameter {key} is not defined in this fit model.")
        super().__setitem__(key, self.format(value))

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    def set_if_empty(self, **kwargs):
        """Set value to the dictionary if not assigned.

        Args:
              kwargs: Key and new value to assign.
        """
        for key, value in kwargs.items():
            if self.get(key) is None:
                self[key] = value

    @staticmethod
    def format(value: Any) -> Any:
        """Format dictionary value.

        Subclasses may override this method to provide their own validation.

        Args:
            value: New value to assign.

        Returns:
            Formatted value.
        """
        return value


class InitialGuesses(OptionsDict):
    """Dictionary providing a float validation for initial guesses."""

    @staticmethod
    def format(value: Any) -> Optional[float]:
        """Validate that value is float a float or None.

        Args:
            value: New value to assign.

        Returns:
            Formatted value.

        Raises:
            AnalysisError: When value is not a float or None.
        """
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError) as ex:
            raise AnalysisError(f"Input value {value} is not valid initial guess. ") from ex


class Boundaries(OptionsDict):
    """Dictionary providing a validation for boundaries."""

    @staticmethod
    def format(value: Any) -> Optional[Tuple[float, float]]:
        """Validate if value is a min-max value tuple.

        Args:
            value: New value to assign.

        Returns:
            Formatted value.

        Raises:
            AnalysisError: When value is invalid format.
        """
        if value is None:
            return None

        try:
            minv, maxv = value
            if minv >= maxv:
                raise AnalysisError(
                    f"The first value is greater than the second value {minv} >= {maxv}."
                )
            return float(minv), float(maxv)
        except (TypeError, ValueError) as ex:
            raise AnalysisError(f"Input boundary {value} is not a min-max value tuple.") from ex


# pylint: disable=invalid-name
class FitOptions:
    """Collection of fitting options.

    This class is initialized with a list of parameter names used in the fit model
    and corresponding default values provided by users.

    This class is hashable, and generates fitter keyword arguments.
    """

    def __init__(
        self,
        parameters: List[str],
        default_p0: Optional[Union[Iterable[float], Dict[str, float]]] = None,
        default_bounds: Optional[Union[Iterable[Tuple], Dict[str, Tuple]]] = None,
        **extra,
    ):
        # These are private members so that user cannot directly override values
        # without implicitly implemented validation logic. No setter will be provided.
        self.__p0 = InitialGuesses(parameters, default_p0)
        self.__bounds = Boundaries(parameters, default_bounds)
        self.__extra = extra

    def __hash__(self):
        return hash((self.__p0, self.__bounds, tuple(sorted(self.__extra.items()))))

    def __eq__(self, other):
        if isinstance(other, FitOptions):
            checks = [
                self.__p0 == other.__p0,
                self.__bounds == other.__bounds,
                self.__extra == other.__extra,
            ]
            return all(checks)
        return False

    def add_extra_options(self, **kwargs):
        """Add more fitter options."""
        self.__extra.update(kwargs)

    def copy(self):
        """Create copy of this option."""
        return FitOptions(
            parameters=list(self.__p0.keys()),
            default_p0=dict(self.__p0),
            default_bounds=dict(self.__bounds),
            **self.__extra,
        )

    @property
    def p0(self) -> InitialGuesses:
        """Return initial guess dictionary."""
        return self.__p0

    @property
    def bounds(self) -> Boundaries:
        """Return bounds dictionary."""
        return self.__bounds

    @property
    def fitter_opts(self) -> Boundaries:
        """Return fitter options dictionary."""
        return self.__extra

    @property
    def options(self):
        """Generate keyword arguments of the curve fitter."""
        bounds = {k: v if v is not None else (-np.inf, np.inf) for k, v in self.__bounds.items()}
        return {"p0": dict(self.__p0), "bounds": bounds, **self.__extra}
