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
import inspect
from typing import Any, Dict, Callable, Union, List, Tuple, Optional, Iterable

import numpy as np
import uncertainties
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
                self.__setitem__(key, value)

    @staticmethod
    def format(value: Any) -> Any:
        """Format dictionary value.

        Subcasses may override this method to provide their own validation.

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
    def options(self):
        """Generate keyword arguments of the curve fitter."""
        bounds = {k: v if v is not None else (-np.inf, np.inf) for k, v in self.__bounds.items()}
        return {"p0": dict(self.__p0), "bounds": bounds, **self.__extra}
