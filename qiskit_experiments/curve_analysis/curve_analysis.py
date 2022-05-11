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
Analysis class for curve fitting.
"""
# pylint: disable=invalid-name

import dataclasses
import functools
import warnings
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
from uncertainties import unumpy as unp, UFloat
from qiskit.utils import detach_prefix

from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import ExperimentData, AnalysisResultData, AnalysisConfig
from qiskit_experiments.warnings import deprecated_function

from .base_curve_analysis import BaseCurveAnalysis
from .curve_data import CurveData, SeriesDef


class CurveAnalysis(BaseCurveAnalysis):
    """Base class for curve analysis with single curve group.

    The fit parameters from the series defined under the analysis class are all shared
    and the analysis performs a single multi-objective function optimization.

    See :class:`BaseCurveAnalysis` for overridable method documentation.
    """

    #: List[SeriesDef]: List of mapping representing a data series
    __series__ = list()

    def __init__(self):
        """Initialize data fields that are privately accessed by methods."""
        super().__init__()

        if hasattr(self, "__fixed_parameters__"):
            warnings.warn(
                "The class attribute __fixed_parameters__ has been deprecated and will be removed. "
                "Now this attribute is absorbed in analysis options as fixed_parameters. "
                "This warning will be dropped in v0.4 along with "
                "the support for the deprecated attribute.",
                DeprecationWarning,
                stacklevel=2,
            )
            # pylint: disable=no-member
            self._options.fixed_parameters = {
                p: self.options.get(p, None) for p in self.__fixed_parameters__
            }

        #: List[CurveData]: Processed experiment data set. For backward compatibility.
        self.__processed_data_set = {}

    @classmethod
    def _fit_params(cls) -> List[str]:
        """Return a list of fitting parameters.

        Returns:
            A list of fit parameter names.

        Raises:
            AnalysisError: When series definitions have inconsistent multi-objective fit function.
            ValueError: When fixed parameter name is not used in the fit function.
        """
        fsigs = set()
        for series_def in cls.__series__:
            fsigs.add(series_def.signature)
        if len(fsigs) > 1:
            raise AnalysisError(
                "Fit functions specified in the series definition have "
                "different function signature. They should receive "
                "the same parameter set for multi-objective function fit."
            )
        return list(next(iter(fsigs)))

    @property
    def parameters(self) -> List[str]:
        """Return parameters of this curve analysis."""
        return [s for s in self._fit_params() if s not in self.options.fixed_parameters]

    # pylint: disable=bad-docstring-quotes
    @deprecated_function(
        last_version="0.4",
        msg=(
            "CurveAnalysis will also drop internal cache of processed data after 0.4. "
            "Relevant method signature has been updated to directly receive curve data "
            "rather than accessing data with this method."
        ),
    )
    def _data(
        self,
        series_name: Optional[str] = None,
        label: Optional[str] = "fit_ready",
    ) -> CurveData:
        """Deprecated. Getter for experiment data set.

        Args:
            series_name: Series name to search for.
            label: Label attached to data set. By default, it returns "fit_ready" data.

        Returns:
            Filtered curve data set.

        Raises:
            AnalysisError: When requested series or label are not defined.
        """
        try:
            data = self.__processed_data_set[label]
        except KeyError as ex:
            raise AnalysisError(f"Requested data with label {label} does not exist.") from ex

        if series_name is None:
            return data
        return data.get_subset_of(series_name)

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List["pyplot.Figure"]]:

        # Update all fit functions in the series definitions if fixed parameter is defined.
        # These lines will be removed once proper fit model class is implemented.
        assigned_params = self.options.fixed_parameters
        if assigned_params:
            # Check if all parameters are assigned.
            if any(v is None for v in assigned_params.values()):
                raise AnalysisError(
                    f"Unassigned fixed-value parameters for the fit "
                    f"function {self.__class__.__name__}."
                    f"All values of fixed-parameters, i.e. {assigned_params}, "
                    "must be provided by the analysis options to run this analysis."
                )
            # Override series definition with assigned fit functions.
            assigned_series = []
            for series_def in self.__series__:
                dict_def = dataclasses.asdict(series_def)
                dict_def["fit_func"] = functools.partial(series_def.fit_func, **assigned_params)
                del dict_def["signature"]
                assigned_series.append(SeriesDef(**dict_def))
            self.__series__ = assigned_series

        # Prepare for fitting
        self._initialize(experiment_data)
        analysis_results = []

        # Run data processing
        processed_data = self._run_data_processing(experiment_data.data(), self.__series__)

        if self.options.plot and self.options.plot_raw_data:
            for s in self.__series__:
                sub_data = processed_data.get_subset_of(s.name)
                self.drawer.draw_raw_data(
                    x_data=sub_data.x,
                    y_data=sub_data.y,
                    ax_index=s.canvas,
                )
        # for backward compatibility, will be removed in 0.4.
        self.__processed_data_set["raw_data"] = processed_data

        # Format data
        formatted_data = self._format_data(processed_data)
        if self.options.plot:
            for s in self.__series__:
                sub_data = formatted_data.get_subset_of(s.name)
                self.drawer.draw_formatted_data(
                    x_data=sub_data.x,
                    y_data=sub_data.y,
                    y_err_data=sub_data.y_err,
                    name=s.name,
                    ax_index=s.canvas,
                    color=s.plot_color,
                    marker=s.plot_symbol,
                )
        # for backward compatibility, will be removed in 0.4.
        self.__processed_data_set["fit_ready"] = formatted_data

        # Run fitting
        fit_data = self._run_curve_fit(formatted_data, self.__series__)

        # Create figure and result data
        if fit_data:
            metadata = self.options.extra.copy()
            metadata["fit_models"] = {
                s.name: s.model_description or "no description" for s in self.__series__
            }
            quality = self._evaluate_quality(fit_data)

            # Create analysis results
            analysis_results.extend(self._create_analysis_results(fit_data, quality, **metadata))
            # calling old extra entry method for backward compatibility
            if hasattr(self, "_extra_database_entry"):
                warnings.warn(
                    "Method '_extra_database_entry' has been deprecated and will be "
                    "removed after 0.4. Please override new method "
                    "'_create_analysis_results' with updated method signature.",
                    DeprecationWarning,
                )
                deprecated_method = getattr(self, "_extra_database_entry")
                analysis_results.extend(deprecated_method(fit_data))

            # Draw fit curves and report
            if self.options.plot:
                for s in self.__series__:
                    interp_x = np.linspace(*fit_data.x_range, 100)

                    params = {}
                    for fitpar in s.signature:
                        if fitpar in self.options.fixed_parameters:
                            params[fitpar] = self.options.fixed_parameters[fitpar]
                        else:
                            params[fitpar] = fit_data.fitval(fitpar)

                    y_data_with_uncertainty = s.fit_func(interp_x, **params)
                    y_mean = unp.nominal_values(y_data_with_uncertainty)
                    y_std = unp.std_devs(y_data_with_uncertainty)
                    # Draw fit line
                    self.drawer.draw_fit_line(
                        x_data=interp_x,
                        y_data=y_mean,
                        ax_index=s.canvas,
                        color=s.plot_color,
                    )
                    # Draw confidence intervals with different n_sigma
                    sigmas = unp.std_devs(y_data_with_uncertainty)
                    if np.isfinite(sigmas).all():
                        for n_sigma, alpha in self.drawer.options.plot_sigma:
                            self.drawer.draw_confidence_interval(
                                x_data=interp_x,
                                y_ub=y_mean + n_sigma * y_std,
                                y_lb=y_mean - n_sigma * y_std,
                                ax_index=s.canvas,
                                alpha=alpha,
                                color=s.plot_color,
                            )

                # Write fitting report
                report_description = ""
                for res in analysis_results:
                    if isinstance(res.value, (float, UFloat)):
                        report_description += f"{analysis_result_to_repr(res)}\n"
                report_description += r"Fit $\chi^2$ = " + f"{fit_data.reduced_chisq: .4g}"
                self.drawer.draw_fit_report(description=report_description)

        # Add raw data points
        analysis_results.extend(self._create_curve_data(formatted_data, self.__series__))

        # Finalize plot
        if self.options.plot:
            self.drawer.format_canvas()
            return analysis_results, [self.drawer.figure]

        return analysis_results, []

    @classmethod
    def from_config(cls, config: Union[AnalysisConfig, Dict]) -> "CurveAnalysis":
        # For backward compatibility. This will be removed in v0.4.

        instance = super().from_config(config)

        # When fixed param value is hard-coded as options. This is deprecated data structure.
        loaded_opts = instance.options.__dict__

        # pylint: disable=no-member
        deprecated_fixed_params = {
            p: loaded_opts[p] for p in instance.parameters if p in loaded_opts
        }
        if any(deprecated_fixed_params):
            warnings.warn(
                "Fixed parameter value should be defined in options.fixed_parameters as "
                "a dictionary values, rather than a standalone analysis option. "
                "Please re-save this experiment to be loaded after deprecation period. "
                "This warning will be dropped in v0.4 along with "
                "the support for the deprecated fixed parameter options.",
                DeprecationWarning,
                stacklevel=2,
            )
            new_fixed_params = instance.options.fixed_parameters
            new_fixed_params.update(deprecated_fixed_params)
            instance.set_options(fixed_parameters=new_fixed_params)

        return instance


def is_error_not_significant(
    val: Union[float, UFloat],
    fraction: float = 1.0,
    absolute: Optional[float] = None,
) -> bool:
    """Check if the standard error of given value is not significant.

    Args:
        val: Input value to evaluate. This is assumed to be float or ufloat.
        fraction: Valid fraction of the nominal part to its standard error.
            This function returns ``False`` if the nominal part is
            smaller than the error by this fraction.
        absolute: Use this value as a threshold if given.

    Returns:
        ``True`` if the standard error of given value is not significant.
    """
    if isinstance(val, float):
        return True

    threshold = absolute if absolute is not None else fraction * val.nominal_value
    if np.isnan(val.std_dev) or val.std_dev < threshold:
        return True

    return False


def analysis_result_to_repr(result: AnalysisResultData) -> str:
    """A helper function to create string representation from analysis result data object.

    Args:
        result: Analysis result data.

    Returns:
        String representation of the data.

    Raises:
        AnalysisError: When the result data is not likely fit parameter.
    """
    if not isinstance(result.value, (float, UFloat)):
        raise AnalysisError(f"Result data {result.name} is not a valid fit parameter data type.")

    unit = result.extra.get("unit", None)

    def _format_val(value):
        # Return value with unit with prefix, i.e. 1000 Hz -> 1 kHz.
        if unit:
            try:
                val, val_prefix = detach_prefix(value, decimal=3)
            except ValueError:
                val = value
                val_prefix = ""
            return f"{val: .3g}", f" {val_prefix}{unit}"
        if np.abs(value) < 1e-3 or np.abs(value) > 1e3:
            return f"{value: .4e}", ""
        return f"{value: .4g}", ""

    if isinstance(result.value, float):
        # Only nominal part
        n_repr, n_unit = _format_val(result.value)
        value_repr = n_repr + n_unit
    else:
        # Nominal part
        n_repr, n_unit = _format_val(result.value.nominal_value)

        # Standard error part
        if result.value.std_dev is not None and np.isfinite(result.value.std_dev):
            s_repr, s_unit = _format_val(result.value.std_dev)
            if n_unit == s_unit:
                value_repr = f" {n_repr} \u00B1 {s_repr}{n_unit}"
            else:
                value_repr = f" {n_repr + n_unit} \u00B1 {s_repr + s_unit}"
        else:
            value_repr = n_repr + n_unit

    return f"{result.name} = {value_repr}"
