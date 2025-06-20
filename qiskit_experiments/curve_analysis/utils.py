# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utils in curve analysis."""

from typing import Union, Optional, List, Dict, Tuple, Callable
import time

import asteval
import lmfit
import numpy as np
import pandas as pd
from qiskit.utils import detach_prefix
from uncertainties import UFloat, wrap as wrap_function
from uncertainties import unumpy

from qiskit_experiments.curve_analysis.curve_data import CurveFitResult
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import AnalysisResultData


UNUMPY_FUNCS = {fn: getattr(unumpy, fn) for fn in unumpy.__all__}


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


def convert_lmfit_result(
    result: lmfit.minimizer.MinimizerResult,
    models: List[lmfit.Model],
    xdata: np.ndarray,
    ydata: np.ndarray,
    residuals: np.ndarray,
) -> CurveFitResult:
    """A helper function to convert LMFIT ``MinimizerResult`` into :class:`.CurveFitResult`.

    :class:`.CurveFitResult` is a dataclass that can be serialized with the experiment JSON decoder.
    In addition, this class converts LMFIT ``Parameter`` objects into ufloats so that
    extra parameter computation in the analysis post-processing can perform
    accurate error propagation with parameter correlation.

    Args:
        result: Output from LMFIT ``minimize``.
        models: Model used for the fitting. Function description is extracted.
        xdata: X values used for the fitting.
        ydata: Y values used for the fitting.
        residuals: The residuals of the ydata from the model.

    Returns:
        QiskitExperiments :class:`.CurveFitResult` object.
    """
    model_descriptions = {}
    for model in models:
        if hasattr(model, "expr"):
            func_repr = model.expr
        else:
            signature = ", ".join(model.independent_vars + model.param_names)
            func_repr = f"F({signature})"
        model_descriptions[model._name] = func_repr

    if result is None:
        return CurveFitResult(
            model_repr=model_descriptions,
            success=False,
            x_data=xdata,
            y_data=ydata,
        )

    covar = getattr(result, "covar", None)
    if covar is not None and np.any(np.diag(covar) < 0):
        covar = None

    return CurveFitResult(
        method=result.method,
        model_repr=model_descriptions,
        success=result.success,
        nfev=result.nfev,
        message=result.message,
        dof=result.nfree,
        init_params={name: param.init_value for name, param in result.params.items()},
        chisq=result.chisqr,
        reduced_chisq=result.redchi,
        aic=result.aic,
        bic=result.bic,
        params={name: param.value for name, param in result.params.items()},
        var_names=result.var_names,
        x_data=xdata,
        y_data=ydata,
        weighted_residuals=result.residual,
        residuals=residuals,
        covar=covar,
    )


def eval_with_uncertainties(
    x: np.ndarray,
    model: lmfit.Model,
    params: Dict[str, UFloat],
) -> np.ndarray:
    """Compute Y values with error propagation.

    Args:
        x: X values.
        model: LMFIT model.
        params: Fitter parameters in correlated ufloats.

    Returns:
        Y values with uncertainty (uarray).
    """
    sub_params = {name: params[name] for name in model.param_names}

    if hasattr(model, "expr"):
        # If the model has string expression, we regenerate unumpy fit function.
        # Note that propagating the error through the function requires computation of
        # derivatives, which is done by uncertainties.wrap (or perhaps manually).
        # However, usually computation of derivative is heavy computing overhead,
        # and it is better to use hard-coded derivative functions if it is known.
        # The unumpy functions provide such derivatives, and it's much faster.
        # Here we parse the expression with ASTEVAL, and replace the mapping to
        # the functions in Python's math or numpy with one in unumpy module.
        # Benchmarking with RamseyXY experiment with 100 data points,
        # this yields roughly 60% computation time reduction.
        interpreter = asteval.Interpreter()
        astcode = interpreter.parse(model.expr.strip())

        # Replace function with unumpy version
        interpreter.symtable.update(UNUMPY_FUNCS)
        # Add parameters
        interpreter.symtable.update(sub_params)
        # Add x values
        interpreter.symtable["x"] = x

        interpreter.start_time = time.time()
        try:
            return interpreter.run(astcode)
        except Exception:  # pylint: disable=broad-except
            # User provided function does not support ufloats.
            # Likely using not defined function in unumpy.
            # This falls into normal derivative computation.
            pass

    wrapfunc = np.vectorize(wrap_function(model.func))
    return wrapfunc(x=x, **sub_params)


def shot_weighted_average(
    yvals: np.ndarray,
    yerrs: np.ndarray,
    shots: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute shot based variance and weighted average of the categorized data frame.

    Sample is weighted by the shot number.

    Args:
        yvals: Y values to average.
        yerrs: Y errors to average.
        shots: Number of shots used to obtain Y value and error.

    Returns:
        Averaged Y value, Y error, and total shots.
    """
    if len(yvals) == 1:
        return yvals[0], yerrs[0], shots[0]

    if any(s is pd.NA for s in shots):
        # Shot number is unknown
        return np.mean(yvals), np.nan, pd.NA

    total_shots = np.sum(shots)
    weights = shots / total_shots

    avg_yval = np.sum(weights * yvals)
    avg_yerr = np.sqrt(np.sum(weights**2 * yerrs**2))

    return avg_yval, avg_yerr, total_shots


def inverse_weighted_variance(
    yvals: np.ndarray,
    yerrs: np.ndarray,
    shots: np.ndarray,
) -> Tuple[float, float, int]:
    """Compute inverse weighted variance and weighted average of the categorized data frame.

    Sample is weighted by the inverse of the data variance.

    Args:
        yvals: Y values to average.
        yerrs: Y errors to average.
        shots: Number of shots used to obtain Y value and error.

    Returns:
        Averaged Y value, Y error, and total shots.
    """
    if len(yvals) == 1:
        return yvals[0], yerrs[0], shots[0]

    total_shots = np.sum(shots)
    weights = 1 / yerrs**2
    yvar = 1 / np.sum(weights)

    avg_yval = yvar * np.sum(weights * yvals)
    avg_yerr = np.sqrt(yvar)

    return avg_yval, avg_yerr, total_shots


# pylint: disable=unused-argument
def sample_average(
    yvals: np.ndarray,
    yerrs: np.ndarray,
    shots: np.ndarray,
) -> Tuple[float, float, int]:
    """Compute sample based variance and average of the categorized data frame.

    Original variance of the data is ignored and variance is computed with the y values.

    Args:
        yvals: Y values to average.
        yerrs: Y errors to average (ignored).
        shots: Number of shots used to obtain Y value and error.

    Returns:
        Averaged Y value, Y error, and total shots.
    """
    if len(yvals) == 1:
        return yvals[0], 0.0, shots[0]

    total_shots = np.sum(shots)

    avg_yval = np.mean(yvals)
    avg_yerr = np.sqrt(np.mean((avg_yval - yvals) ** 2) / len(yvals))

    return avg_yval, avg_yerr, total_shots


def level2_probability(data: Dict[str, any], outcome: str) -> Tuple[float, float]:
    """Return the outcome probability mean and variance.

    Args:
        data: A data dict containing count data.
        outcome: bitstring for desired outcome probability.

    Returns:
        tuple: (p_mean, p_var) of the probability mean and variance
               estimated from the counts.

    .. note::

        This assumes a binomial distribution where :math:`K` counts
        of the desired outcome from :math:`N` shots the
        mean probability is :math:`p = K / N` and the variance is
        :math:`\\sigma^2 = p (1-p) / N`.
    """
    counts = data["counts"]

    shots = sum(counts.values())
    p_mean = counts.get(outcome, 0.0) / shots
    p_var = p_mean * (1 - p_mean) / shots
    return p_mean, p_var


def probability(outcome: str) -> Callable:
    """Return probability data processor callback used by the analysis classes."""

    def data_processor(data):
        return level2_probability(data, outcome)

    return data_processor
