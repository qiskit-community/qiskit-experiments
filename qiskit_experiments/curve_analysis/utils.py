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
from qiskit.utils.deprecation import deprecate_func
from qiskit.utils import detach_prefix
from uncertainties import UFloat, wrap as wrap_function
from uncertainties import unumpy

from qiskit_experiments.curve_analysis.curve_data import CurveFitResult
from qiskit_experiments.exceptions import AnalysisError, QiskitError
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


@deprecate_func(
    since="0.6",
    additional_msg="The curve data representation has been replaced by the `DataFrame` format.",
    package_name="qiskit-experiments",
    pending=True,
)
def filter_data(data: List[Dict[str, any]], **filters) -> List[Dict[str, any]]:
    """Return the list of filtered data

    Args:
        data: list of data dicts.
        filters: kwargs for filtering based on metadata
                 values.

    Returns:
        The list of filtered data. If no filters are provided this will be the
        input list.
    """
    if not filters:
        return data
    filtered_data = []
    for datum in data:
        include = True
        metadata = datum["metadata"]
        for key, val in filters.items():
            if key not in metadata or metadata[key] != val:
                include = False
                break
        if include:
            filtered_data.append(datum)
    return filtered_data


@deprecate_func(
    since="0.6",
    additional_msg="The curve data representation has been replaced by the `DataFrame` format.",
    package_name="qiskit-experiments",
    pending=True,
)
def mean_xy_data(
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    shots: Optional[np.ndarray] = None,
    method: str = "sample",
) -> Tuple[np.ndarray, ...]:
    r"""Return (x, y_mean, sigma) data.

    The mean is taken over all :math:`y` data values with the same :math:`x` data value using
    the specified method. For each :math:`x` the mean :math:`\overline{y}` and variance
    :math:`\sigma^2` are computed as

    * ``"sample"`` (default): *Sample mean and variance*

      * :math:`\overline{y} = \sum_{i=1}^N y_i / N`,

      * :math:`\sigma^2 = \sum_{i=1}^N ((\overline{y} - y_i)^2) / N`

    * ``"iwv"``: *Inverse-weighted variance*

      * :math:`\overline{y} = (\sum_{i=1}^N y_i / \sigma_i^2 ) \sigma^2`
      * :math:`\sigma^2 = 1 / (\sum_{i=1}^N 1 / \sigma_i^2)`

    * ``"shots_weighted_variance"``: *Sample mean and variance with weights from shots*

      * :math:`\overline{y} = \sum_{i=1}^N n_i y_i / M`,

      * :math:`\sigma^2 = \sum_{i=1}^N (n_i \sigma_i / M)^2`,
        where :math:`n_i` is the number of shots per data point and :math:`M = \sum_{i=1}^N n_i`
        is a total number of shots from different circuit execution at the same :math:`x` value.
        If ``shots`` is not provided, this applies uniform weights to all values.

    Args:
        xdata: 1D or 2D array of xdata from curve_fit_data or
            multi_curve_fit_data
        ydata: array of ydata returned from curve_fit_data or
            multi_curve_fit_data
        sigma: Optional, array of standard deviations in ydata.
        shots: Optional, array of shots used to get a data point.
        method: The method to use for computing y means and
            standard deviations sigma (default: "sample").

    Returns:
        tuple: ``(x, y_mean, sigma, shots)``, where ``x`` is an arrays of unique
        x-values, ``y`` is an array of sample mean y-values, ``sigma`` is an array of
        sample standard deviation of y values, and ``shots`` are the total number of
        experiment shots used to evaluate the data point. If ``shots`` in the function
        call is ``None``, the numbers appear in the returned value will represent just a
        number of duplicated x value entries.

    Raises:
        QiskitError: If the "ivw" method is used without providing a sigma.
    """
    x_means = np.unique(xdata, axis=0)
    y_means = np.zeros(x_means.size)
    y_sigmas = np.zeros(x_means.size)
    y_shots = np.zeros(x_means.size)

    if shots is None or any(np.isnan(shots)):
        # this will become standard average
        shots = np.ones_like(xdata)

    # Sample mean and variance method
    if method == "sample":
        for i in range(x_means.size):
            # Get positions of y to average
            idxs = xdata == x_means[i]
            ys = ydata[idxs]
            ns = shots[idxs]

            # Compute sample mean and sample standard error of the mean
            y_means[i] = np.mean(ys)
            y_sigmas[i] = np.sqrt(np.mean((y_means[i] - ys) ** 2) / ys.size)
            y_shots[i] = np.sum(ns)

        return x_means, y_means, y_sigmas, y_shots

    # Inverse-weighted variance method
    if method == "iwv":
        if sigma is None:
            raise QiskitError(
                "The inverse-weighted variance method cannot be used with `sigma=None`"
            )
        for i in range(x_means.size):
            # Get positions of y to average
            idxs = xdata == x_means[i]
            ys = ydata[idxs]
            ns = shots[idxs]

            # Compute the inverse-variance weighted y mean and variance
            weights = 1 / sigma[idxs] ** 2
            y_var = 1 / np.sum(weights)
            y_means[i] = y_var * np.sum(weights * ys)
            y_sigmas[i] = np.sqrt(y_var)
            y_shots[i] = np.sum(ns)

        return x_means, y_means, y_sigmas, y_shots

    # Quadrature sum of variance
    if method == "shots_weighted":
        for i in range(x_means.size):
            # Get positions of y to average
            idxs = xdata == x_means[i]
            ys = ydata[idxs]
            ss = sigma[idxs]
            ns = shots[idxs]
            weights = ns / np.sum(ns)

            # Compute sample mean and sum of variance with weights based on shots
            y_means[i] = np.sum(weights * ys)
            y_sigmas[i] = np.sqrt(np.sum(weights**2 * ss**2))
            y_shots[i] = np.sum(ns)

        return x_means, y_means, y_sigmas, y_shots

    # Invalid method
    raise QiskitError(f"Unsupported method {method}")


@deprecate_func(
    since="0.6",
    additional_msg="The curve data representation has been replaced by the `DataFrame` format.",
    package_name="qiskit-experiments",
    pending=True,
)
def multi_mean_xy_data(
    series: np.ndarray,
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    shots: Optional[np.ndarray] = None,
    method: str = "sample",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Take mean of multi series data set. See :func:`.mean_xy_data`.

    Args:
        series: Series index.
        xdata: 1D or 2D array of xdata from curve_fit_data or
               multi_curve_fit_data
        ydata: array of ydata returned from curve_fit_data or
               multi_curve_fit_data
        sigma: Optional, array of standard deviations in ydata.
        shots: Optional, array of shots used to get a data point.
        method: The method to use for computing y means and
                standard deviations sigma (default: "sample").

    Returns:
        Tuple of ``(series, xdata, ydata, sigma, shots)``.

    """
    series_vals = np.unique(series)

    series_means = []
    xdata_means = []
    ydata_means = []
    sigma_means = []
    shots_sums = []

    # Get x, y, sigma data for series and process mean data
    for series_val in series_vals:
        idxs = series == series_val
        sigma_i = sigma[idxs] if sigma is not None else None
        shots_i = shots[idxs] if shots is not None else None

        x_mean, y_mean, sigma_mean, shots_sum = mean_xy_data(
            xdata[idxs], ydata[idxs], sigma=sigma_i, shots=shots_i, method=method
        )
        series_means.append(np.full(x_mean.size, series_val, dtype=int))
        xdata_means.append(x_mean)
        ydata_means.append(y_mean)
        sigma_means.append(sigma_mean)
        shots_sums.append(shots_sum)

    # Concatenate lists
    return (
        np.concatenate(series_means),
        np.concatenate(xdata_means),
        np.concatenate(ydata_means),
        np.concatenate(sigma_means),
        np.concatenate(shots_sums),
    )


@deprecate_func(
    since="0.6",
    additional_msg="The curve data representation has been replaced by the `DataFrame` format.",
    package_name="qiskit-experiments",
    pending=True,
)
def data_sort(
    series: np.ndarray,
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    shots: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort data.

    Input x values may not be lined up in order, since experiment may accept user input array,
    or data may be concatenated with previous scan. This sometimes confuses the algorithmic
    generation of initial guesses especially when guess depends on derivative.

    This returns data set that is sorted by xdata and series in ascending order.

    Args:
        series: Series index.
        xdata: 1D or 2D array of xdata.
        ydata: Array of ydata.
        sigma: Optional, array of standard deviations in ydata.
        shots: Optional, array of shots used to get a data point.

    Returns:
        Tuple of (series, xdata, ydata, sigma, shots) sorted in ascending order of xdata
        and series.
    """
    if sigma is None:
        sigma = np.full(series.size, np.nan, dtype=float)

    if shots is None:
        shots = np.full(series.size, np.nan, dtype=float)

    sorted_data = sorted(zip(series, xdata, ydata, sigma, shots), key=lambda d: (d[0], d[1]))

    return np.asarray(sorted_data).T


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
