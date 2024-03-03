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
import warnings

# pylint: disable=invalid-name

from typing import Dict, List, Tuple, Union, Optional
from functools import partial

from copy import deepcopy
import lmfit
import numpy as np
import pandas as pd
from uncertainties import unumpy as unp

from qiskit_experiments.framework import (
    ExperimentData,
    AnalysisResultData,
)
from qiskit_experiments.framework.containers import FigureType, ArtifactData
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.visualization import PlotStyle

from .base_curve_analysis import BaseCurveAnalysis, DATA_ENTRY_PREFIX, PARAMS_ENTRY_PREFIX
from .curve_data import FitOptions, CurveFitResult
from .scatter_table import ScatterTable
from .utils import (
    eval_with_uncertainties,
    convert_lmfit_result,
    shot_weighted_average,
    inverse_weighted_variance,
    sample_average,
)


class CurveAnalysis(BaseCurveAnalysis):
    """Base class for curve analysis with single curve group.

    The fit parameters from the series defined under the analysis class are all shared
    and the analysis performs a single multi-objective function optimization.

    A subclass may override these methods to customize the fit workflow.

    .. rubric:: _run_data_processing

    This method performs data processing and returns the processed dataset.
    By default, it internally calls the :class:`.DataProcessor` instance from
    the `data_processor` analysis option and processes the experiment data payload
    to create Y data with uncertainty.
    X data and other metadata are generated within this method by inspecting the
    circuit metadata. The series classification is also performed based upon the
    matching of circuit metadata and :attr:`SeriesDef.filter_kwargs`.

    .. rubric:: _format_data

    This method consumes the processed dataset and outputs the formatted dataset.
    By default, this method takes the average of y values over
    the same x values and then sort the entire data by x values.

    .. rubric:: _generate_fit_guesses

    This method creates initial guesses for the fit parameters.
    See :ref:`curve_analysis_init_guess` for details.

    .. rubric:: _run_curve_fit

    This method performs the fitting with predefined fit models and the formatted dataset.
    This method internally calls the :meth:`_generate_fit_guesses` method.
    Note that this is a core functionality of the :meth:`_run_analysis` method,
    that creates fit result objects from the formatted dataset.

    .. rubric:: _evaluate_quality

    This method evaluates the quality of the fit based on the fit result.
    This returns "good" when reduced chi-squared is less than 3.0.
    Usually it returns string "good" or "bad" according to the evaluation.

    .. rubric:: _create_analysis_results

    This method creates analysis results for important fit parameters
    that might be defined by analysis options ``result_parameters``.

    .. rubric:: _create_figures

    This method creates figures by consuming the scatter table data.
    Figures are created when the analysis option ``plot`` is ``True``.

    .. rubric:: _initialize

    This method initializes analysis options against input experiment data.
    Usually this method is called before other methods are called.

    """

    def __init__(
        self,
        models: Optional[List[lmfit.Model]] = None,
        name: Optional[str] = None,
    ):
        """Initialize data fields that are privately accessed by methods.

        Args:
            models: List of LMFIT ``Model`` class to define fitting functions and
                parameters. If multiple models are provided, the analysis performs
                multi-objective optimization where the parameters with the same name
                are shared among provided models. When multiple models are provided,
                user must specify the ``data_subfit_map`` value in the analysis options
                to allocate experimental results to a particular fit model.
            name: Optional. Name of this analysis.
        """
        super().__init__()

        self._models = models or []
        self._name = name or self.__class__.__name__
        self._plot_config_cache = {}

    @property
    def name(self) -> str:
        """Return name of this analysis."""
        return self._name

    @property
    def parameters(self) -> List[str]:
        """Return parameters of this curve analysis."""
        unite_params = []
        for model in self._models:
            for name in model.param_names:
                if name not in unite_params and name not in self.options.fixed_parameters:
                    unite_params.append(name)
        return unite_params

    @property
    def models(self) -> List[lmfit.Model]:
        """Return fit models."""
        return self._models

    def model_names(self) -> List[str]:
        """Return model names."""
        return [getattr(m, "_name", f"model-{i}") for i, m in enumerate(self._models)]

    def set_options(self, **fields):
        """Set the analysis options for :meth:`run` method.

        Args:
            fields: The fields to update the options

        Raises:
            KeyError: When removed option ``curve_fitter`` is set.
        """
        if fields.get("plot_residuals") and not self.options.get("plot_residuals"):
            # checking there are no subplots for the figure to prevent collision in subplot indices.
            if self.plotter.options.get("subplots") != (1, 1):
                warnings.warn(
                    "Residuals plotting is currently supported for analysis with 1 subplot.",
                    UserWarning,
                    stacklevel=2,
                )
                fields["plot_residuals"] = False
            else:
                self._add_residuals_plot_config()
        if not fields.get("plot_residuals", True) and self.options.get("plot_residuals"):
            self._remove_residuals_plot_config()

        super().set_options(**fields)

    def _add_residuals_plot_config(self):
        """Configure plotter options for residuals plot."""
        # check we have model to fit into
        residual_plot_y_axis_size = 3
        if self.models:
            # Cache figure options.
            self._plot_config_cache["figure_options"] = {}
            self._plot_config_cache["figure_options"]["ylabel"] = self.plotter.figure_options.get(
                "ylabel"
            )
            self._plot_config_cache["figure_options"]["series_params"] = deepcopy(
                self.plotter.figure_options.get("series_params")
            )
            self._plot_config_cache["figure_options"]["sharey"] = self.plotter.figure_options.get(
                "sharey"
            )

            self.plotter.set_figure_options(
                ylabel=[
                    self.plotter.figure_options.get("ylabel", ""),
                    "Residuals",
                ],
            )

            model_names = self.model_names()
            series_params = self.plotter.figure_options["series_params"]
            for model_name in model_names:
                if series_params.get(model_name):
                    series_params[model_name]["canvas"] = 0
                else:
                    series_params[model_name] = {"canvas": 0}
                series_params[model_name + "_residuals"] = series_params[model_name].copy()
                series_params[model_name + "_residuals"]["canvas"] = 1
            self.plotter.set_figure_options(sharey=False, series_params=series_params)

            # Cache plotter options.
            self._plot_config_cache["plotter"] = {}
            self._plot_config_cache["plotter"]["subplots"] = self.plotter.options.get("subplots")
            self._plot_config_cache["plotter"]["style"] = deepcopy(
                self.plotter.options.get("style", PlotStyle({}))
            )

            # removing the name from the plotter style, so it will not clash with the new name
            previous_plotter_style = self._plot_config_cache["plotter"]["style"].copy()
            previous_plotter_style.pop("style_name", "")

            # creating new fig size based on previous size
            new_figsize = self.plotter.drawer.options.get("figsize", (8, 5))
            new_figsize = (new_figsize[0], new_figsize[1] + residual_plot_y_axis_size)

            # Here add the configuration for the residuals plot:
            self.plotter.set_options(
                subplots=(2, 1),
                style=PlotStyle.merge(
                    PlotStyle(
                        {
                            "figsize": new_figsize,
                            "textbox_rel_pos": (0.28, -0.10),
                            "sub_plot_heights_list": [7 / 10, 3 / 10],
                            "sub_plot_widths_list": [1],
                            "style_name": "residuals",
                        }
                    ),
                    previous_plotter_style,
                ),
            )

    def _remove_residuals_plot_config(self):
        """set options for a single plot to its cached values."""
        if self.models:
            self.plotter.set_figure_options(
                ylabel=self._plot_config_cache["figure_options"]["ylabel"],
                sharey=self._plot_config_cache["figure_options"]["sharey"],
                series_params=self._plot_config_cache["figure_options"]["series_params"],
            )

            # Here add the style_name so the plotter will know not to print the residual data.
            self.plotter.set_options(
                subplots=self._plot_config_cache["plotter"]["subplots"],
                style=PlotStyle.merge(
                    self._plot_config_cache["plotter"]["style"],
                    PlotStyle({"style_name": "canceled_residuals"}),
                ),
            )

        self._plot_config_cache = {}

    def _run_data_processing(
        self,
        raw_data: List[Dict],
        category: str = "raw",
    ) -> ScatterTable:
        """Perform data processing from the experiment result payload.

        Args:
            raw_data: Payload in the experiment data.
            category: Category string of the output dataset.

        Returns:
            Processed data that will be sent to the formatter method.

        Raises:
            DataProcessorError: When key for x values is not found in the metadata.
            ValueError: When data processor is not provided.
        """
        opt = self.options

        # Create table
        if opt.filter_data:
            to_process = [d for d in raw_data if opt.filter_data.items() <= d["metadata"].items()]
        else:
            to_process = raw_data

        # Compute y value
        if not self.options.data_processor:
            raise ValueError(
                f"Data processor is not set for the {self.__class__.__name__} instance. "
                "Initialize the instance with the experiment data, or set the "
                "data_processor analysis options."
            )
        processed = self.options.data_processor(to_process)
        yvals = unp.nominal_values(processed).flatten()
        with np.errstate(invalid="ignore"):
            # For averaged data, the processed std dev will be NaN.
            # Setting std_devs to NaN will trigger floating point exceptions
            # which we can ignore. See https://stackoverflow.com/q/75656026
            yerrs = unp.std_devs(processed).flatten()

        # Prepare circuit metadata to data class mapper from data_subfit_map value.
        if len(self._models) == 1:
            classifier = {self.model_names()[0]: {}}
        else:
            classifier = self.options.data_subfit_map

        table = ScatterTable()
        for datum, yval, yerr in zip(to_process, yvals, yerrs):
            metadata = datum["metadata"]
            try:
                xval = metadata[opt.x_key]
            except KeyError as ex:
                raise DataProcessorError(
                    f"X value key {opt.x_key} is not defined in the circuit metadata."
                ) from ex

            # Assign series name and series id
            for series_id, (series_name, spec) in enumerate(classifier.items()):
                if spec.items() <= metadata.items():
                    break
            else:
                # This is unclassified data.
                series_name = pd.NA
                series_id = pd.NA
            table.add_row(
                xval=xval,
                yval=yval,
                yerr=yerr,
                series_name=series_name,
                series_id=series_id,
                category=category,
                shots=datum.get("shots", pd.NA),
                analysis=self.name,
            )
        return table

    def _format_data(
        self,
        curve_data: ScatterTable,
        category: str = "formatted",
    ) -> ScatterTable:
        """Postprocessing for preparing the fitting data.

        Args:
            curve_data: Processed dataset created from experiment results.
            category: Category string of the output dataset.

        Returns:
            New scatter table instance including data to fit.
        """
        averaging_methods = {
            "shots_weighted": shot_weighted_average,
            "iwv": inverse_weighted_variance,
            "sample": sample_average,
        }
        average = averaging_methods[self.options.average_method]
        model_names = self.model_names()

        for (series_name, xval), sub_data in curve_data.iter_groups("series_name", "xval"):
            avg_yval, avg_yerr, shots = average(
                sub_data.y,
                sub_data.y_err,
                sub_data.shots,
            )
            try:
                series_id = model_names.index(series_name)
            except ValueError:
                series_id = pd.NA
            curve_data.add_row(
                xval=xval,
                yval=avg_yval,
                yerr=avg_yerr,
                series_name=series_name,
                series_id=series_id,
                category=category,
                shots=shots,
                analysis=self.name,
            )

        return curve_data

    def _generate_fit_guesses(
        self,
        user_opt: FitOptions,
        curve_data: ScatterTable,  # pylint: disable=unused-argument
    ) -> Union[FitOptions, List[FitOptions]]:
        """Create algorithmic initial fit guess from analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        return user_opt

    def _run_curve_fit(
        self,
        curve_data: ScatterTable,
    ) -> CurveFitResult:
        """Perform curve fitting on given data collection and fit models.

        Args:
            curve_data: Formatted data to fit.

        Returns:
            The best fitting outcome with minimum reduced chi-squared value.
        """
        unite_parameter_names = []
        for model in self._models:
            # Seems like this is not efficient looping, but using set operation sometimes
            # yields bad fit. Not sure if this is an edge case, but
            # `TestRamseyXY` unittest failed due to the significant chisq value
            # in which the least_square fitter terminates with `xtol` rather than `ftol`
            # condition, i.e. `ftol` condition indicates termination by cost function.
            # This code respects the ordering of parameters so that it matches with
            # the signature of fit function and it is backward compatible.
            # In principle this should not matter since LMFIT maps them with names
            # rather than index. Need more careful investigation.
            for name in model.param_names:
                if name not in unite_parameter_names:
                    unite_parameter_names.append(name)

        default_fit_opt = FitOptions(
            parameters=unite_parameter_names,
            default_p0=self.options.p0,
            default_bounds=self.options.bounds,
            **self.options.lmfit_options,
        )

        # Bind fixed parameters if not empty
        if self.options.fixed_parameters:
            fixed_parameters = {
                k: v for k, v in self.options.fixed_parameters.items() if k in unite_parameter_names
            }
            default_fit_opt.p0.set_if_empty(**fixed_parameters)
        else:
            fixed_parameters = {}

        fit_options = self._generate_fit_guesses(default_fit_opt, curve_data)

        if isinstance(fit_options, FitOptions):
            fit_options = [fit_options]

        # Create convenient function to compute residual of the models.
        partial_weighted_residuals = []
        valid_uncertainty = np.all(np.isfinite(curve_data.y_err))

        # creating storage for residual plotting
        if self.options.get("plot_residuals"):
            residual_weights_list = []

        for idx, sub_data in curve_data.iter_by_series_id():
            if valid_uncertainty:
                nonzero_yerr = np.where(
                    np.isclose(sub_data.y_err, 0.0),
                    np.finfo(float).eps,
                    sub_data.y_err,
                )
                raw_weights = 1 / nonzero_yerr
                # Remove outlier. When all sample values are the same with sample average,
                # or sampling error is zero with shot-weighted average,
                # some yerr values might be very close to zero, yielding significant weights.
                # With such outlier, the fit doesn't sense residual of other data points.
                maximum_weight = np.percentile(raw_weights, 90)
                weights_list = np.clip(raw_weights, 0.0, maximum_weight)
            else:
                weights_list = None
            model_weighted_residual = partial(
                self._models[idx]._residual,
                data=sub_data.y,
                weights=weights_list,
                x=sub_data.x,
            )
            partial_weighted_residuals.append(model_weighted_residual)

            # adding weights to weights_list for residuals
            if self.options.get("plot_residuals"):
                if weights_list is None:
                    residual_weights_list.append(None)
                else:
                    residual_weights_list.append(weights_list)

        # Run fit for each configuration
        res = None
        for fit_option in fit_options:
            # Setup parameter configuration, i.e. init value, bounds
            guess_params = lmfit.Parameters()
            for name in unite_parameter_names:
                bounds = fit_option.bounds[name] or (-np.inf, np.inf)
                guess_params.add(
                    name=name,
                    value=fit_option.p0[name],
                    min=bounds[0],
                    max=bounds[1],
                    vary=name not in fixed_parameters,
                )

            try:
                with np.errstate(all="ignore"):
                    new = lmfit.minimize(
                        fcn=lambda x: np.concatenate([p(x) for p in partial_weighted_residuals]),
                        params=guess_params,
                        method=self.options.fit_method,
                        scale_covar=not valid_uncertainty,
                        nan_policy="omit",
                        **fit_option.fitter_opts,
                    )
            except Exception:  # pylint: disable=broad-except
                continue

            if res is None or not res.success:
                res = new
                continue

            if new.success and res.redchi > new.redchi:
                res = new

        # if `plot_residuals` is ``False`` I would like the `residuals_model` be None to emphasize it
        # wasn't calculated.
        residuals_model = [] if self.options.get("plot_residuals") else None
        if res and res.success and self.options.get("plot_residuals"):
            for weights in residual_weights_list:
                if weights is None:
                    residuals_model.append(res.residual)
                else:
                    residuals_model.append(
                        [
                            weighted_res / np.abs(weight)
                            for weighted_res, weight in zip(res.residual, weights)
                        ]
                    )

        if residuals_model is not None:
            residuals_model = np.array(residuals_model)

        return convert_lmfit_result(
            res,
            self._models,
            curve_data.x,
            curve_data.y,
            residuals_model,
        )

    def _create_figures(
        self,
        curve_data: ScatterTable,
    ) -> List["matplotlib.figure.Figure"]:
        """Create a list of figures from the curve data.

        Args:
            curve_data: Scatter data table containing all data points.

        Returns:
            A list of figures.
        """
        for series_id, sub_data in curve_data.iter_by_series_id():
            model_name = self.model_names()[series_id]
            # Plot raw data scatters
            if self.options.plot_raw_data:
                raw_data = sub_data.filter(category="raw")
                self.plotter.set_series_data(
                    series_name=model_name,
                    x=raw_data.x,
                    y=raw_data.y,
                )
            # Plot formatted data scatters
            formatted_data = sub_data.filter(category=self.options.fit_category)
            self.plotter.set_series_data(
                series_name=model_name,
                x_formatted=formatted_data.x,
                y_formatted=formatted_data.y,
                y_formatted_err=formatted_data.y_err,
            )
            # Plot fit lines
            line_data = sub_data.filter(category="fitted")
            if len(line_data) == 0:
                continue
            self.plotter.set_series_data(
                series_name=model_name,
                x_interp=line_data.x,
                y_interp=line_data.y,
            )
            fit_stdev = line_data.y_err
            if np.isfinite(fit_stdev).all():
                self.plotter.set_series_data(
                    series_name=model_name,
                    y_interp_err=fit_stdev,
                )

            if self.options.get("plot_residuals"):
                residuals_data = sub_data.filter(category="residuals")
                self.plotter.set_series_data(
                    series_name=model_name,
                    x_residuals=residuals_data.x,
                    y_residuals=residuals_data.y,
                )

        return [self.plotter.figure()]

    def _run_analysis(
        self,
        experiment_data: ExperimentData,
    ) -> Tuple[List[Union[AnalysisResultData, ArtifactData]], List[FigureType]]:
        figures: List[FigureType] = []
        result_data: List[Union[AnalysisResultData, ArtifactData]] = []
        artifacts: list[ArtifactData] = []

        # Flag for plotting can be "always", "never", or "selective"
        # the analysis option overrides self._generate_figures if set
        if self.options.get("plot", None):
            plot = "always"
        elif self.options.get("plot", None) is False:
            plot = "never"
        else:
            plot = getattr(self, "_generate_figures", "always")

        # Prepare for fitting
        self._initialize(experiment_data)

        table = self._format_data(self._run_data_processing(experiment_data.data()))
        formatted_subset = table.filter(category=self.options.fit_category)
        fit_data = self._run_curve_fit(formatted_subset)

        if fit_data.success:
            quality = self._evaluate_quality(fit_data)
        else:
            quality = "bad"

        # After the quality is determined, plot can become a boolean flag for whether
        # to generate the figure
        plot_bool = plot == "always" or (plot == "selective" and quality == "bad")

        if self.options.return_fit_parameters:
            # Store fit status overview entry regardless of success.
            # This is sometime useful when debugging the fitting code.
            overview = AnalysisResultData(
                name=PARAMS_ENTRY_PREFIX + self.name,
                value=fit_data,
                quality=quality,
                extra=self.options.extra,
            )
            result_data.append(overview)

        if fit_data.success:
            # Add fit data to curve data table
            model_names = self.model_names()
            for series_id, sub_data in formatted_subset.iter_by_series_id():
                xval = sub_data.x
                if len(xval) == 0:
                    # If data is empty, skip drawing this model.
                    # This is the case when fit model exist but no data to fit is provided.
                    continue
                # Compute X, Y values with fit parameters.
                xval_arr_fit = np.linspace(np.min(xval), np.max(xval), num=100, dtype=float)
                uval_arr_fit = eval_with_uncertainties(
                    x=xval_arr_fit,
                    model=self._models[series_id],
                    params=fit_data.ufloat_params,
                )
                yval_arr_fit = unp.nominal_values(uval_arr_fit)
                if fit_data.covar is not None:
                    yerr_arr_fit = unp.std_devs(uval_arr_fit)
                else:
                    yerr_arr_fit = np.zeros_like(xval_arr_fit)
                for xval, yval, yerr in zip(xval_arr_fit, yval_arr_fit, yerr_arr_fit):
                    table.add_row(
                        xval=xval,
                        yval=yval,
                        yerr=yerr,
                        series_name=model_names[series_id],
                        series_id=series_id,
                        category="fitted",
                        analysis=self.name,
                    )

                if self.options.get("plot_residuals"):
                    # need to add here the residuals plot.
                    xval_residual = sub_data.x
                    yval_residuals = unp.nominal_values(fit_data.residuals[series_id])

                    for xval, yval in zip(xval_residual, yval_residuals):
                        table.add_row(
                            xval=xval,
                            yval=yval,
                            series_name=model_names[series_id],
                            series_id=series_id,
                            category="residuals",
                            analysis=self.name,
                        )

            result_data.extend(
                self._create_analysis_results(
                    fit_data=fit_data,
                    quality=quality,
                    **self.options.extra.copy(),
                )
            )

        if self.options.return_data_points:
            # Add raw data points
            warnings.warn(
                f"{DATA_ENTRY_PREFIX + self.name} has been moved to experiment data artifacts. "
                "Saving this result with 'return_data_points'=True will be disabled in "
                "Qiskit Experiments 0.7.",
                DeprecationWarning,
            )
            result_data.extend(self._create_curve_data(curve_data=formatted_subset))

        artifacts.append(
            ArtifactData(
                name="curve_data",
                data=table,
            )
        )
        artifacts.append(
            ArtifactData(
                name="fit_summary",
                data=fit_data,
            )
        )

        if plot_bool:
            if fit_data.success:
                self.plotter.set_supplementary_data(
                    fit_red_chi=fit_data.reduced_chisq,
                    primary_results=[r for r in result_data if not r.name.startswith("@")],
                )
            figures.extend(self._create_figures(curve_data=table))

        return result_data + artifacts, figures

    def __getstate__(self):
        state = self.__dict__.copy()
        # Convert models into JSON str.
        # This object includes local function and cannot be pickled.
        source = [m.dumps() for m in state["_models"]]
        state["_models"] = source
        return state

    def __setstate__(self, state):
        model_objs = []
        for source in state.pop("_models"):
            tmp_mod = lmfit.Model(func=None)
            mod = tmp_mod.loads(s=source)
            model_objs.append(mod)
        self.__dict__.update(state)
        self._models = model_objs
