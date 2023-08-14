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
Analysis class for multi-group curve fitting.
"""
# pylint: disable=invalid-name
import warnings
from typing import Dict, List, Optional, Tuple, Union

import lmfit
import numpy as np
import pandas as pd
from uncertainties import unumpy as unp

from qiskit.utils.deprecation import deprecate_func

from qiskit_experiments.framework import (
    AnalysisResultData,
    BaseAnalysis,
    ExperimentData,
    Options,
)
from qiskit_experiments.visualization import (
    BaseDrawer,
    BasePlotter,
    CurvePlotter,
    LegacyCurveCompatDrawer,
    MplDrawer,
)

from .base_curve_analysis import PARAMS_ENTRY_PREFIX, BaseCurveAnalysis
from .curve_data import CurveFitResult
from .scatter_table import ScatterTable
from .utils import eval_with_uncertainties


class CompositeCurveAnalysis(BaseAnalysis):
    r"""Composite Curve Analysis.

    The :class:`.CompositeCurveAnalysis` takes multiple curve analysis instances
    and performs each analysis on the same experimental results.
    These analyses are performed independently, thus fit parameters have no correlation.
    Note that this is different from :class:`.CompositeAnalysis` which
    analyses the outcome of a composite experiment, in which multiple different
    experiments are performed.
    The :class:`.CompositeCurveAnalysis` is attached to a single experiment instance,
    which may execute similar circuits with slightly different settings.
    Experiments with different settings might be distinguished by the circuit
    metadata. The outcomes of the same set of experiments are assigned to a
    specific analysis instance in the composite curve analysis.
    This mapping is usually done with the analysis option ``filter_data`` dictionary.
    Otherwise, all analyses are performed on the same set of outcomes.

    Examples:

        In this example, we write up a composite analysis consisting of two oscillation
        analysis instances, assuming two Rabi experiments in 1-2 subspace
        starting with different initial states :math:`\in \{|0\rangle, |1\rangle\}`.
        This is a typical procedure to measure the thermal population of the qubit.

        .. code-block:: python

            from qiskit_experiments import curve_analysis as curve

            analyses = []
            for qi in (0, 1):
                analysis = curve.OscillationAnalysis(name=f"init{qi}")
                analysis.set_options(
                    return_fit_parameters=["freq"],
                    filter_data={"init_state": qi},
                )
            analysis = CompositeCurveAnalysis(analyses=analyses)

        This ``analysis`` will return two analysis result data for the fit parameter "freq"
        for experiments with the initial state :math:`|0\rangle` and :math:`|1\rangle`.
        The experimental circuits starting with different initial states must be
        distinguished by the circuit metadata ``{"init_state": 0}`` or ``{"init_state": 1}``,
        along with the "xval" in the same dictionary.
        If you want to compute another quantity using two fitting outcomes, you can
        override :meth:`CompositeCurveAnalysis._create_curve_data` in subclass.

    :class:`.CompositeCurveAnalysis` subclass may override following methods.

    .. rubric:: _evaluate_quality

    This method evaluates the quality of the composite fit based on
    the all analysis outcomes.
    This returns "good" when all fit outcomes are evaluated as "good",
    otherwise it returns "bad".

    .. rubric:: _create_analysis_results

    This method is passed all the group fit outcomes and can return a list of
    new values to be stored in the analysis results.

    .. rubric:: _create_figures

    This method creates figures by consuming the scatter table data.
    Figures are created when the analysis option ``plot`` is ``True``.

    """

    def __init__(
        self,
        analyses: List[BaseCurveAnalysis],
        name: Optional[str] = None,
    ):
        super().__init__()

        self._analyses = analyses
        self._name = name or self.__class__.__name__

    @property
    def parameters(self) -> List[str]:
        """Return parameters of this curve analysis."""
        unite_params = []
        for analysis in self._analyses:
            # Respect ordering of parameters
            for name in analysis.parameters:
                if name not in unite_params:
                    unite_params.append(name)
        return unite_params

    @property
    def name(self) -> str:
        """Return name of this analysis."""
        return self._name

    @property
    def models(self) -> Dict[str, List[lmfit.Model]]:
        """Return fit models."""
        models = {}
        for analysis in self._analyses:
            models[analysis.name] = analysis.models
        return models

    @property
    def plotter(self) -> BasePlotter:
        """A short-cut to the plotter instance."""
        return self._options.plotter

    @property
    @deprecate_func(
        since="0.5",
        additional_msg="Use `plotter` from the new visualization module instead.",
        removal_timeline="after 0.6",
        package_name="qiskit-experiments",
    )
    def drawer(self) -> BaseDrawer:
        """A short-cut for curve drawer instance, if set. ``None`` otherwise."""
        if hasattr(self._options, "curve_drawer"):
            return self._options.curve_drawer
        else:
            return None

    def analyses(
        self, index: Optional[Union[str, int]] = None
    ) -> Union[BaseCurveAnalysis, List[BaseCurveAnalysis]]:
        """Return curve analysis instance.

        Args:
            index: Name of group or numerical index.

        Returns:
            Curve analysis instance.
        """
        if index is None:
            return self._analyses
        if isinstance(index, str):
            group_names = [analysis.name for analysis in self._analyses]
            num_index = group_names.index(index)
            return self._analyses[num_index]
        return self._analyses[index]

    def _evaluate_quality(
        self,
        fit_data: Dict[str, CurveFitResult],
    ) -> Union[str, None]:
        """Evaluate quality of the fit result.

        Args:
            fit_data: Fit outcome keyed on the analysis name.

        Returns:
            String that represents fit result quality. Usually "good" or "bad".
        """
        for analysis in self._analyses:
            if analysis._evaluate_quality(fit_data[analysis.name]) != "good":
                return "bad"
        return "good"

    # pylint: disable=unused-argument
    def _create_analysis_results(
        self,
        fit_data: Dict[str, CurveFitResult],
        quality: str,
        **metadata,
    ) -> List[AnalysisResultData]:
        """Create analysis results based on all analysis outcomes.

        Args:
            fit_data: Fit outcome keyed on the analysis name.
            quality: Quality of fit outcome.

        Returns:
            List of analysis result data.
        """
        return []

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
        for analysis in self.analyses():
            sub_data = curve_data[curve_data.model_name.str.endswith(f"_{analysis.name}")]
            for model_name, data in list(sub_data.groupby("model_name")):
                # Plot raw data scatters
                if analysis.options.plot_raw_data:
                    raw_data = data[data.format == "raw"]
                    self.plotter.set_series_data(
                        series_name=model_name,
                        x=raw_data.xval.to_numpy(),
                        y=raw_data.yval.to_numpy(),
                    )
                # Plot formatted data scatters
                formatted_data = data[data.format == "fit-ready"]
                self.plotter.set_series_data(
                    series_name=model_name,
                    x_formatted=formatted_data.xval.to_numpy(),
                    y_formatted=formatted_data.yval.to_numpy(),
                    y_formatted_err=formatted_data.yerr.to_numpy(),
                )
                # Plot fit lines
                line_data = data[data.format == "fit"]
                if len(line_data) == 0:
                    continue
                fit_stdev = line_data.yerr.to_numpy()
                self.plotter.set_series_data(
                    series_name=model_name,
                    x_interp=line_data.xval.to_numpy(),
                    y_interp=line_data.yval.to_numpy(),
                    y_interp_err=fit_stdev if np.isfinite(fit_stdev).all() else None,
                )

        return [self.plotter.figure()]

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options.

        Analysis Options:
            plotter (BasePlotter): A plotter instance to visualize
                the analysis result.
            plot (bool): Set ``True`` to create figure for fit result.
                This is ``True`` by default.
            return_fit_parameters (bool): Set ``True`` to return all fit model parameters
                with details of the fit outcome. Default to ``True``.
            return_data_points (bool): Set ``True`` to include in the analysis result
                the formatted data points given to the fitter. Default to ``False``.
            extra (Dict[str, Any]): A dictionary that is appended to all database entries
                as extra information.
        """
        options = super()._default_options()
        options.update_options(
            plotter=CurvePlotter(MplDrawer()),
            plot=True,
            return_fit_parameters=True,
            return_data_points=False,
            extra={},
        )

        # Set automatic validator for particular option values
        options.set_validator(field="plotter", validator_value=BasePlotter)

        return options

    def set_options(self, **fields):
        # TODO remove this in Qiskit Experiments 0.6
        if "curve_drawer" in fields:
            warnings.warn(
                "The option 'curve_drawer' is replaced with 'plotter'. "
                "This option will be removed in Qiskit Experiments 0.6.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Set the plotter drawer to `curve_drawer`. If `curve_drawer` is the right type, set it
            # directly. If not, wrap it in a compatibility drawer.
            if isinstance(fields["curve_drawer"], BaseDrawer):
                plotter = self.options.plotter
                plotter.drawer = fields.pop("curve_drawer")
                fields["plotter"] = plotter
            else:
                drawer = fields["curve_drawer"]
                compat_drawer = LegacyCurveCompatDrawer(drawer)
                plotter = self.options.plotter
                plotter.drawer = compat_drawer
                fields["plotter"] = plotter

        for field in fields:
            if not hasattr(self.options, field):
                warnings.warn(
                    f"Specified option {field} doesn't exist in this analysis instance. "
                    f"Note that {self.__class__.__name__} is a composite curve analysis instance, "
                    "which consists of multiple child curve analyses. "
                    "This options may exist in each analysis instance. "
                    "Please try setting options to child analyses through '.analyses()'.",
                    UserWarning,
                )
        super().set_options(**fields)

    def _run_analysis(
        self,
        experiment_data: ExperimentData,
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:

        analysis_results = []
        figures = []

        fit_dataset = {}
        curve_data_set = []
        for analysis in self._analyses:
            analysis._initialize(experiment_data)
            analysis.set_options(plot=False)

            metadata = analysis.options.extra.copy()
            metadata["group"] = analysis.name

            curve_data = analysis._format_data(
                analysis._run_data_processing(experiment_data.data())
            )
            fit_data = analysis._run_curve_fit(curve_data[curve_data.format == "fit-ready"])
            fit_dataset[analysis.name] = fit_data

            if fit_data.success:
                quality = analysis._evaluate_quality(fit_data)
            else:
                quality = "bad"

            if self.options.return_fit_parameters:
                # Store fit status overview entry regardless of success.
                # This is sometime useful when debugging the fitting code.
                overview = AnalysisResultData(
                    name=PARAMS_ENTRY_PREFIX + analysis.name,
                    value=fit_data,
                    quality=quality,
                    extra=metadata,
                )
                analysis_results.append(overview)

            if fit_data.success:
                # Add fit data to curve data table
                fit_curves = []
                formatted = curve_data[curve_data.format == "fit-ready"]
                for (i, name), sub_data in list(formatted.groupby(["model_id", "model_name"])):
                    xval = sub_data.xval.to_numpy()
                    if len(xval) == 0:
                        # If data is empty, skip drawing this model.
                        # This is the case when fit model exist but no data to fit is provided.
                        continue
                    # Compute X, Y values with fit parameters.
                    xval_fit = np.linspace(np.min(xval), np.max(xval), num=100)
                    yval_fit = eval_with_uncertainties(
                        x=xval_fit,
                        model=analysis.models[i],
                        params=fit_data.ufloat_params,
                    )
                    model_fit = np.full((100, len(curve_data.columns)), np.nan, dtype=object)
                    fit_curves.append(model_fit)
                    # xval
                    model_fit[:, 0] = xval_fit
                    # yval
                    model_fit[:, 1] = unp.nominal_values(yval_fit)
                    # yerr
                    if fit_data.covar is not None:
                        model_fit[:, 2] = unp.std_devs(yval_fit)
                    # model_name
                    model_fit[:, 3] = name
                    # model_id
                    model_fit[:, 4] = i
                    # type
                    model_fit[:, 6] = "fit"
                curve_data = curve_data.append_list_values(
                    other=np.vstack(fit_curves),
                    prefix="fit",
                )
                analysis_results.extend(
                    analysis._create_analysis_results(
                        fit_data=fit_data,
                        quality=quality,
                        **metadata.copy(),
                    )
                )

            if self.options.return_data_points:
                # Add raw data points
                analysis_results.extend(
                    analysis._create_curve_data(
                        curve_data=curve_data[curve_data.format == "fit-ready"],
                        **metadata,
                    )
                )

            curve_data.model_name += f"_{analysis.name}"
            curve_data_set.append(curve_data)

        combined_curve_data = pd.concat(curve_data_set)
        total_quality = self._evaluate_quality(fit_dataset)

        # Create analysis results by combining all fit data
        if all(fit_data.success for fit_data in fit_dataset.values()):
            composite_results = self._create_analysis_results(
                fit_data=fit_dataset, quality=total_quality, **self.options.extra.copy()
            )
            analysis_results.extend(composite_results)
        else:
            composite_results = []

        if self.options.plot:
            self.plotter.set_supplementary_data(
                fit_red_chi={k: v.reduced_chisq for k, v in fit_dataset.items() if v.success},
                primary_results=composite_results,
            )
            figures.extend(self._create_figures(curve_data=combined_curve_data))

        return analysis_results, figures
