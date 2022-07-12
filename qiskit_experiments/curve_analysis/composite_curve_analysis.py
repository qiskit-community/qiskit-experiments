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
from typing import Dict, List, Tuple, Optional, Union

import lmfit
import numpy as np
from uncertainties import unumpy as unp, UFloat

from qiskit_experiments.framework import BaseAnalysis, ExperimentData, AnalysisResultData, Options
from .base_curve_analysis import BaseCurveAnalysis, PARAMS_ENTRY_PREFIX
from .curve_data import CurveFitResult
from .utils import analysis_result_to_repr, eval_with_uncertainties
from .visualization import MplCurveDrawer, BaseCurveDrawer


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
    def drawer(self) -> BaseCurveDrawer:
        """A short-cut for curve drawer instance."""
        return self._options.curve_drawer

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

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options.

        Analysis Options:
            curve_drawer (BaseCurveDrawer): A curve drawer instance to visualize
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
            curve_drawer=MplCurveDrawer(),
            plot=True,
            return_fit_parameters=True,
            return_data_points=False,
            extra={},
        )

        # Set automatic validator for particular option values
        options.set_validator(field="curve_drawer", validator_value=BaseCurveDrawer)

        return options

    def set_options(self, **fields):
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

        # Initialize canvas
        if self.options.plot:
            self.drawer.initialize_canvas()

        fit_dataset = {}
        for analysis in self._analyses:
            analysis._initialize(experiment_data)

            metadata = analysis.options.extra.copy()
            metadata["group"] = analysis.name

            processed_data = analysis._run_data_processing(
                raw_data=experiment_data.data(),
                models=analysis.models,
            )

            if self.options.plot and analysis.options.plot_raw_data:
                for model in analysis.models:
                    sub_data = processed_data.get_subset_of(model._name)
                    self.drawer.draw_raw_data(
                        x_data=sub_data.x,
                        y_data=sub_data.y,
                        name=model._name + f"_{analysis.name}",
                    )

            # Format data
            formatted_data = analysis._format_data(processed_data)
            if self.options.plot:
                for model in analysis.models:
                    sub_data = formatted_data.get_subset_of(model._name)
                    self.drawer.draw_formatted_data(
                        x_data=sub_data.x,
                        y_data=sub_data.y,
                        y_err_data=sub_data.y_err,
                        name=model._name + f"_{analysis.name}",
                    )

            # Run fitting
            fit_data = analysis._run_curve_fit(
                curve_data=formatted_data,
                models=analysis.models,
            )

            if fit_data.success:
                quality = analysis._evaluate_quality(fit_data)
            else:
                quality = "bad"

            if self.options.return_fit_parameters:
                overview = AnalysisResultData(
                    name=PARAMS_ENTRY_PREFIX + analysis.name,
                    value=fit_data,
                    quality=quality,
                    extra=metadata,
                )
                analysis_results.append(overview)

            if fit_data.success:
                # Add extra analysis results
                analysis_results.extend(
                    analysis._create_analysis_results(
                        fit_data=fit_data, quality=quality, **metadata.copy()
                    )
                )

                # Draw fit result
                if self.options.plot:
                    interp_x = np.linspace(
                        np.min(formatted_data.x), np.max(formatted_data.x), num=100
                    )
                    for model in analysis.models:
                        y_data_with_uncertainty = eval_with_uncertainties(
                            x=interp_x,
                            model=model,
                            params=fit_data.ufloat_params,
                        )
                        y_mean = unp.nominal_values(y_data_with_uncertainty)
                        # Draw fit line
                        self.drawer.draw_fit_line(
                            x_data=interp_x,
                            y_data=y_mean,
                            name=model._name + f"_{analysis.name}",
                        )
                        if fit_data.covar is not None:
                            # Draw confidence intervals with different n_sigma
                            sigmas = unp.std_devs(y_data_with_uncertainty)
                            if np.isfinite(sigmas).all():
                                for n_sigma, alpha in self.drawer.options.plot_sigma:
                                    self.drawer.draw_confidence_interval(
                                        x_data=interp_x,
                                        y_ub=y_mean + n_sigma * sigmas,
                                        y_lb=y_mean - n_sigma * sigmas,
                                        name=model._name + f"_{analysis.name}",
                                        alpha=alpha,
                                    )

            # Add raw data points
            if self.options.return_data_points:
                analysis_results.extend(
                    analysis._create_curve_data(curve_data=formatted_data, models=analysis.models)
                )

            fit_dataset[analysis.name] = fit_data

        total_quality = self._evaluate_quality(fit_dataset)

        # Create analysis results by combining all fit data
        analysis_results.extend(
            self._create_analysis_results(
                fit_data=fit_dataset, quality=total_quality, **self.options.extra.copy()
            )
        )

        if self.options.plot:
            # Write fitting report
            report = ""
            for res in analysis_results:
                if isinstance(res.value, (float, UFloat)):
                    report += f"{analysis_result_to_repr(res)}\n"
            chisqs = []
            for group, fit_data in fit_dataset.items():
                chisqs.append(r"reduced-$\chi^2$ = " + f"{fit_data.reduced_chisq: .4g} ({group})")
            report += "\n".join(chisqs)
            self.drawer.draw_fit_report(description=report)

            # Finalize canvas
            self.drawer.format_canvas()
            return analysis_results, [self.drawer.figure]

        return analysis_results, []
