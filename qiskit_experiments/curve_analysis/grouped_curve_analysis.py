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

from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import lmfit
import numpy as np
from uncertainties import unumpy as unp, UFloat

from qiskit_experiments.framework import ExperimentData, AnalysisResultData, Options
from qiskit_experiments.exceptions import AnalysisError

from .base_curve_analysis import BaseCurveAnalysis, PARAMS_ENTRY_PREFIX
from .curve_data import CurveFitResult
from .utils import analysis_result_to_repr, eval_with_uncertainties


class MultiGroupCurveAnalysis(BaseCurveAnalysis):
    r"""Curve analysis with multiple independent groups.

    The :class:`.MultiGroupCurveAnalysis` provides a fitting framework for the case

    .. math::

        \Theta_{i, \mbox{opt}} = \arg\min_{\Theta_{i, \mbox{fit}}} (F(X_i, \Theta_i)-Y_i)^2,

    where :math:`F` is a common fit model for multiple independent dataset :math:`(X_i, Y_i)`.
    The fit generates multiple fit outcomes :math:`\Theta_{i, \mbox{opt}}` for each dataset.

    Each experiment circuit must have metadata of data sorting keys for model mapping and
    group mapping. For example, if the experiment has two experiment conditions "A" and "B"
    while fitting the outcomes with three models "X", "Y", "Z", then the metadata must consist
    of the condition and model, e.g. ``{"condition": "A", "model": "X"}``.

    In this example, "condition" should be defined in the ``group_data_sort_key``
    of the :class:`.MultiGroupCurveAnalysis`, and "model" should be defined in
    the ``data_sort_key`` for each LMFIT model.

    .. code-block:: python

        import lmfit
        from qiskit_experiments.curve_analysis import MultiGroupCurveAnalysis

        analysis = MultiGroupCurveAnalysis(
            groups=["groupA", "groupB"],
            group_data_sort_key=[{"condition": "A"}, {"condition": "B"}],
            models=[
                lmfit.Model(fit_func_x, name="curve_X", data_sort_key={"model": "X"}),
                lmfit.Model(fit_func_y, name="curve_Y", data_sort_key={"model": "Y"}),
                lmfit.Model(fit_func_z, name="curve_Z", data_sort_key={"model": "Z"}),
            ]
        )

    In above setting, each fit curve will appear in the output figure with unique name
    in the form of ``{group name}_{model name}``, e.g. ``groupA_curve_X``.

    A subclass can implement the following extra method.

    .. rubric:: _create_composite_analysis_results

    This method computes new quantities by taking all fit outcomes and
    store the new values in the analysis results.
    By default, no extra quantity is computed.

    """

    def __init__(
        self,
        groups: List[str],
        group_data_sort_key: List[Dict[str, Any]],
        models: Optional[List[lmfit.Model]] = None,
    ):
        """Create new analysis.

        Args:
            groups: List of group names.
            group_data_sort_key: List of sort key dictionary for each group.
            models: List of LMFIT ``Model`` class to define fitting functions and
                parameters. Provided models are shared among groups.

        Raises:
            AnalysisError: When groups and group_data_sort_key don't match.
        """
        super().__init__()

        if len(groups) != len(group_data_sort_key):
            raise AnalysisError(
                "Number of groups and data sort key doesn't match. "
                "Data sort key must be provided for each group."
            )

        self._groups = groups
        self._group_data_sort_key = group_data_sort_key
        self._models = models

    @classmethod
    def _default_options(cls) -> Options:
        """Default analysis options.

        Analysis Options:
            p0 (Union[List[Dict[str, float]], Dict[str, float]]): A dictionary of initial guesses
                for the fit parameters keyed on the fit parameter names.
                This can be provided for either each fit group or entire groups.
                To provide initial guesses for each group, a list of dictionary must be
                set here. The list ordering corresponds to the groups.
        """
        return super()._default_options()

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

    # pylint: disable=unused-argument
    def _create_composite_analysis_results(
        self,
        fit_dataset: List[CurveFitResult],
        quality: str,
        **metadata,
    ) -> List[AnalysisResultData]:
        """Create new analysis data from combination of all fit outcomes.

        Args:
            fit_dataset: Fit outcomes. The ordering of data corresponds to the
                groups defined in the ``self._groups``.
            quality: Overall quality of fittings.

        Returns:
            List of analysis results.
        """
        return []

    def _run_analysis(
        self,
        experiment_data: ExperimentData,
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:

        self._initialize(experiment_data)
        analysis_results = []

        # Extract experiment data for this group
        # TODO use dataframe
        group_data = defaultdict(list)
        for datum in experiment_data.data():
            for group, dkey in zip(self._groups, self._group_data_sort_key):
                if all(datum["metadata"].get(k, None) == v for k, v in dkey.items()):
                    group_data[group].append(datum)
                    break

        fit_qualities = []
        fit_dataset = []
        for group in self._groups:
            metadata = self.options.extra.copy()
            metadata["group"] = group

            # Run data processing
            processed_data = self._run_data_processing(
                raw_data=group_data[group],
                models=self._models,
            )

            if self.options.plot and self.options.plot_raw_data:
                for model in self._models:
                    sub_data = processed_data.get_subset_of(model._name)
                    self.drawer.draw_raw_data(
                        x_data=sub_data.x,
                        y_data=sub_data.y,
                        name=f"{model._name}_{group}",
                    )

            # Format data
            formatted_data = self._format_data(processed_data)
            if self.options.plot:
                for model in self._models:
                    sub_data = formatted_data.get_subset_of(model._name)
                    self.drawer.draw_formatted_data(
                        x_data=sub_data.x,
                        y_data=sub_data.y,
                        y_err_data=sub_data.y_err,
                        name=f"{model._name}_{group}",
                    )

            # Run fitting
            if isinstance(self.options.p0, dict):
                initial_guesses = self.options.p0
            else:
                initial_guesses = self.options.p0[self._groups.index(group)]

            fit_data = self._run_curve_fit(
                curve_data=formatted_data,
                models=self._models,
                init_guesses=initial_guesses,
            )

            if fit_data.success:
                quality = self._evaluate_quality(fit_data)
            else:
                quality = "bad"

            if self.options.return_fit_parameters:
                overview = AnalysisResultData(
                    name=f"{PARAMS_ENTRY_PREFIX}{self.__class__.__name__}_{group}",
                    value=fit_data,
                    quality=quality,
                    extra=metadata,
                )
                analysis_results.append(overview)

            if fit_data.success:
                # Add extra analysis results
                analysis_results.extend(
                    self._create_analysis_results(
                        fit_data=fit_data, quality=quality, **metadata.copy()
                    )
                )

                # Draw fit result
                if self.options.plot:
                    interp_x = np.linspace(
                        np.min(formatted_data.x), np.max(formatted_data.x), num=100
                    )
                    for model in self._models:
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
                            name=f"{model._name}_{group}",
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
                                        name=f"{model._name}_{group}",
                                        alpha=alpha,
                                    )

            # Add raw data points
            analysis_results.extend(
                self._create_curve_data(curve_data=formatted_data, models=self._models)
            )

            fit_qualities.append(quality)
            fit_dataset.append(fit_data)

        quality = "good" if all(q == "good" for q in fit_qualities) else "bad"

        # Create analysis results by combining all fit data
        analysis_results.extend(
            self._create_composite_analysis_results(
                fit_dataset=fit_dataset, quality=quality, **self.options.extra.copy()
            )
        )

        if self.options.plot:
            # Write fitting report
            report = ""
            for res in analysis_results:
                if isinstance(res.value, (float, UFloat)):
                    report += f"{analysis_result_to_repr(res)}\n"
            chisqs = []
            for group, fit_data in zip(self._groups, fit_dataset):
                chisqs.append(r"reduced-$\chi^2$ = " + f"{fit_data.reduced_chisq: .4g} ({group})")
            report += "\n".join(chisqs)
            self.drawer.draw_fit_report(description=report)

        # Finalize plot
        if self.options.plot:
            self.drawer.format_canvas()
            return analysis_results, [self.drawer.figure]

        return analysis_results, []

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
