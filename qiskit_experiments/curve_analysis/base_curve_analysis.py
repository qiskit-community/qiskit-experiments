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

"""
Base class of curve analysis.
"""

import warnings

from abc import ABC, abstractmethod
from typing import List, Dict, Union

import numpy as np
from uncertainties import unumpy as unp

from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData, Options, ExperimentData
from qiskit_experiments.data_processing import DataProcessor
from qiskit_experiments.data_processing.processor_library import get_processor
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.exceptions import AnalysisError

from .curve_data import CurveData, SeriesDef, FitData, ParameterRepr, FitOptions
from .data_processing import multi_mean_xy_data, data_sort
from .curve_fit import multi_curve_fit
from .visualization import MplCurveDrawer, BaseCurveDrawer

PARAMS_ENTRY_PREFIX = "@Parameters_"
DATA_ENTRY_PREFIX = "@Data_"


class BaseCurveAnalysis(BaseAnalysis, ABC):
    """Abstract superclass of curve analysis base classes.

    Note that this class doesn't define :meth:`_run_analysis` method,
    and no actual fitting protocol is implemented in this base class.
    However, this class defines several common methods that can be reused.
    A curve analysis subclass can construct proper fitting protocol
    by combining following methods, i.e. subroutines.
    See :ref:`curve_analysis_workflow` for how these subroutines are called.

    .. rubric:: _generate_fit_guesses

    This method creates initial guesses for the fit parameters.
    This might be overridden by subclass.
    See :ref:`curve_analysis_init_guess` for details.

    .. rubric:: _format_data

    This method consumes the processed dataset and outputs the formatted dataset.
    By default, this method takes the average of y values over
    the same x values and then sort the entire data by x values.

    .. rubric:: _evaluate_quality

    This method evaluates the quality of the fit based on the fit result.
    This returns "good" when reduced chi-squared is less than 3.0.
    Usually it returns string "good" or "bad" according to the evaluation.
    This criterion can be updated by subclass.

    .. rubric:: _run_data_processing

    This method performs data processing and returns the processed dataset.
    By default, it internally calls :class:`DataProcessor` instance from the analysis options
    and processes experiment data payload to create Y data with uncertainty.
    X data and other metadata are generated within this method by inspecting the
    circuit metadata. The series classification is also performed by based upon the
    matching of circuit metadata and :attr:`SeriesDef.filter_kwargs`.

    .. rubric:: _run_curve_fit

    This method performs the fitting with predefined fit models and the formatted dataset.
    This method internally calls :meth:`_generate_fit_guesses` method.
    Note that this is a core functionality of the :meth:`_run_analysis` method,
    that creates fit result object from the formatted dataset.

    .. rubric:: _create_analysis_results

    This method to creates analysis results for important fit parameters
    that might be defined by analysis options ``result_parameters``.
    In addition, another entry for all fit parameters is created when
    the analysis option ``return_fit_parameters`` is ``True``.

    .. rubric:: _create_curve_data

    This method to creates analysis results for the formatted dataset, i.e. data used for the fitting.
    Entries are created when the analysis option ``return_data_points`` is ``True``.
    If analysis consists of multiple series, analysis result is created for
    each curve data in the series definitions.

    .. rubric:: _initialize

    This method initializes analysis options against input experiment data.
    Usually this method is called before other methods are called.

    """

    @property
    @abstractmethod
    def parameters(self) -> List[str]:
        """Return parameters estimated by this analysis."""

    @property
    def drawer(self) -> BaseCurveDrawer:
        """A short-cut for curve drawer instance."""
        return self._options.curve_drawer

    @classmethod
    def _default_options(cls) -> Options:
        """Return default analysis options.

        Analysis Options:
            curve_drawer (BaseCurveDrawer): A curve drawer instance to visualize
                the analysis result.
            plot_raw_data (bool): Set ``True`` to draw processed data points,
                dataset without formatting, on canvas. This is ``False`` by default.
            plot (bool): Set ``True`` to create figure for fit result.
                This is ``True`` by default.
            return_fit_parameters (bool): Set ``True`` to return all fit model parameters
                with details of the fit outcome. Default to ``True``.
            return_data_points (bool): Set ``True`` to include in the analysis result
                the formatted data points given to the fitter. Default to ``False``.
            data_processor (Callable): A callback function to format experiment data.
                This can be a :class:`~qiskit_experiments.data_processing.DataProcessor`
                instance that defines the `self.__call__` method.
            normalization (bool) : Set ``True`` to normalize y values within range [-1, 1].
                Default to ``False``.
            p0 (Dict[str, float]): Initial guesses for the fit parameters.
                The dictionary is keyed on the fit parameter names.
            bounds (Dict[str, Tuple[float, float]]): Boundary of fit parameters.
                The dictionary is keyed on the fit parameter names and
                values are the tuples of (min, max) of each parameter.
            curve_fitter_options (Dict[str, Any]) Options that are passed to the
                scipy curve fit which performs the least square fitting on the experiment results.
            x_key (str): Circuit metadata key representing a scanned value.
            result_parameters (List[Union[str, ParameterRepr]): Parameters reported in the
                database as a dedicated entry. This is a list of parameter representation
                which is either string or ParameterRepr object. If you provide more
                information other than name, you can specify
                ``[ParameterRepr("alpha", "\u03B1", "a.u.")]`` for example.
                The parameter name should be defined in the series definition.
                Representation should be printable in standard output, i.e. no latex syntax.
            extra (Dict[str, Any]): A dictionary that is appended to all database entries
                as extra information.
            fixed_parameters (Dict[str, Any]): Fitting model parameters that are fixed
                during the curve fitting. This should be provided with default value
                keyed on one of the parameter names in the series definition.
        """
        options = super()._default_options()

        options.curve_drawer = MplCurveDrawer()
        options.plot_raw_data = False
        options.plot = True
        options.return_fit_parameters = True
        options.return_data_points = False
        options.data_processor = None
        options.normalization = False
        options.x_key = "xval"
        options.result_parameters = []
        options.extra = {}
        options.curve_fitter_options = {}
        options.p0 = {}
        options.bounds = {}
        options.fixed_parameters = {}

        # Set automatic validator for particular option values
        options.set_validator(field="data_processor", validator_value=DataProcessor)
        options.set_validator(field="curve_drawer", validator_value=BaseCurveDrawer)

        return options

    def set_options(self, **fields):
        """Set the analysis options for :meth:`run` method.

        Args:
            fields: The fields to update the options

        Raises:
            KeyError: When removed option ``curve_fitter`` is set.
        """
        # TODO remove this in Qiskit Experiments v0.4
        if "curve_plotter" in fields:
            warnings.warn(
                "The analysis option 'curve_plotter' has been deprecated. "
                "The option is replaced with 'curve_drawer' that takes 'MplCurveDrawer' instance. "
                "If this is a loaded analysis, please save this instance again to update option value. "
                "The 'curve_plotter' argument along with this warning will be removed "
                "in Qiskit Experiments 0.4.",
                DeprecationWarning,
                stacklevel=2,
            )
            del fields["curve_plotter"]

        if "curve_fitter" in fields:
            warnings.warn(
                "Setting curve fitter to analysis options has been deprecated and "
                "the option has been removed. The fitter setting is dropped. "
                "Now you can directly override '_run_curve_fit' method to apply custom fitter. "
                "The `curve_fitter` argument along with this warning will be removed "
                "in Qiskit Experiments 0.4.",
                DeprecationWarning,
                stacklevel=2,
            )
            del fields["curve_fitter"]

        # pylint: disable=no-member
        draw_options = set(self.drawer.options.__dict__.keys()) | {"style"}
        deprecated = draw_options & fields.keys()
        if any(deprecated):
            warnings.warn(
                f"Option(s) {deprecated} have been moved to draw_options and will be removed soon. "
                "Use self.drawer.set_options instead. "
                "If this is a loaded analysis, please save this instance again to update option value. "
                "These arguments along with this warning will be removed "
                "in Qiskit Experiments 0.4.",
                DeprecationWarning,
                stacklevel=2,
            )
            draw_options = dict()
            for depopt in deprecated:
                if depopt == "style":
                    for k, v in fields.pop("style").items():
                        draw_options[k] = v
                else:
                    draw_options[depopt] = fields.pop(depopt)
            self.drawer.set_options(**draw_options)

        super().set_options(**fields)

    def _generate_fit_guesses(
        self,
        user_opt: FitOptions,
        curve_data: CurveData,  # pylint: disable=unused-argument
    ) -> Union[FitOptions, List[FitOptions]]:
        """Create algorithmic guess with analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        return user_opt

    def _format_data(
        self,
        curve_data: CurveData,
    ) -> CurveData:
        """Postprocessing for the processed dataset.

        Args:
            curve_data: Processed dataset created from experiment results.

        Returns:
            Formatted data.
        """
        # take average over the same x value by keeping sigma
        data_allocation, xdata, ydata, sigma, shots = multi_mean_xy_data(
            series=curve_data.data_allocation,
            xdata=curve_data.x,
            ydata=curve_data.y,
            sigma=curve_data.y_err,
            shots=curve_data.shots,
            method="shots_weighted",
        )

        # sort by x value in ascending order
        data_allocation, xdata, ydata, sigma, shots = data_sort(
            series=data_allocation,
            xdata=xdata,
            ydata=ydata,
            sigma=sigma,
            shots=shots,
        )

        return CurveData(
            x=xdata,
            y=ydata,
            y_err=sigma,
            shots=shots,
            data_allocation=data_allocation,
            labels=curve_data.labels,
        )

    def _evaluate_quality(
        self,
        fit_data: FitData,
    ) -> Union[str, None]:
        """Evaluate quality of the fit result.

        Args:
            fit_data: Fit outcome.

        Returns:
            String that represents fit result quality. Usually "good" or "bad".
        """
        if fit_data.reduced_chisq < 3.0:
            return "good"
        return "bad"

    def _run_data_processing(
        self,
        raw_data: List[Dict],
        series: List[SeriesDef],
    ) -> CurveData:
        """Perform data processing from the experiment result payload.

        Args:
            raw_data: Payload in the experiment data.
            series: List of series definition defining filtering condition.

        Returns:
            Processed data that will be sent to the formatter method.

        Raises:
            DataProcessorError: When key for x values is not found in the metadata.
        """
        x_key = self.options.x_key

        try:
            xdata = np.asarray([datum["metadata"][x_key] for datum in raw_data], dtype=float)
        except KeyError as ex:
            raise DataProcessorError(
                f"X value key {x_key} is not defined in circuit metadata."
            ) from ex

        ydata = self.options.data_processor(raw_data)
        shots = np.asarray([datum.get("shots", np.nan) for datum in raw_data])

        def _matched(metadata, **filters):
            try:
                return all(metadata[key] == val for key, val in filters.items())
            except KeyError:
                return False

        data_allocation = np.full(xdata.size, -1, dtype=int)
        for sind, series_def in enumerate(series):
            matched_inds = np.asarray(
                [_matched(d["metadata"], **series_def.filter_kwargs) for d in raw_data], dtype=bool
            )
            data_allocation[matched_inds] = sind

        return CurveData(
            x=xdata,
            y=unp.nominal_values(ydata),
            y_err=unp.std_devs(ydata),
            shots=shots,
            data_allocation=data_allocation,
            labels=[s.name for s in series],
        )

    def _run_curve_fit(
        self,
        curve_data: CurveData,
        series: List[SeriesDef],
    ) -> Union[None, FitData]:
        """Perform curve fitting on given data collection and fit models.

        Args:
            curve_data: Formatted data to fit.
            series: A list of fit models.

        Returns:
            The best fitting outcome with minimum reduced chi-squared value.
        """
        # Create a list of initial guess
        default_fit_opt = FitOptions(
            parameters=self.parameters,
            default_p0=self.options.p0,
            default_bounds=self.options.bounds,
            **self.options.curve_fitter_options,
        )
        try:
            fit_options = self._generate_fit_guesses(default_fit_opt, curve_data)
        except TypeError:
            warnings.warn(
                "Calling '_generate_fit_guesses' method without curve data has been "
                "deprecated and will be prohibited after 0.4. "
                "Update the method signature of your custom analysis class.",
                DeprecationWarning,
            )
            # pylint: disable=no-value-for-parameter
            fit_options = self._generate_fit_guesses(default_fit_opt)
        if isinstance(fit_options, FitOptions):
            fit_options = [fit_options]

        # Run fit for each configuration
        fit_results = []
        for fit_opt in set(fit_options):
            try:
                fit_result = multi_curve_fit(
                    funcs=[sdef.fit_func for sdef in series],
                    series=curve_data.data_allocation,
                    xdata=curve_data.x,
                    ydata=curve_data.y,
                    sigma=curve_data.y_err,
                    **fit_opt.options,
                )
                fit_results.append(fit_result)
            except AnalysisError:
                # Some guesses might be too far from the true parameters and may thus fail.
                # We ignore initial guesses that fail and continue with the next fit candidate.
                pass

        # Find best value with chi-squared value
        if len(fit_results) == 0:
            warnings.warn(
                "All initial guesses and parameter boundaries failed to fit the data. "
                "Please provide better initial guesses or fit parameter boundaries.",
                UserWarning,
            )
            # at least return raw data points rather than terminating
            return None

        return sorted(fit_results, key=lambda r: r.reduced_chisq)[0]

    def _create_analysis_results(
        self,
        fit_data: FitData,
        quality: str,
        **metadata,
    ) -> List[AnalysisResultData]:
        """Create analysis results for important fit parameters.

        Args:
            fit_data: Fit outcome.
            quality: Quality of fit outcome.

        Returns:
            List of analysis result data.
        """
        outcomes = []

        # Create entry for all fit parameters
        if self.options.return_fit_parameters:
            fit_parameters = AnalysisResultData(
                name=PARAMS_ENTRY_PREFIX + self.__class__.__name__,
                value=[p.nominal_value for p in fit_data.popt],
                chisq=fit_data.reduced_chisq,
                quality=quality,
                extra={
                    "popt_keys": fit_data.popt_keys,
                    "dof": fit_data.dof,
                    "covariance_mat": fit_data.pcov,
                    **metadata,
                },
            )
            outcomes.append(fit_parameters)

        # Create entries for important parameters
        for param_repr in self.options.result_parameters:
            if isinstance(param_repr, ParameterRepr):
                p_name = param_repr.name
                p_repr = param_repr.repr or param_repr.name
                unit = param_repr.unit
            else:
                p_name = param_repr
                p_repr = param_repr
                unit = None

            fit_val = fit_data.fitval(p_name)
            if unit:
                par_metadata = metadata.copy()
                par_metadata["unit"] = unit
            else:
                par_metadata = metadata

            outcome = AnalysisResultData(
                name=p_repr,
                value=fit_val,
                chisq=fit_data.reduced_chisq,
                quality=quality,
                extra=par_metadata,
            )
            outcomes.append(outcome)

        return outcomes

    def _create_curve_data(
        self,
        curve_data: CurveData,
        series: List[SeriesDef],
        **metadata,
    ) -> List[AnalysisResultData]:
        """Create analysis results for raw curve data.

        Args:
            curve_data: Formatted data that is used for the fitting.
            series: List of series definition associated with the curve data.

        Returns:
            List of analysis result data.
        """
        samples = []

        if not self.options.return_data_points:
            return samples

        for sdef in series:
            s_data = curve_data.get_subset_of(sdef.name)
            raw_datum = AnalysisResultData(
                name=DATA_ENTRY_PREFIX + self.__class__.__name__,
                value={
                    "xdata": s_data.x,
                    "ydata": s_data.y,
                    "sigma": s_data.y_err,
                },
                extra={
                    "name": sdef.name,
                    **metadata,
                },
            )
            samples.append(raw_datum)

        return samples

    def _initialize(
        self,
        experiment_data: ExperimentData,
    ):
        """Initialize curve analysis with experiment data.

        This method is called ahead of other processing.

        Args:
            experiment_data: Experiment data to analyze.
        """
        # Initialize canvas
        if self.options.plot:
            self.drawer.initialize_canvas()

        # Initialize data processor
        # TODO move this to base analysis in follow-up
        data_processor = self.options.data_processor or get_processor(experiment_data, self.options)

        if not data_processor.is_trained:
            data_processor.train(data=experiment_data.data())
        self.set_options(data_processor=data_processor)
