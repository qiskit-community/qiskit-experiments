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

import lmfit

from qiskit_experiments.data_processing import DataProcessor
from qiskit_experiments.data_processing.processor_library import get_processor
from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData, Options, ExperimentData
from .curve_data import CurveData, ParameterRepr, CurveFitResult
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


    Subclass must implement following methods.

    .. rubric:: _run_data_processing

    This method performs data processing and returns the processed dataset.
    Input data is a list of dictionaries, where each entry represents an outcome
    of circuit sampling along with the metadata attached to it.

    .. rubric:: _format_data

    This method consumes the processed dataset and outputs the formatted dataset.
    For example, this may include averaging Y values over the same X data points.

    .. rubric:: _run_curve_fit

    This method performs the fitting with the predefined fit models and the formatted dataset.
    This is a core functionality of the :meth:`_run_analysis` method
    that creates fit result objects from the formatted dataset.

    Optionally, a subclass may override following methods.
    These methods have default implementations as described below.


    .. rubric:: _evaluate_quality

    This method evaluates the quality of the fit based on the fit result.
    This returns "good" when reduced chi-squared is less than 3.0.
    Usually it returns string "good" or "bad" according to the evaluation.
    This criterion can be updated by subclass.

    .. rubric:: _run_curve_fit

    This method performs the fitting with predefined fit models and the formatted dataset.
    This method internally calls :meth:`_generate_fit_guesses` method.
    Note that this is a core functionality of the :meth:`_run_analysis` method,
    that creates fit result object from the formatted dataset.

    .. rubric:: _create_analysis_results

    This method creates analysis results for important fit parameters
    that might be defined by analysis options ``result_parameters``.

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
    @abstractmethod
    def name(self) -> str:
        """Return name of this analysis."""

    @property
    @abstractmethod
    def models(self) -> List[lmfit.Model]:
        """Return fit models."""

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
            fit_method (str): Fit method that LMFIT minimizer uses.
                Default to ``least_squares`` method which implements the
                Trust Region Reflective algorithm to solve the minimization problem.
                See LMFIT documentation for available options.
            lmfit_options (Dict[str, Any]): Options that are passed to the
                LMFIT minimizer. Acceptable options depend on fit_method.
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
            filter_data (Dict[str, Any]): Dictionary of experiment data metadata to filter.
                Experiment outcomes with metadata that matches with this dictionary
                are used in the analysis. If not specified, all experiment data are
                input to the curve fitter. By default no filtering condition is set.
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
        options.fit_method = "least_squares"
        options.lmfit_options = {}
        options.p0 = {}
        options.bounds = {}
        options.fixed_parameters = {}
        options.filter_data = {}

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
        # TODO remove this in Qiskit Experiments v0.5

        if "curve_fitter_options" in fields:
            warnings.warn(
                "The option 'curve_fitter_options' is replaced with 'lmfit_options.' "
                "This option will be removed in Qiskit Experiments 0.5.",
                DeprecationWarning,
                stacklevel=2,
            )
            fields["lmfit_options"] = fields.pop("curve_fitter_options")

        super().set_options(**fields)

    @abstractmethod
    def _run_data_processing(
        self,
        raw_data: List[Dict],
        models: List[lmfit.Model],
    ) -> CurveData:
        """Perform data processing from the experiment result payload.

        Args:
            raw_data: Payload in the experiment data.
            models: A list of LMFIT models that provide the model name and
                optionally data sorting keys.

        Returns:
            Processed data that will be sent to the formatter method.

        Raises:
            DataProcessorError: When model is multi-objective function but
                data sorting option is not provided.
            DataProcessorError: When key for x values is not found in the metadata.
        """

    @abstractmethod
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

    @abstractmethod
    def _run_curve_fit(
        self,
        curve_data: CurveData,
        models: List[lmfit.Model],
    ) -> CurveFitResult:
        """Perform curve fitting on given data collection and fit models.

        Args:
            curve_data: Formatted data to fit.
            models: A list of LMFIT models that are used to build a cost function
                for the LMFIT minimizer.

        Returns:
            The best fitting outcome with minimum reduced chi-squared value.
        """

    def _evaluate_quality(
        self,
        fit_data: CurveFitResult,
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

    def _create_analysis_results(
        self,
        fit_data: CurveFitResult,
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

            if unit:
                par_metadata = metadata.copy()
                par_metadata["unit"] = unit
            else:
                par_metadata = metadata

            outcome = AnalysisResultData(
                name=p_repr,
                value=fit_data.ufloat_params[p_name],
                chisq=fit_data.reduced_chisq,
                quality=quality,
                extra=par_metadata,
            )
            outcomes.append(outcome)

        return outcomes

    def _create_curve_data(
        self,
        curve_data: CurveData,
        models: List[lmfit.Model],
        **metadata,
    ) -> List[AnalysisResultData]:
        """Create analysis results for raw curve data.

        Args:
            curve_data: Formatted data that is used for the fitting.
            models: A list of LMFIT models that provides model names
                to extract subsets of experiment data.

        Returns:
            List of analysis result data.
        """
        samples = []

        for model in models:
            sub_data = curve_data.get_subset_of(model._name)
            raw_datum = AnalysisResultData(
                name=DATA_ENTRY_PREFIX + self.__class__.__name__,
                value={
                    "xdata": sub_data.x,
                    "ydata": sub_data.y,
                    "sigma": sub_data.y_err,
                },
                extra={
                    "name": model._name,
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
        # Initialize data processor
        # TODO move this to base analysis in follow-up
        data_processor = self.options.data_processor or get_processor(experiment_data, self.options)

        if not data_processor.is_trained:
            data_processor.train(data=experiment_data.data())
        self.set_options(data_processor=data_processor)
