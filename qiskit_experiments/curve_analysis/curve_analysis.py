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

import copy
import itertools
import warnings
from abc import ABC
from typing import Any, Dict, List, Tuple, Callable, Union, Optional, Iterator

import numpy as np
import scipy.optimize as opt
import uncertainties
from uncertainties import unumpy as unp

from qiskit.providers import Backend
from qiskit_experiments.curve_analysis.curve_data import (
    CurveData,
    CompositeFitFunction,
    FitData,
    ParameterRepr,
    FitOptions,
)
from qiskit_experiments.curve_analysis.data_processing import multi_mean_xy_data, data_sort
from qiskit_experiments.curve_analysis.visualization import FitResultPlotters, PlotterStyle
from qiskit_experiments.data_processing import DataProcessor
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.data_processing.processor_library import get_processor
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import (
    BaseAnalysis,
    ExperimentData,
    AnalysisResultData,
    Options,
)

PARAMS_ENTRY_PREFIX = "@Parameters_"
DATA_ENTRY_PREFIX = "@Data_"


class CurveAnalysis(BaseAnalysis, ABC):
    """A base class for curve fit type analysis.

    The subclasses can override class attributes to define the behavior of
    data extraction and fitting. This docstring describes how code developers can
    create a new curve fit analysis subclass inheriting from this base class.
    See :mod:`qiskit_experiments.curve_analysis` for the developer note.

    Class Attributes:
        - ``__series__``: A list of :class:`SeriesDef` defining a fitting model
            for every experimental data set. The attribute cannot be overridden.
            Subclass of curve analysis must define this attribute.

        - ``__fixed_parameters__``: A list of parameter names fixed during the fitting.
            These parameters should be provided via the analysis options.
    """

    #: List[SeriesDef]: List of mapping representing a data series
    __series__ = list()

    #: List[str]: Fixed parameter in fit function. Value should be set to the analysis options.
    __fixed_parameters__ = list()

    # Automatically generated fitting functions of child class
    _composite_funcs = None

    # Automatically generated fitting parameters of child class
    _fit_params = None

    def __init_subclass__(cls, **kwargs):
        """Parse series definition of subclass and set fit function and signature."""

        super().__init_subclass__(**kwargs)

        # Validate if all fixed parameter names are defined in the fit model
        if cls.__fixed_parameters__:
            all_params = set(itertools.chain.from_iterable(s.signature for s in cls.__series__))
            if any(p not in all_params for p in cls.__fixed_parameters__):
                raise AnalysisError("Not existing parameter is fixed.")

        # Parse series information and generate function and signature
        fit_groups = dict()
        for series in cls.__series__:
            if series.group not in fit_groups:
                fit_groups[series.group] = {
                    "fit_functions": [series.fit_func],
                    "signatures": [series.signature],
                    "models": [series.model_description],
                }
            else:
                fit_groups[series.group]["fit_functions"].append(series.fit_func)
                fit_groups[series.group]["signatures"].append(series.signature)
                fit_groups[series.group]["models"].append(series.model_description)

        composite_funcs = [
            CompositeFitFunction(
                **config,
                fixed_parameters=cls.__fixed_parameters__,
                group=group,
            )
            for group, config in fit_groups.items()
        ]

        # Dictionary of fit functions to each group
        cls._composite_funcs = composite_funcs

        # All fit parameter names that this analysis manages
        # Let's keep order of parameters rather than using set, though code is bit messy.
        # It is better to match composite function signature with the func in series definition.
        fit_args = []
        for func in composite_funcs:
            for param in func.signature:
                if param not in fit_args:
                    fit_args.append(param)
        cls._fit_params = fit_args

    def __init__(self):
        """Initialize data fields that are privately accessed by methods."""
        super().__init__()

        #: Dict[str, Any]: Experiment metadata
        self.__experiment_metadata = None

        #: List[CurveData]: Processed experiment data set.
        self.__processed_data_set = list()

        #: Backend: backend object used for experimentation
        self.__backend = None

    @staticmethod
    def curve_fit(
        func: CompositeFitFunction,
        xdata: np.ndarray,
        ydata: np.ndarray,
        sigma: np.ndarray,
        p0: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]],
        **kwargs,
    ) -> FitData:
        """Perform curve fitting.

        This is the scipy curve fit wrapper to manage named fit parameters and
        return outcomes as ufloat objects with parameter correlation computed based on the
        covariance matrix from the fitting. Result is returned as
        :class:`~qiskit_experiments.curve_analysis.FitData` which is a special data container
        for curve analysis. This method can perform multi-objective optimization with
        multiple data series with related fit models.

        Args:
            func: A fit function that can consist of multiple data series.
            xdata: Numpy array representing X values.
            ydata: Numpy array representing Y values.
            sigma: Numpy array representing standard error of Y values.
            p0: Dictionary of initial guesses for given fit function.
            bounds: Dictionary of parameter boundary for given fit function.
            **kwargs: Solver options.

        Returns:
            Fit result.

        Raises:
            AnalysisError: When invalid fit function is provided.
            AnalysisError: When number of data points is too small.
            AnalysisError: When curve fitting does not converge.
        """
        if not isinstance(func, CompositeFitFunction):
            raise AnalysisError(
                "CurveAnalysis subclass requires CompositeFitFunction instance to perform fitting. "
                "Standard callback function is not acceptable due to missing signature metadata."
            )

        lower = [bounds[p][0] for p in func.signature]
        upper = [bounds[p][1] for p in func.signature]
        scipy_bounds = (lower, upper)
        scipy_p0 = list(p0.values())

        dof = len(ydata) - len(func.signature)
        if dof < 1:
            raise AnalysisError(
                "The number of degrees of freedom of the fit data and model "
                " (len(ydata) - len(p0)) is less than 1"
            )

        if np.any(np.nan_to_num(sigma) == 0):
            # Sigma = 0 causes zero division error
            sigma = None
        else:
            if "absolute_sigma" not in kwargs:
                kwargs["absolute_sigma"] = True

        try:
            # pylint: disable = unbalanced-tuple-unpacking
            popt, pcov = opt.curve_fit(
                func,
                xdata,
                ydata,
                sigma=sigma,
                p0=scipy_p0,
                bounds=scipy_bounds,
                **kwargs,
            )
        except Exception as ex:
            raise AnalysisError(
                "scipy.optimize.curve_fit failed with error: {}".format(str(ex))
            ) from ex

        # Compute outcome with errors correlation
        if np.isfinite(pcov).all():
            # Keep parameter correlations in following analysis steps
            fit_params = uncertainties.correlated_values(nom_values=popt, covariance_mat=pcov)
        else:
            # Ignore correlations, add standard error if finite.
            fit_params = [
                uncertainties.ufloat(nominal_value=n, std_dev=s if np.isfinite(s) else np.nan)
                for n, s in zip(popt, np.sqrt(np.diag(pcov)))
            ]

        # Calculate the reduced chi-squared for fit
        yfits = func(xdata, *popt)
        residues = (yfits - ydata) ** 2
        if sigma is not None:
            residues = residues / (sigma**2)
        reduced_chisq = np.sum(residues) / dof

        # Compute data range for fit
        xdata_range = np.min(xdata), np.max(xdata)
        ydata_range = np.min(ydata), np.max(ydata)

        fit_model_descriptions = func.metadata.get("models", [])
        if all(desc for desc in fit_model_descriptions):
            fit_model_repr = ",".join(fit_model_descriptions)
        else:
            fit_model_repr = "not defined"

        return FitData(
            popt=list(fit_params),
            popt_keys=func.signature,
            pcov=pcov,
            reduced_chisq=reduced_chisq,
            dof=dof,
            x_range=xdata_range,
            y_range=ydata_range,
            fit_mdoel=fit_model_repr,
            group=func.group,
        )

    @property
    def composite_funcs(self) -> Iterator[CompositeFitFunction]:
        """Return parsed composite fit function for this analysis instance."""
        for fit_func in self._composite_funcs:
            # Return copy of the composite fit function
            # Note that this is a statefull class attribute, which can be modified during the fit.
            # This may cause unexpected behavior in multithread execution, i.e. composite analysis.
            yield fit_func.copy()

    @property
    def fit_params(self) -> List[str]:
        """Return parameters of this curve analysis."""
        return self._fit_params.copy()

    @classmethod
    def _default_options(cls) -> Options:
        """Return default analysis options.

        Analysis Options:
            data_processor (Callable): A callback function to format experiment data.
                This can be a :class:`~qiskit_experiments.data_processing.DataProcessor`
                instance that defines the `self.__call__` method.
            normalization (bool) : Set ``True`` to normalize y values within range [-1, 1].
            p0 (Dict[str, float]): Array-like or dictionary
                of initial parameters.
            bounds (Dict[str, Tuple[float, float]]): Array-like or dictionary
                of (min, max) tuple of fit parameter boundaries.
            x_key (str): Circuit metadata key representing a scanned value.
            plot (bool): Set ``True`` to create figure for fit result.
            axis (AxesSubplot): Optional. A matplotlib axis object to draw.
            xlabel (str): X label of fit result figure.
            ylabel (str): Y label of fit result figure.
            xlim (Tuple[float, float]): Min and max value of horizontal axis of the fit plot.
            ylim (Tuple[float, float]): Min and max value of vertical axis of the fit plot.
            xval_unit (str): SI unit of x values. No prefix is needed here.
                For example, when the x values represent time, this option will be just "s"
                rather than "ms". In the fit result plot, the prefix is automatically selected
                based on the maximum value. If your x values are in [1e-3, 1e-4], they
                are displayed as [1 ms, 10 ms]. This option is likely provided by the
                analysis class rather than end-users. However, users can still override
                if they need different unit notation. By default, this option is set to ``None``,
                and no scaling is applied. X axis will be displayed in the scientific notation.
            yval_unit (str): Unit of y values. Same as ``xval_unit``.
                This value is not provided in most experiments, because y value is usually
                population or expectation values.
            result_parameters (List[Union[str, ParameterRepr]): Parameters reported in the
                database as a dedicated entry. This is a list of parameter representation
                which is either string or ParameterRepr object. If you provide more
                information other than name, you can specify
                ``[ParameterRepr("alpha", "\u03B1", "a.u.")]`` for example.
                The parameter name should be defined in the series definition.
                Representation should be printable in standard output, i.e. no latex syntax.
            return_data_points (bool): Set ``True`` to return formatted XY data.
            curve_plotter (str): A name of plotter function used to generate
                the curve fit result figure. This refers to the mapper
                :py:class:`~qiskit_experiments.curve_analysis.visualization.FitResultPlotters`
                to retrieve the corresponding callback function.
            style (PlotterStyle): An instance of
                :py:class:`~qiskit_experiments.curve_analysis.visualization.style.PlotterStyle`
                that contains a set of configurations to create a fit plot.
            extra (Dict[str, Any]): A dictionary that is appended to all database entries
                as extra information.
            curve_fitter_options (Dict[str, Any]) Options that are passed to the
                specified curve fitting function.
        """
        options = super()._default_options()

        options.data_processor = None
        options.normalization = False
        options.x_key = "xval"
        options.plot = True
        options.axis = None
        options.xlabel = None
        options.ylabel = None
        options.xlim = None
        options.ylim = None
        options.xval_unit = None
        options.yval_unit = None
        options.result_parameters = None
        options.return_data_points = False
        options.curve_plotter = "mpl_single_canvas"
        options.style = PlotterStyle()
        options.extra = dict()
        options.curve_fitter_options = dict()

        # automatically populate initial guess and boundary
        options.p0 = {par_name: None for par_name in cls._fit_params}
        options.bounds = {par_name: None for par_name in cls._fit_params}

        return options

    def set_options(self, **fields):
        """Set the analysis options for :meth:`run` method.

        Args:
            fields: The fields to update the options

        Raises:
            KeyError: When removed option ``curve_fitter`` is set.
        """
        # TODO remove this in Qiskit Experiments v0.4
        if "curve_fitter" in fields:
            raise KeyError(
                "Option curve_fitter has been removed. Please directly override curve_fit method."
            )

        super().set_options(**fields)

    def _generate_fit_guesses(self, user_opt: FitOptions) -> Union[FitOptions, List[FitOptions]]:
        """Create algorithmic guess with analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.

        Returns:
            List of fit options that are passed to the fitter function.
        """

        return user_opt

    def _format_data(self, data: CurveData) -> CurveData:
        """An optional subroutine to perform data pre-processing.

        Returns:
            Formatted CurveData instance.
        """
        # take average over the same x value by keeping sigma
        series, xdata, ydata, sigma, shots = multi_mean_xy_data(
            series=data.data_index,
            xdata=data.x,
            ydata=data.y,
            sigma=data.y_err,
            shots=data.shots,
            method="shots_weighted",
        )

        # sort by x value in ascending order
        series, xdata, ydata, sigma, shots = data_sort(
            series=series,
            xdata=xdata,
            ydata=ydata,
            sigma=sigma,
            shots=shots,
        )

        return CurveData(
            label="fit_ready",
            x=xdata,
            y=ydata,
            y_err=sigma,
            shots=shots,
            data_index=series,
        )

    # pylint: disable=unused-argument
    def _extra_database_entry(self, fit_data: FitData) -> List[AnalysisResultData]:
        """Calculate new quantity from the fit result.

        Args:
            fit_data: Fit result.

        Returns:
            List of database entry created from the fit data.
        """
        return []

    # pylint: disable=unused-argument
    def _evaluate_quality(self, fit_data: FitData) -> Union[str, None]:
        """Evaluate quality of the fit result.

        Args:
            fit_data: Fit result.

        Returns:
            String that represents fit result quality. Usually "good" or "bad".
        """
        return None

    def _extract_curves(
        self, experiment_data: ExperimentData, data_processor: Union[Callable, DataProcessor]
    ):
        """Extract curve data from experiment data.

        This method internally populates two types of curve data.

        - raw_data:

            This is the data directly obtained from the experiment data.
            You can access this data with ``self._data(label="raw_data")``.

        - fit_ready:

            This is the formatted data created by pre-processing defined by
            `self._format_data()` method. This method is implemented by subclasses.
            You can access to this data with ``self._data(label="fit_ready")``.

        If multiple series exist, you can optionally specify ``series_name`` in
        ``self._data`` method to filter data in the target series.

        .. notes::
            The target metadata properties to define each curve entry is described by
            the class attribute __series__ (see `filter_kwargs`).

        Args:
            experiment_data: ExperimentData object to fit parameters.
            data_processor: A callable or DataProcessor instance to format data into numpy array.
                This should take a list of dictionaries and return two tuple of float values,
                that represent a y value and an error of it.
        Raises:
            DataProcessorError: When `x_key` specified in the analysis option is not
                defined in the circuit metadata.
            AnalysisError: When formatted data has label other than fit_ready.
        """
        self.__processed_data_set = list()

        def _is_target_series(datum, **filters):
            try:
                return all(datum["metadata"][key] == val for key, val in filters.items())
            except KeyError:
                return False

        # Extract X, Y, Y_sigma data
        data = experiment_data.data()

        x_key = self.options.x_key
        try:
            xdata = np.asarray([datum["metadata"][x_key] for datum in data], dtype=float)
        except KeyError as ex:
            raise DataProcessorError(
                f"X value key {x_key} is not defined in circuit metadata."
            ) from ex

        if isinstance(data_processor, DataProcessor):
            ydata = data_processor(data)
        else:
            y_nominals, y_stderrs = zip(*map(data_processor, data))
            ydata = unp.uarray(y_nominals, y_stderrs)

        # Store metadata
        metadata = np.asarray([datum["metadata"] for datum in data], dtype=object)

        # Store shots
        shots = np.asarray([datum.get("shots", np.nan) for datum in data])

        # Find series (invalid data is labeled as -1)
        data_index = np.full(xdata.size, -1, dtype=int)
        for idx, series_def in enumerate(self.__series__):
            data_matched = np.asarray(
                [_is_target_series(datum, **series_def.filter_kwargs) for datum in data], dtype=bool
            )
            data_index[data_matched] = idx

        # Store raw data
        raw_data = CurveData(
            label="raw_data",
            x=xdata,
            y=unp.nominal_values(ydata),
            y_err=unp.std_devs(ydata),
            shots=shots,
            data_index=data_index,
            metadata=metadata,
        )
        self.__processed_data_set.append(raw_data)

        # Format raw data
        formatted_data = self._format_data(raw_data)
        if formatted_data.label != "fit_ready":
            raise AnalysisError(f"Not expected data label {formatted_data.label} != fit_ready.")
        self.__processed_data_set.append(formatted_data)

    @property
    def _experiment_type(self) -> str:
        """Return type of experiment."""
        try:
            return self.__experiment_metadata["experiment_type"]
        except (TypeError, KeyError):
            # Ignore experiment metadata is not set or key is not found
            return None

    @property
    def _num_qubits(self) -> int:
        """Getter for qubit number."""
        try:
            return self.__experiment_metadata["num_qubits"]
        except (TypeError, KeyError):
            # Ignore experiment metadata is not set or key is not found
            return None

    @property
    def _physical_qubits(self) -> List[int]:
        """Getter for physical qubit indices."""
        try:
            return list(self.__experiment_metadata["physical_qubits"])
        except (TypeError, KeyError):
            # Ignore experiment metadata is not set or key is not found
            return None

    @property
    def _backend(self) -> Backend:
        """Getter for backend object."""
        return self.__backend

    def _experiment_options(self, index: int = -1) -> Dict[str, Any]:
        """Return the experiment options of given job index.

        Args:
            index: Index of job metadata to extract. Default to -1 (latest).

        Returns:
            Experiment options. This option is used for circuit generation.
        """
        try:
            return self.__experiment_metadata["job_metadata"][index]["experiment_options"]
        except (TypeError, KeyError, IndexError):
            # Ignore experiment metadata or job metadata is not set or key is not found
            return None

    def _run_options(self, index: int = -1) -> Dict[str, Any]:
        """Returns the run options of given job index.

        Args:
            index: Index of job metadata to extract. Default to -1 (latest).

        Returns:
            Run options. This option is used for backend execution.
        """
        try:
            return self.__experiment_metadata["job_metadata"][index]["run_options"]
        except (TypeError, KeyError, IndexError):
            # Ignore experiment metadata or job metadata is not set or key is not found
            return None

    def _transpile_options(self, index: int = -1) -> Dict[str, Any]:
        """Returns the transpile options of given job index.

        Args:
            index: Index of job metadata to extract. Default to -1 (latest).

        Returns:
            Transpile options. This option is used for circuit optimization.
        """
        try:
            return self.__experiment_metadata["job_metadata"][index]["transpile_options"]
        except (TypeError, KeyError, IndexError):
            # Ignore experiment metadata or job metadata is not set or key is not found
            return None

    def _data(
        self,
        series_name: Optional[str] = None,
        label: Optional[str] = "fit_ready",
    ) -> CurveData:
        """Getter for experiment data set.

        Args:
            series_name: Series name to search for.
            label: Label attached to data set. By default it returns "fit_ready" data.

        Returns:
            Filtered curve data set.

        Raises:
            AnalysisError: When requested series or label are not defined.
        """
        # pylint: disable = undefined-loop-variable
        for data in self.__processed_data_set:
            if data.label == label:
                break
        else:
            raise AnalysisError(f"Requested data with label {label} does not exist.")

        if series_name is None:
            return data

        for idx, series_def in enumerate(self.__series__):
            if series_def.name == series_name:
                locs = data.data_index == idx
                return CurveData(
                    label=label,
                    x=data.x[locs],
                    y=data.y[locs],
                    y_err=data.y_err[locs],
                    shots=data.shots[locs],
                    data_index=idx,
                    metadata=data.metadata[locs] if data.metadata is not None else None,
                )

        raise AnalysisError(f"Specified series {series_name} is not defined in this analysis.")

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List["pyplot.Figure"]]:

        # get experiment metadata
        try:
            self.__experiment_metadata = experiment_data.metadata

        except AttributeError:
            pass

        # get backend
        try:
            self.__backend = experiment_data.backend
        except AttributeError:
            pass

        #
        # 1. Setup data processor
        #

        # If no data processor was provided at run-time we infer one from the job
        # metadata and default to the data processor for averaged classified data.
        data_processor = self.options.data_processor

        if not data_processor:
            data_processor = get_processor(experiment_data, self.options)

        if isinstance(data_processor, DataProcessor) and not data_processor.is_trained:
            # Qiskit DataProcessor instance. May need calibration.
            data_processor.train(data=experiment_data.data())

        #
        # 2. Extract curve entries from experiment data
        #
        self._extract_curves(experiment_data=experiment_data, data_processor=data_processor)

        #
        # 3. Run fitting
        #
        formatted_data = self._data(label="fit_ready")
        fixed_params = {p: self.options.get(p) for p in self.__fixed_parameters__}

        fit_results = []
        for fit_func in self.composite_funcs:
            # Set parameter and data index to the composite fit function
            if fixed_params:
                fit_func.bind_parameters(**fixed_params)
            fit_func.data_index = formatted_data.data_index

            # Generate algorithmic initial guesses and boundaries
            default_fit_opt = FitOptions(
                group=fit_func.group,
                parameters=fit_func.signature,
                default_p0=self.options.p0,
                default_bounds=self.options.bounds,
                **self.options.curve_fitter_options,
            )

            fit_options = self._generate_fit_guesses(default_fit_opt)
            if isinstance(fit_options, FitOptions):
                fit_options = [fit_options]

            # Run fit for each configuration
            temp_results = []
            for fit_opt in set(fit_options):
                try:
                    fit_result = self.curve_fit(
                        func=fit_func,
                        xdata=formatted_data.x,
                        ydata=formatted_data.y,
                        sigma=formatted_data.y_err,
                        **fit_opt.options,
                    )
                    temp_results.append(fit_result)
                except AnalysisError:
                    # Some guesses might be too far from the true parameters and may thus fail.
                    # We ignore initial guesses that fail and continue with the next fit candidate.
                    pass

            # Find best value with chi-squared value
            if len(temp_results) == 0:
                warnings.warn(
                    "All initial guesses and parameter boundaries failed to fit the data "
                    f"in the fit group {fit_func.group}. Please provide better initial guesses "
                    "or fit parameter boundaries.",
                    UserWarning,
                )
            else:
                best_fit_result = sorted(temp_results, key=lambda r: r.reduced_chisq)[0]
                fit_results.append(best_fit_result)

        #
        # 4. Create database entry
        #
        analysis_results = []
        for fit_result in fit_results:

            # pylint: disable=assignment-from-none
            quality = self._evaluate_quality(fit_data=fit_result)

            # overview entry
            analysis_results.append(
                AnalysisResultData(
                    name=PARAMS_ENTRY_PREFIX + self.__class__.__name__,
                    value=[p.nominal_value for p in fit_result.popt],
                    chisq=fit_result.reduced_chisq,
                    quality=quality,
                    extra={
                        "group": fit_result.group,
                        "popt_keys": fit_result.popt_keys,
                        "dof": fit_result.dof,
                        "covariance_mat": fit_result.pcov,
                        "fit_models": fit_result.fit_mdoel,
                        **self.options.extra,
                    },
                )
            )

            # output special parameters
            result_parameters = self.options.result_parameters
            if result_parameters:
                for param_repr in result_parameters:
                    if isinstance(param_repr, ParameterRepr):
                        p_name = param_repr.name
                        p_repr = param_repr.repr or param_repr.name
                        unit = param_repr.unit
                    else:
                        p_name = param_repr
                        p_repr = param_repr
                        unit = None

                    fit_val = fit_result.fitval(p_name)
                    if unit:
                        metadata = copy.copy(self.options.extra)
                        metadata["unit"] = unit
                    else:
                        metadata = self.options.extra

                    result_entry = AnalysisResultData(
                        name=p_repr,
                        value=fit_val,
                        chisq=fit_result.reduced_chisq,
                        quality=quality,
                        extra={
                            "group": fit_result.group,
                            "fit_models": fit_result.fit_mdoel,
                            **metadata,
                        },
                    )
                    analysis_results.append(result_entry)

        # add extra database entries
        extra_entries = self._extra_database_entry(
            fit_results[0] if len(fit_results) == 1 else fit_results  # for backward compatibility
        )
        analysis_results.extend(extra_entries)

        if self.options.return_data_points:
            # save raw data points in the data base if option is set (default to false)
            raw_data_dict = dict()
            for series_def in self.__series__:
                series_data = self._data(series_name=series_def.name, label="raw_data")
                raw_data_dict[series_def.name] = {
                    "xdata": series_data.x,
                    "ydata": series_data.y,
                    "sigma": series_data.y_err,
                }
            raw_data_entry = AnalysisResultData(
                name=DATA_ENTRY_PREFIX + self.__class__.__name__,
                value=raw_data_dict,
                extra={
                    "x-unit": self.options.xval_unit,
                    "y-unit": self.options.yval_unit,
                },
            )
            analysis_results.append(raw_data_entry)

        #
        # 5. Create figures
        #
        if self.options.plot:
            fit_figure = FitResultPlotters[self.options.curve_plotter].value.draw(
                series_defs=self.__series__,
                raw_samples=[self._data(ser.name, "raw_data") for ser in self.__series__],
                fit_samples=[self._data(ser.name, "fit_ready") for ser in self.__series__],
                tick_labels={
                    "xval_unit": self.options.xval_unit,
                    "yval_unit": self.options.yval_unit,
                    "xlabel": self.options.xlabel,
                    "ylabel": self.options.ylabel,
                    "xlim": self.options.xlim,
                    "ylim": self.options.ylim,
                },
                fit_data=fit_results,
                fix_parameters=fixed_params,
                result_entries=analysis_results,
                style=self.options.style,
                axis=self.options.axis,
            )
            figures = [fit_figure]
        else:
            figures = []

        return analysis_results, figures


def is_error_not_significant(
    val: Union[float, uncertainties.UFloat],
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
