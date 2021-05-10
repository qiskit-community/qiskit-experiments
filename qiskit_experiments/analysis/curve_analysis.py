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

from abc import abstractmethod, abstractstaticmethod
from typing import Any, NamedTuple, Dict, List, Optional, Tuple, Callable

import numpy as np
import scipy.optimize as opt
from qiskit.exceptions import QiskitError

from collections import defaultdict
import functools

from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.experiment_data import AnalysisResult, ExperimentData

# Description of data properties for single curve entry
SeriesDef = NamedTuple(
    "SeriesDef",
    [
        ("name", str),
        ("param_names", List[str]),
        ("fit_func_index", int),
        ("filter_kwargs", Optional[Dict[str, Any]]),
    ],
)

# Human readable data set for single curve entry
CurveEntry = NamedTuple(
    "CurveEntry",
    [
        ("curve_name", str),
        ("x_values", np.ndarray),
        ("y_values", np.ndarray),
        ("y_sigmas", np.ndarray),
        ("metadata", dict),
    ],
)


class CurveAnalysis(BaseAnalysis):

    #: str: Metadata key representing a scanned value.
    __x_key__ = ""

    #: List[SeriesDef]: List of mapping representing a data series
    __series__ = None

    #: Callable: Data processor
    __data_processor__ = None

    #: List[Callable]: A callback function to define the expected curve
    __fit_funcs__ = None

    #: List[str]: Parameter name list
    __param_names__ = list()

    # TODO data processor may be initialized with some variables.
    # For example, if it contains front end discriminator, it may be initialized with
    # some discrimination line parameters. These parameters cannot be hard-coded here.

    @abstractmethod
    def _run_fitting(self, curve_data: List[CurveEntry]):
        pass

    @abstractmethod
    def _create_figure(self, curve_data: List[CurveEntry], fit_data: AnalysisResult):
        pass

    def _run_analysis(self, experiment_data, **options):
        """Run analysis on circuit data.

        Args:
            experiment_data (ExperimentData): the experiment data to analyze.
            options: kwarg options for analysis function.

        Returns:
            tuple: A pair ``(analysis_results, figures)`` where
                   ``analysis_results`` may be a single or list of
                   AnalysisResult objects, and ``figures`` may be
                   None, a single figure, or a list of figures.
        """
        analysis_result = AnalysisResult()

        try:
            curve_data = self._data_processing(experiment_data)
            # TODO this data should be kept in somewhere in analysis result
            # however the raw data may be avoided to be saved in remote database
        except DataProcessorError:
            analysis_result["success"] = False
            return analysis_result, list()

        try:
            fit_data = self._run_fitting(curve_data=curve_data)
            analysis_result.update(fit_data)
            analysis_result["success"] = True
        except Exception:
            analysis_result["success"] = False

        analysis_result = self._post_processing(analysis_result)
        figures = self._create_figure(curve_data=curve_data, fit_data=analysis_result)

        return analysis_result, figures

    def _data_processing(self, experiment_data: ExperimentData) -> List[CurveEntry]:
        """Extract curve data from experiment data.

        .. notes::
            The target metadata properties to define each curve entry is described by
            the class attribute __series__. This method returns the same numbers
            of curve data entries as one defined in this attribute.
            The returned CurveData entry contains circuit metadata fields that are
            common to the entire curve scan, i.e. series-level metadata.

        Args:
            experiment_data: ExperimentData object to fit parameters.

        Returns:
            List of ``CurveEntry`` containing x-values, y-values, and y values sigma.
        """
        if self.__series__:
            series = self.__series__
        else:
            series = [
                SeriesDef(
                    name="",
                    param_names=self.__param_names__,
                    fit_func_index=0,
                    filter_kwargs=None,
                )
            ]

        def _is_target_series(datum, **filters):
            try:
                return all(datum["metadata"][key] == val for key, val in filters.items())
            except KeyError:
                return False

        curve_data = list()
        for curve_properties in series:
            if curve_properties.filter_kwargs:
                # filter data
                series_data = [
                    datum
                    for datum in experiment_data.data
                    if _is_target_series(datum, **curve_properties.filter_kwargs)
                ]
            else:
                # use data as-is
                series_data = experiment_data
            xvals, yvals, sigmas = self.__data_processor__(series_data)
            # TODO data processor may need calibration.
            # If we use the level1 data, it may be necessary to calculate principal component
            # with entire scan data. Otherwise we need to use real or imaginary part.

            # Get common metadata fields except for xval and filter args.
            # These properties are obvious.
            common_keys = list(
                functools.reduce(
                    lambda k1, k2: k1 & k2,
                    map(lambda d: d.keys(), [datum["metadata"] for datum in series_data]),
                )
            )
            common_keys.remove(self.__x_key__)
            if curve_properties.filter_kwargs:
                for key in curve_properties.filter_kwargs:
                    common_keys.remove(key)

            # Extract common metadata for the curve
            curve_metadata = defaultdict(set)
            for datum in series_data:
                for key in common_keys:
                    curve_metadata[key].add(datum["metadata"][key])

            curve_data.append(
                CurveEntry(
                    curve_name=curve_properties.name,
                    x_values=xvals,
                    y_values=yvals,
                    y_sigmas=sigmas,
                    metadata=curve_metadata,
                )
            )

        return curve_data

    @staticmethod
    def _post_processing(analysis_result: AnalysisResult) -> AnalysisResult:
        return analysis_result

    def _series_curve_fit(
            self,
            curve_data: List[CurveEntry],
            p0: np.ndarray,
            weights: Optional[np.ndarray] = None,
            bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            **kwargs
    ):

        # remap parameters
        pass

    @staticmethod
    def _single_curve_fit(
        func: Callable,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray,
        sigma: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs,
    ) -> AnalysisResult:
        r"""Perform a non-linear least squares to fit

        This solves the optimization problem

        .. math::
            \Theta_{\mbox{opt}} = \arg\min_\Theta \sum_i \sigma_i^{-2} (f(x_i, \Theta) -  y_i)^2

        using ``scipy.optimize.curve_fit``.

        Args:
            func: a fit function `f(x, *params)`.
            xdata: a 1D float array of x-data.
            ydata: a 1D float array of y-data.
            p0: initial guess for optimization parameters.
            sigma: Optional, a 1D array of standard deviations in ydata in absolute units.
            bounds: Optional, lower and upper bounds for optimization parameters.
            kwargs: additional kwargs for scipy.optimize.curve_fit.

        Returns:
            result containing ``popt`` the optimal fit parameters,
            ``popt_err`` the standard error estimates popt,
            ``pcov`` the covariance matrix for the fit,
            ``reduced_chisq`` the reduced chi-squared parameter of fit,
            ``dof`` the degrees of freedom of the fit,
            ``xrange`` the range of xdata values used for fit.

        Raises:
            QiskitError: if the number of degrees of freedom of the fit is
                         less than 1.

        .. note::
            ``sigma`` is assumed to be specified in the same units as ``ydata``
            (absolute units). If sigma is instead specified in relative units
            the `absolute_sigma=False` kwarg of scipy curve_fit must be used.
            This affects the returned covariance ``pcov`` and error ``popt_err``
            parameters via ``pcov(absolute_sigma=False) = pcov * reduced_chisq``
            ``popt_err(absolute_sigma=False) = popt_err * sqrt(reduced_chisq)``.
        """
        # Check the degrees of freedom is greater than 0
        dof = len(ydata) - len(p0)
        if dof < 1:
            raise QiskitError(
                "The number of degrees of freedom of the fit data and model "
                " (len(ydata) - len(p0)) is less than 1"
            )

        # Override scipy.curve_fit default for absolute_sigma=True
        # if sigma is specified.
        if sigma is not None and "absolute_sigma" not in kwargs:
            kwargs["absolute_sigma"] = True

        # Run curve fit
        # pylint: disable = unbalanced-tuple-unpacking
        try:
            popt, pcov = opt.curve_fit(
                f=func, xdata=xdata, ydata=ydata, sigma=sigma, p0=p0, bounds=bounds, **kwargs
            )
        except Exception as ex:
            # TODO do some error handling
            return AnalysisResult()

        popt_err = np.sqrt(np.diag(pcov))

        # Calculate the reduced chi-squared for fit
        yfits = func(xdata, *popt)
        residues = (yfits - ydata) ** 2
        if sigma is not None:
            residues = residues / (sigma ** 2)
        reduced_chisq = np.sum(residues) / dof

        # Compute xdata range for fit
        xdata_range = [min(xdata), max(xdata)]

        result = {
            "popt": popt,
            # "popt_keys": self.__param_names__,
            "popt_err": popt_err,
            "pcov": pcov,
            "reduced_chisq": reduced_chisq,
            "dof": dof,
            "xrange": xdata_range,
        }

        return AnalysisResult(result)
