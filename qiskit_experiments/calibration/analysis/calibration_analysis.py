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

"""Base analysis class for calibrations."""

import numpy as np
from abc import abstractmethod
from typing import Iterator, List, Optional, Tuple

from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.calibration import DataProcessor
from qiskit_experiments import ExperimentData


class BaseCalibrationAnalysis(BaseAnalysis):
    """Abstract base class for all calibration analysis classes."""

    def __init__(self, experiment_data,
                      data_processor: Optional[DataProcessor] = None,
                      **options):
        """Run analysis on circuit data.

        Args:
            experiment_data (ExperimentData): the experiment data to analyze.
            data_processor: Specifies the actions to apply when processing the measured data.
            options: kwarg options for analysis function.

        Returns:
            tuple: A pair ``(analysis_results, figures)`` where
                   ``analysis_results`` may be a single or list of
                   AnalysisResult objects, and ``figures`` may be
                   None, a single figure, or a list of figures.
        """

        self._data_processor = data_processor

    @property
    def data_processor(self) -> DataProcessor:
        """Return the data processor."""
        return self._data_processor

    @data_processor.setter
    def data_processor(self, data_processor: DataProcessor):
        """Set the data processor."""
        self._data_processor = data_processor

    @abstractmethod
    def initial_guess(self, xvals: np.ndarray, yvals: np.ndarray) -> Iterator[np.ndarray]:
        """Create initial guess for fit parameters.

        Args:
            xvals: x values to fit.
            yvals: y values to fit.

        Yield:
            Set of initial guess for parameters.
            If multiple guesses are returned fit is performed for all parameter set.
            Error is measured by Chi squared value and the best fit result is chosen.

        Note:
            This should return values with yield rather than return.
        """

    @abstractmethod
    def fit_boundary(self, xvals: np.ndarray, yvals: np.ndarray) -> List[Tuple[float, float]]:
        """Returns boundary of parameters to fit.

        Args:
            xvals: x values to fit.
            yvals: y values to fit.
        """

    @abstractmethod
    def fit_function(self, xvals: np.ndarray, *args) -> np.ndarray:
        """Fit function.

        Args:
            xvals: x values to fit.
        """

    def chi_squared(self,
                    parameters: np.ndarray,
                    xvals: np.ndarray,
                    yvals: np.ndarray):
        """Calculate reduced Chi squared value.

        Args:
            parameters: Parameters for fit function.
            xvals: X values to fit.
            yvals: Y values to fit.
        """
        fit_yvals = self.fit_function(xvals, *parameters)

        chi_sq = sum((fit_yvals - yvals) ** 2)
        dof = len(xvals) - len(parameters)

        return chi_sq / dof

    def _run_analysis(self, experiment_data: ExperimentData, **kwargs) -> any:
        """
        TODO BIG TODO!!!!

        Analyze the given experiment data.

        Args:
            qubit: Index of qubit to analyze the result.

        Returns:
            any: the output of the analysis,
        """

        metadata = experiment_data.data.header.metadata

        for idx, data in enumerate(experiment_data.data):
        data = self.data_processor.format_data(data, data['metadata'])


        qubit_data = experiment_data.groupby('qubit').get_group(qubit)
        temp_results = dict()

        # fit for each initial guess
        for xvals, yvals, series in self._get_target_data(qubit_data):
            # fit for each series
            best_result = None
            for initial_guess in self.initial_guess(xvals, yvals):
                # fit for each initial guess if there are many starting point
                fit_result = optimize.minimize(
                    fun=self.chi_squared,
                    x0=initial_guess,
                    args=(xvals, yvals),
                    bounds=self.fit_boundary(xvals, yvals),
                    **kwargs
                )
                if fit_result.success:
                    if not best_result or best_result.chisq > fit_result.fun:
                        best_result = FitResult(
                            fitvals=fit_result.x,
                            chisq=fit_result.fun,
                            xvals=xvals,
                            yvals=yvals
                        )
                else:
                    # fit failed, output log `fit_result.message`
                    pass

            # keep the best result
            temp_results[series] = best_result

        # update analysis result
        if self._result:
            self._result[qubit] = temp_results
        else:
            self._result = {qubit: temp_results}

        return temp_results

    def _get_target_data(self, data: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Iterator to retrieve the series from the data.

        The user can generate arbitrary data sets to fit to.
        Each data (xvals, yvals) in the data set should be returned as iterator
        with the tagged name. Analysis `.run` method receives this data and perform
        fitting. User can overwrite this method with respect to the data structure
        that generator defines.

        Args:
            data: Data source. The table has column of fit parameter names defined by
                associated generator, series, experiment, and value.

        Yield:
            Set of x-values and y-values with a string identifying the series.
        """
        if len(self.x_values) > 1:
            raise CalExpError('Default method does not support multi dimensional scan.')

        for series in data['series'].unique():
            xvals = np.array(data[data['series'] == series][self.x_values[0]])
            yvals = np.array(data[data['series'] == series]['value'])

            yield xvals, yvals, series
