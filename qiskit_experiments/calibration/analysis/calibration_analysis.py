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

from abc import abstractmethod
from typing import Iterator, List, Tuple
import numpy as np
from scipy import optimize

from qiskit_experiments.experiment_data import AnalysisResult
from qiskit_experiments.experiment_data import ExperimentData
from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.data_processing.data_processor import DataProcessor
from qiskit_experiments.calibration.metadata import CalibrationMetadata
from qiskit_experiments.data_processing.processed_data import ProcessedData
from .fit_result import FitResult


try:
    import matplotlib
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class BaseCalibrationAnalysis(BaseAnalysis):
    """Abstract base class for all calibration analysis classes."""

    @abstractmethod
    def initial_guess(self, xvals: np.ndarray, yvals: np.ndarray) -> Iterator[np.ndarray]:
        """Create initial guess for fit parameters.

        Args:
            xvals: x values to fit.
            yvals: y values to fit.

        Yield:
            A set of initial parameter guesses.
            If multiple guesses are returned a fit is performed for each guess.
            The error of the fit is measured by the Chi squared and the best fit result is chosen.

        Note:
            This should return values with yield rather than return.
        """

    @abstractmethod
    def fit_boundary(self, xvals: np.ndarray, yvals: np.ndarray) -> List[Tuple[float, float]]:
        """Returns the boundary of the parameters to fit.

        Args:
            xvals: x values to fit.
            yvals: y values to fit.
        """

    @abstractmethod
    def fit_function(self, xvals: np.ndarray, *args) -> np.ndarray:
        """Fit function.

        Args:
            xvals: x values to fit.
            args: The parameters of the fit function.
        """

    def chi_squared(self,
                    parameters: np.ndarray,
                    xvals: np.ndarray,
                    yvals: np.ndarray) -> float:
        """Calculate the reduced Chi squared value.

        Args:
            parameters: Parameters for the fit function.
            xvals: x values to fit.
            yvals: y values to fit.

        Returns:
            The chi-squared value.
        """
        fit_yvals = self.fit_function(xvals, *parameters)

        chi_sq = sum((fit_yvals - yvals) ** 2)
        dof = len(xvals) - len(parameters)

        return chi_sq / dof

    #pylint: disable = arguments-differ
    def _run_analysis(self, experiment_data: ExperimentData, qubit=0,
                      data_processor=DataProcessor(), plot=False, **kwargs) -> any:
        """
        Analyze the given experiment data.

        Notes: data is a List of Dict.

        Args:
            experiment_data: The data to analyse.
            qubit: The qubit for which to analyse the data.
            data_processor: The data processor which transforms the measured data to
                a format that can be analyzed. For example, IQ data may be converted to
                a signal by taking the real-part.

        Returns:
            any: the output of the analysis,
        """

        # 1) Format the data using the DataProcessor for the analysis.
        for data in experiment_data.data:
            data_processor.format_data(data)

        # 2) Extract series information from the data
        key = data_processor.output_key()
        series = ProcessedData()
        for data in experiment_data.data:
            metadata = CalibrationMetadata(**data['metadata'])

            # Single-shot data.
            if isinstance(data[key][0], list):
                yval = [data[key][_][qubit] for _ in range(len(data[key]))]

            # Averaged data.
            else:
                yval = data[key][qubit]

            series.add_data_point(metadata.x_values, yval, metadata.series)

        # 3) Fit the data for each initial guess and series.
        results = AnalysisResult()
        for xvals, yvals, series in series.series():
            best_result = None
            for initial_guess in self.initial_guess(xvals, yvals):

                fit_result = optimize.minimize(
                    fun=self.chi_squared,
                    x0=initial_guess,
                    args=(xvals, yvals),
                    bounds=self.fit_boundary(xvals, yvals)
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

            results[series] = best_result

        experiment_data.add_analysis_result(results)

        figures = None
        if plot and HAS_MATPLOTLIB:
            figures = self.plot(qubit, results, **kwargs)

        return results, figures

    def plot(self, qubit, results, **kwargs) -> List:
        """
        Plot the result.

        Args:
            qubit: The qubit for which the analysis was done.
            results: The results of the fit. Holds the fit function, xvals, and yvals.
            kwargs: key word arguments supported are
                - figsize: the size of the figure
                - ax: the axis on which to plot the results

        Returns:
            A matplotlib figure.
        """
        if 'ax' in kwargs:
            figure = kwargs['ax'].figure
        else:
            figure = plt.figure(figsize=kwargs.get('figsize', (6, 4)))
            ax = figure.add_subplot(111)

        line_counts = 0
        for series_name, result in results.items():

            # plot fit line
            xval_interp = np.linspace(result.xvals[0], result.xvals[-1], 100)
            yval_fit = self.fit_function(xval_interp, *result.fitvals)

            fit_line_color = plt.cm.tab20.colors[(2*line_counts+1) % plt.cm.tab20.N]
            data_label = '{tag} (Q{qubit:d})'.format(tag=series_name, qubit=qubit)
            ax.plot(xval_interp, yval_fit, '--', color=fit_line_color, label=data_label)

            # plot data scatter
            data_scatter_color = plt.cm.tab20.colors[(2*line_counts) % plt.cm.tab20.N]
            ax.plot(result.xvals, result.yvals, 'o', color=data_scatter_color)

            ax.set_xlim(result.xvals[0], result.xvals[-1])

            line_counts += 1

        ax.set_xlabel(kwargs.get('xlabel', kwargs.get('xlabel', 'Parameter')), fontsize=14)
        ax.set_ylabel(kwargs.get('ylabel', kwargs.get('ylabel', 'Signal')), fontsize=14)
        ax.grid()
        ax.legend()

        if matplotlib.get_backend() in ['module://ipykernel.pylab.backend_inline', 'nbAgg']:
            plt.close(figure)

        return [figure]
