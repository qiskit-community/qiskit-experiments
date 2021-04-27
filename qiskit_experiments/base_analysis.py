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
Base analysis class.
"""

from abc import ABC, abstractmethod

from qiskit.exceptions import QiskitError
from qiskit.providers.experiment import AnalysisResult, ExperimentData


class BaseAnalysis(ABC):
    """Base Analysis class for analyzing Experiment data."""

    # Expected experiment data container for analysis
    __experiment_data__ = ExperimentData

    def run(self, experiment_data, save=True, return_figures=False, **options):
        """Run analysis and update stored ExperimentData with analysis result.

        Args:
            experiment_data (ExperimentData): the experiment data to analyze.
            save (bool): if True save analysis results and figures to the
                         :class:`ExperimentData`.
            return_figures (bool): if true return a pair of
                                   ``(analysis_results, figures)``,
                                    otherwise return only analysis_results.
            options: kwarg options for analysis function.

        Returns:
            AnalysisResult: the output of the analysis that produces a
                            single result.
            List[AnalysisResult]: the output for analysis that produces
                                  multiple results.
            Tuple: If ``return_figures=True`` the output is a pair
                   ``(analysis_results, figures)`` where  ``analysis_results``
                   may be a single or list of :class:`AnalysisResult` objects, and
                   ``figures`` may be None, a single figure, or a list of figures.

        Raises:
            QiskitError: if experiment_data container is not valid for analysis.
        """
        if not isinstance(experiment_data, self.__experiment_data__):
            raise QiskitError(
                f"Invalid experiment data type, expected {self.__experiment_data__.__name__}"
                f" but received {type(experiment_data).__name__}"
            )

        # Wait for experiment job to finish
        # experiment_data.block_for_result()

        # Run analysis
        analysis_results, figures = self._run_analysis(experiment_data, **options)
        # Save to experiment data
        if save:
            if isinstance(analysis_results, AnalysisResult):
                experiment_data.add_analysis_result(analysis_results)
            else:
                for res in analysis_results:
                    experiment_data.add_analysis_result(res)
            if figures:
                for fig in figures:
                    experiment_data.add_figure(fig)
        if return_figures:
            return analysis_results, figures
        return analysis_results

    @abstractmethod
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
        pass
