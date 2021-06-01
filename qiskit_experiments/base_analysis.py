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
from typing import List, Tuple

from qiskit.providers.options import Options
from qiskit.exceptions import QiskitError

from qiskit_experiments.experiment_data import ExperimentData, AnalysisResult

# pylint: disable = unused-import
from qiskit_experiments.matplotlib import pyplot


class BaseAnalysis(ABC):
    """Base Analysis class for analyzing Experiment data.

    The data produced by experiments (i.e. subclasses of BaseExperiment)
    are analyzed with subclasses of BaseExperiment. The analysis is
    typically run after the data has been gathered by the experiment.
    For example, an analysis may perform some data processing of the
    measured data and a fit to a function to extract a parameter.

    When designing Analysis subclasses default values for any kwarg
    analysis options of the `run` method should be set by overriding
    the `_default_options` class method. When calling `run` these
    default values will be combined with all other option kwargs in the
    run method and passed to the `_run_analysis` function.
    """

    # Expected experiment data container for analysis
    __experiment_data__ = ExperimentData

    @classmethod
    def _default_options(cls) -> Options:
        return Options()

    def run(
        self,
        experiment_data: ExperimentData,
        save: bool = True,
        return_figures: bool = False,
        **options,
    ):
        """Run analysis and update ExperimentData with analysis result.

        Args:
            experiment_data: the experiment data to analyze.
            save: if True save analysis results and figures to the
                  :class:`ExperimentData`.
            return_figures: if true return a pair of
                            ``(analysis_results, figures)``,
                            otherwise return only analysis_results.
            options: additional analysis options. See class documentation for
                     supported options.

        Returns:
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
        # Get analysis options
        analysis_options = self._default_options()
        analysis_options.update_options(**options)
        analysis_options = analysis_options.__dict__

        # Run analysis
        # pylint: disable=broad-except
        try:
            analysis_results, figures = self._run_analysis(experiment_data, **analysis_options)
            analysis_results["success"] = True
        except Exception as ex:
            analysis_results = AnalysisResult(success=False, error_message=ex)
            figures = None

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
    def _run_analysis(
        self, experiment_data: ExperimentData, **options
    ) -> Tuple[List[AnalysisResult], List["pyplot.Figure"]]:
        """Run analysis on circuit data.

        Args:
            experiment_data: the experiment data to analyze.
            options: additional options for analysis. By default the fields and
                     values in :meth:`options` are used and any provided values
                     can override these.

        Returns:
            A pair ``(analysis_results, figures)`` where ``analysis_results``
            may be a single or list of AnalysisResult objects, and ``figures``
            is a list of any figures for the experiment.
        """
        pass
