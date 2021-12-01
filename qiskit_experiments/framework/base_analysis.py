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
from uncertainties.core import UFloat

from qiskit_experiments.database_service.device_component import Qubit
from qiskit_experiments.database_service.db_fitval import FitVal
from qiskit_experiments.framework import Options
from qiskit_experiments.framework.experiment_data import ExperimentData
from qiskit_experiments.framework.analysis_result_data import AnalysisResultData
from qiskit_experiments.database_service import DbAnalysisResultV1


class BaseAnalysis(ABC):
    """Abstract base class for analyzing Experiment data.

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

    @classmethod
    def _default_options(cls) -> Options:
        return Options()

    def run(
        self,
        experiment_data: ExperimentData,
        replace_results: bool = False,
        **options,
    ) -> ExperimentData:
        """Run analysis and update ExperimentData with analysis result.

        Args:
            experiment_data: the experiment data to analyze.
            replace_results: if True clear any existing analysis results and
                             figures in the experiment data and replace with
                             new results. See note for additional information.
            options: additional analysis options. See class documentation for
                     supported options.

        Returns:
            An experiment data object containing the analysis results and figures.

        Raises:
            QiskitError: if experiment_data container is not valid for analysis.

        .. note::
            **Updating Results**

            If analysis is run with ``replace_results=True`` then any analysis results
            and figures in the experiment data will be cleared and replaced with the
            new analysis results. Saving this experiment data will replace any
            previously saved data in a database service using the same experiment ID.

            If analysis is run with ``replace_results=False`` and the experiment data
            being analyzed has already been saved to a database service, or already
            contains analysis results or figures, a copy with a unique experiment ID
            will be returned containing only the new analysis results and figures.
            This data can then be saved as its own experiment to a database service.
        """
        # Make a new copy of experiment data if not updating results
        if not replace_results and (
            experiment_data._created_in_db
            or experiment_data._analysis_results
            or experiment_data._figures
            or getattr(experiment_data, "_child_data", None)
        ):
            experiment_data = experiment_data.copy()

        # Get experiment device components
        if "physical_qubits" in experiment_data.metadata:
            experiment_components = [
                Qubit(qubit) for qubit in experiment_data.metadata["physical_qubits"]
            ]
        else:
            experiment_components = []

        # Get analysis options
        analysis_options = self._default_options()
        analysis_options.update_options(**options)
        analysis_options = analysis_options.__dict__

        def run_analysis(expdata):
            results, figures = self._run_analysis(expdata, **analysis_options)
            # Add components
            analysis_results = [
                self._format_analysis_result(result, expdata.experiment_id, experiment_components)
                for result in results
            ]
            # Update experiment data with analysis results
            experiment_data._clear_results()
            if analysis_results:
                expdata.add_analysis_results(analysis_results)
            if figures:
                expdata.add_figures(figures)

        experiment_data.add_analysis_callback(run_analysis)

        return experiment_data

    def _format_analysis_result(self, data, experiment_id, experiment_components=None):
        """Format run analysis result to DbAnalysisResult"""
        device_components = []
        if data.device_components:
            device_components = data.device_components
        elif experiment_components:
            device_components = experiment_components

        # Convert ufloat to FitVal so that database service can parse
        # TODO completely deprecate FitVal. We can store UFloat in database.
        if isinstance(data.value, UFloat):
            value = FitVal(
                value=data.value.nominal_value,
                stderr=data.value.std_dev,
                unit=data.unit,
            )
        else:
            value = data.value

        return DbAnalysisResultV1(
            name=data.name,
            value=value,
            device_components=device_components,
            experiment_id=experiment_id,
            chisq=data.chisq,
            quality=data.quality,
            extra=data.extra,
        )

    @abstractmethod
    def _run_analysis(
        self, experiment_data: ExperimentData, **options
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        """Run analysis on circuit data.

        Args:
            experiment_data: the experiment data to analyze.
            options: additional options for analysis. By default the fields and
                     values in :meth:`options` are used and any provided values
                     can override these.

        Returns:
            A pair ``(analysis_results, figures)`` where ``analysis_results``
            is a list of :class:`AnalysisResultData` objects, and ``figures``
            is a list of any figures for the experiment.

        Raises:
            AnalysisError: if the analysis fails.
        """
        pass
