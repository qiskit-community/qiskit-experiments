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
Composite Experiment Analysis class.
"""

from qiskit.exceptions import QiskitError
from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData
from .composite_experiment_data import CompositeExperimentData


class CompositeAnalysis(BaseAnalysis):
    """Analysis class for CompositeExperiment"""

    __experiment_data__ = CompositeExperimentData

    # pylint: disable = arguments-differ
    def _run_analysis(self, experiment_data: CompositeExperimentData, **options):
        """Run analysis on circuit data.

        Args:
            experiment_data: the experiment data to analyze.
            options: kwarg options for analysis function.

        Returns:
            tuple: A pair ``(analysis_results, figures)`` where ``analysis_results``
                   is a list of :class:`AnalysisResultData` objects, and ``figures``
                   is a list of any figures for the experiment.

        Raises:
            QiskitError: if analysis is attempted on non-composite
                         experiment data.
        """
        if not isinstance(experiment_data, CompositeExperimentData):
            raise QiskitError("CompositeAnalysis must be run on CompositeExperimentData.")

        comp_exp = experiment_data.experiment

        for i in range(comp_exp.num_experiments):
            # Run analysis for sub-experiments and add sub-experiment metadata
            exp = comp_exp.component_experiment(i)
            expdata = experiment_data.component_experiment_data(i)
            exp.run_analysis(expdata, **options)

        result = AnalysisResultData(name="composite_experiment", value=None)

        return [], []
