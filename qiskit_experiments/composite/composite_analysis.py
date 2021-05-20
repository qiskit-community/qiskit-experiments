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
from qiskit_experiments.base_analysis import BaseAnalysis, AnalysisResult
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
            tuple: A pair ``(analysis_results, figures)`` where
                   ``analysis_results`` may be a single or list of
                   AnalysisResult objects, and ``figures`` may be
                   None, a single figure, or a list of figures.

        Raises:
            QiskitError: if analysis is attempted on non-composite
                         experiment data.
        """
        if not isinstance(experiment_data, CompositeExperimentData):
            raise QiskitError("CompositeAnalysis must be run on CompositeExperimentData.")

        # Run analysis for sub-experiments
        for expr, expr_data in zip(
            experiment_data._experiment._experiments, experiment_data._components
        ):
            expr.run_analysis(expr_data, **options)

        # Add sub-experiment metadata as result of batch experiment
        # Note: if Analysis results had ID's these should be included here
        # rather than just the sub-experiment IDs
        sub_types = []
        sub_ids = []
        sub_qubits = []

        comp_exp = experiment_data.experiment
        for i in range(comp_exp.num_experiments):
            # Run analysis for sub-experiments and add sub-experiment metadata
            expdata = experiment_data.component_experiment_data(i)
            comp_exp.component_analysis(i).run(expdata, **options)

            # Add sub-experiment metadata as result of batch experiment
            # Note: if Analysis results had ID's these should be included here
            # rather than just the sub-experiment IDs
            sub_types.append(expdata.experiment_type)
            sub_ids.append(expdata.experiment_id)
            sub_qubits.append(expdata.experiment.physical_qubits)

        analysis_result = AnalysisResult(
            {
                "experiment_types": sub_types,
                "experiment_ids": sub_ids,
                "experiment_qubits": sub_qubits,
            }
        )
        return analysis_result, None
