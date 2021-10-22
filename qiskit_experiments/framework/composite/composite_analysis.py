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

from qiskit.result import marginal_counts
from qiskit_experiments.framework import BaseAnalysis, ExperimentData


class CompositeAnalysis(BaseAnalysis):
    """Analysis class for CompositeExperiment"""

    # pylint: disable = arguments-differ
    def _run_analysis(self, experiment_data: ExperimentData, **options):
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
        # Maginalize data
        self._marginalize_data(experiment_data)

        comp_exp = experiment_data.experiment

        for i in range(comp_exp.num_experiments):
            # Run analysis for sub-experiments and add sub-experiment metadata
            exp = comp_exp.component_experiment(i)
            expdata = experiment_data.child_data(i)
            exp.run_analysis(expdata, **options)

        return [], []

    def _marginalize_data(self, experiment_data: ExperimentData):
        """Maginalize composite data and store in child experiments"""
        # Marginalize data
        child_data = {}
        for datum in experiment_data.data():
            metadata = datum.get("metadata", {})

            # Add marginalized data to sub experiments
            if "composite_clbits" in metadata:
                composite_clbits = metadata["composite_clbits"]
            else:
                composite_clbits = None
            for i, index in enumerate(metadata["composite_index"]):
                if index not in child_data:
                    # Initialize data list for child data
                    child_data[index] = []
                sub_data = {"metadata": metadata["composite_metadata"][i]}
                if "counts" in datum:
                    if composite_clbits is not None:
                        sub_data["counts"] = marginal_counts(datum["counts"], composite_clbits[i])
                    else:
                        sub_data["counts"] = datum["counts"]
                child_data[index].append(sub_data)

        # Add child data
        for index, data in child_data.items():
            experiment_data.child_data(index).add_data(data)
