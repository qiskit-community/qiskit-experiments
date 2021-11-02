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
        composite_exp = experiment_data.experiment
        component_exps = composite_exp.component_experiment()
        if "component_job_metadata" in experiment_data.metadata:
            component_metadata = experiment_data.metadata["component_job_metadata"][-1]
        else:
            component_metadata = [{}] * composite_exp.num_experiments

        # Check if component experiment data has already been initialized
        components_exist = self._components_initialized(composite_exp, experiment_data)

        # Compute marginalize data
        marginalized_data = self._marginalize_data(experiment_data.data())

        # Construct component experiment data
        for i, (sub_data, sub_exp) in enumerate(zip(marginalized_data, component_exps)):
            if components_exist:
                # Get existing component ExperimentData and clear any previously
                # stored data
                sub_exp_data = experiment_data.component_experiment_data(i)
                sub_exp_data._data.clear()
            else:
                # Initialize component ExperimentData and add as child data
                sub_exp_data = sub_exp._initialize_experiment_data()
                experiment_data.add_child_data(sub_exp_data)

            # Add component job metadata
            sub_exp_data._metadata["job_metadata"] = [component_metadata[i]]

            # Add marginalized data
            sub_exp_data.add_data(sub_data)

            # Run analysis
            sub_exp.run_analysis(sub_exp_data)

        return [], []

    def _components_initialized(self, experiment, experiment_data):
        """Return True if component experiment data is initialized"""
        if len(experiment_data.child_data()) != experiment.num_experiments:
            return False
        for data, exp in zip(experiment.composite_experiment(), experiment_data.child_data()):
            if exp.experiment_type == data.experiment_type:
                return False
        return True

    def _marginalize_data(self, composite_data):
        """Return marginalized data for component experiments"""
        # Marginalize data
        marginalized_data = {}
        for datum in composite_data:
            metadata = datum.get("metadata", {})

            # Add marginalized data to sub experiments
            if "composite_clbits" in metadata:
                composite_clbits = metadata["composite_clbits"]
            else:
                composite_clbits = None
            for i, index in enumerate(metadata["composite_index"]):
                if index not in marginalized_data:
                    # Initialize data list for marginalized
                    marginalized_data[index] = []
                sub_data = {"metadata": metadata["composite_metadata"][i]}
                if "counts" in datum:
                    if composite_clbits is not None:
                        sub_data["counts"] = marginal_counts(datum["counts"], composite_clbits[i])
                    else:
                        sub_data["counts"] = datum["counts"]
                marginalized_data[index].append(sub_data)

        # Sort by index
        return [marginalized_data[i] for i in sorted(marginalized_data.keys())]
