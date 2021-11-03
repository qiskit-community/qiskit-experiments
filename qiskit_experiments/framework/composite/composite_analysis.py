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

        # Initialize component data for updating and get the experiment IDs for
        # the component child experiments
        component_ids = self._initialize_components(composite_exp, experiment_data)

        # Compute marginalize data
        marginalized_data = self._marginalize_data(experiment_data.data())

        # Construct component experiment data
        for i, (sub_data, sub_exp) in enumerate(zip(marginalized_data, component_exps)):
            sub_exp_data = experiment_data.child_data(component_ids[i])

            # Clear any previously stored data and add marginalized data
            sub_exp_data._data.clear()
            sub_exp_data.add_data(sub_data)

            # Add component job metadata
            sub_exp_data._metadata["job_metadata"] = [component_metadata[i]]

            # Run analysis
            # Since copy for replace result is handled at the parent level
            # we always run with replace result on component analysis
            sub_exp.run_analysis(sub_exp_data, replace_results=True)

        return [], []

    def _initialize_components(self, experiment, experiment_data):
        """Initialize child data components and return list of child experiment IDs"""
        component_index = experiment_data._metadata.get("component_child_index", [])
        if not component_index:
            # Construct component data and update indices
            start_index = len(experiment_data.child_data())
            component_index = []
            for i, sub_exp in enumerate(experiment.component_experiment()):
                sub_data = sub_exp._initialize_experiment_data()
                experiment_data.add_child_data(sub_data)
                component_index.append(start_index + i)
            experiment_data._metadata["component_child_index"] = component_index

        # Child components exist so we can get their ID for accessing them
        child_ids = experiment_data._child_data.keys()
        component_ids = [child_ids[idx] for idx in component_index]
        return component_ids

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
