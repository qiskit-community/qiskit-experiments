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

from typing import List, Dict
import numpy as np
from qiskit.result import marginal_counts
from qiskit_experiments.framework import BaseAnalysis, ExperimentData, AnalysisResultData
from qiskit_experiments.database_service.device_component import Qubit


class CompositeAnalysis(BaseAnalysis):
    """Run analysis for composite experiments.

    Composite experiments consist of several component experiments
    run together in a single execution, the results of which are returned
    as a single list of circuit result data in the :class:`ExperimentData`
    container. Analysis of this composite circuit data involves constructing
    a child experiment data container for each component experiment containing
    the marginalized circuit result data for that experiment. Each component
    child data is then analyzed using the analysis class from the corresponding
    component experiment.

    .. note::

        The the child :class:`ExperimentData` for each component experiment
        does not already exist in the experiment data they will be initialized
        and added to the experiment data when :meth:`run` is called on the
        composite :class:`ExperimentData`.

        When calling :meth:`run` on experiment data already containing
        initialized component experiment child data, any previously stored
        circuit data will be cleared and replaced with the marginalized data
        reconstructed from the parent composite experiment data.
    """

    def _run_analysis(self, experiment_data: ExperimentData):
        # Extract job metadata for the component experiments so it can be added
        # to the child experiment data incase it is required by the child experiments
        # analysis classes
        composite_exp = experiment_data.experiment
        component_exps = composite_exp.component_experiment()
        component_metadata = experiment_data.metadata.get(
            "component_metadata", [{}] * composite_exp.num_experiments
        )

        # Initialize component data for updating and get the experiment IDs for
        # the component child experiments in case there are other child experiments
        # in the experiment data
        component_ids = self._initialize_components(composite_exp, experiment_data)

        # Compute marginalize data for each component experiment
        marginalized_data = self._marginalize_data(experiment_data.data())

        # Add the marginalized component data and component job metadata
        # to each component child experiment. Note that this will clear
        # any currently stored data in the experiment. Since copying of
        # child data is handled by the `replace_results` kwarg of the
        # parent container it is safe to always clear and replace the
        # results of child containers in this step
        analysis_results = []
        for i, (sub_data, sub_exp) in enumerate(zip(marginalized_data, component_exps)):
            sub_exp_data = experiment_data.child_data(component_ids[i])

            # Clear any previously stored data and add marginalized data
            sub_exp_data._data.clear()
            sub_exp_data.add_data(sub_data)

            # Add component job metadata
            sub_exp_data.metadata.update(component_metadata[i])

            # Run analysis
            # Since copy for replace result is handled at the parent level
            # we always run with replace result on component analysis
            sub_exp.analysis.run(sub_exp_data, replace_results=True)

            # Record the component experiment id and type as an analysis result
            # for evidence analysis has started and to display in the service DB
            result = AnalysisResultData(
                name=sub_exp_data.experiment_type,
                value=sub_exp_data.experiment_id,
                device_components=[
                    Qubit(qubit) for qubit in sub_exp_data.metadata.get("physical_qubits", [])
                ],
            )
            analysis_results.append(result)

        # Add callback to wait for all component analysis to finish before returning
        # the parent experiment analysis results
        def _wait_for_components(experiment_data, component_ids):
            for comp_id in component_ids:
                experiment_data.child_data(comp_id).block_for_results()

        experiment_data.add_analysis_callback(_wait_for_components, component_ids=component_ids)

        return analysis_results, []

    def _initialize_components(self, experiment, experiment_data):
        """Initialize child data components and return list of child experiment IDs"""
        # Check if component child experiment data containers have already
        # been created. If so the list of indices for their positions in the
        # ordered dict should exist. Index is used to extract the experiment
        # IDs for each child experiment which can change when re-running analysis
        # if replace_results=False, so that we update the correct child data
        # for each component experiment
        component_index = experiment_data.metadata.get("component_child_index", [])
        if not component_index:
            # If the experiment Construct component data and update indices
            start_index = len(experiment_data.child_data())
            component_index = []
            for i, sub_exp in enumerate(experiment.component_experiment()):
                sub_data = sub_exp._initialize_experiment_data()
                experiment_data.add_child_data(sub_data)
                component_index.append(start_index + i)
            experiment_data.metadata["component_child_index"] = component_index

        # Child components exist so we can get their ID for accessing them
        child_ids = experiment_data._child_data.keys()
        component_ids = [child_ids[idx] for idx in component_index]
        return component_ids

    def _marginalize_data(self, composite_data: List[Dict]) -> List[Dict]:
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
                if "memory" in datum:
                    if composite_clbits is not None:
                        sub_data["memory"] = (
                            np.array(datum["memory"])[composite_clbits[i]]
                        ).tolist()
                    else:
                        sub_data["memory"] = datum["memory"]
                marginalized_data[index].append(sub_data)

        # Sort by index
        return [marginalized_data[i] for i in sorted(marginalized_data.keys())]
