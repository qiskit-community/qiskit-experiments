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

from typing import List, Dict, Union, Optional, Callable
import numpy as np
from qiskit.result import marginal_counts
from qiskit_experiments.framework import BaseAnalysis, ExperimentData
from qiskit_experiments.framework.base_analysis import _requires_copy
from qiskit_experiments.exceptions import AnalysisError


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

        If the child :class:`ExperimentData` for each component experiment
        does not already exist in the experiment data they will be initialized
        and added to the experiment data when :meth:`run` is called on the
        composite :class:`ExperimentData`.

        When calling :meth:`run` on experiment data already containing
        initialized component experiment child data, any previously stored
        circuit data will be cleared and replaced with the marginalized data
        reconstructed from the parent composite experiment data.
    """

    def __init__(self, analyses: List[BaseAnalysis]):
        """Initialize a composite analysis class.

        Args:
            analyses: a list of component experiment analysis objects.
        """
        super().__init__()
        self._analyses = analyses

    def component_analysis(self, index=None) -> Union[BaseAnalysis, List[BaseAnalysis]]:
        """Return the component experiment Analysis object"""
        if index is None:
            return self._analyses
        return self._analyses[index]

    def copy(self):
        ret = super().copy()
        # Recursively copy analysis
        ret._analyses = [analysis.copy() for analysis in ret._analyses]
        return ret

    def run(
        self,
        experiment_data: ExperimentData,
        replace_results: bool = False,
        **options,
    ) -> ExperimentData:
        # Make a new copy of experiment data if not updating results
        if not replace_results and _requires_copy(experiment_data):
            experiment_data = experiment_data.copy()

        # Initialize child components if they are not initalized.
        self._initialize_child_data(experiment_data)

        # Run analysis with replace_results = True since we have already
        # created the copy if it was required
        return super().run(experiment_data, replace_results=True, **options)

    def _run_analysis(self, experiment_data: ExperimentData):
        # Return list of experiment data containers for each component experiment
        # containing the marginalied data from the composite experiment
        component_expdata = self._component_experiment_data(experiment_data)

        # Run the component analysis on each component data
        for i, sub_expdata in enumerate(component_expdata):
            # Since copy for replace result is handled at the parent level
            # we always run with replace result on component analysis
            self._analyses[i].run(sub_expdata, replace_results=True)

        # Analysis is running in parallel so we add loop to wait
        # for all component analysis to finish before returning
        # the parent experiment analysis results
        for sub_expdata in component_expdata:
            sub_expdata.block_for_results()

        return [], []

    def _component_experiment_data(self, experiment_data: ExperimentData) -> List[ExperimentData]:
        """Return a list of component child experiment data"""
        # Retrieve or initialize the component data for updating
        component_index = experiment_data.metadata.get("component_child_index", [])
        if not component_index:
            raise AnalysisError("Unable to extract component child experiment data")
        component_expdata = [experiment_data.child_data(i) for i in component_index]

        # Compute marginalize data for each component experiment
        marginalized_data = self._marginalized_component_data(experiment_data.data())

        # Add the marginalized component data and component job metadata
        # to each component child experiment. Note that this will clear
        # any currently stored data in the experiment. Since copying of
        # child data is handled by the `replace_results` kwarg of the
        # parent container it is safe to always clear and replace the
        # results of child containers in this step
        for sub_expdata, sub_data in zip(component_expdata, marginalized_data):
            # Clear any previously stored data and add marginalized data
            sub_expdata._data.clear()
            sub_expdata.add_data(sub_data)

        return component_expdata

    def _marginalized_component_data(self, composite_data: List[Dict]) -> List[List[Dict]]:
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

    def _initialize_child_data(self, experiment_data: ExperimentData):
        """Initialize component experiment data objects as child data"""
        component_index = experiment_data.metadata.get("component_child_index", [])
        if component_index:
            # Child components are already initialized
            return

        # Initialize the component experiment data containers and add them
        # as child data to the current experiment data
        child_components = self._initialize_component_data(experiment_data)
        start_index = len(experiment_data.child_data())
        for i, subdata in enumerate(child_components):
            experiment_data.add_child_data(subdata)
            component_index.append(start_index + i)

        # Store the indices of the added child data in metadata
        experiment_data.metadata["component_child_index"] = component_index

    def _initialize_component_data(self, experiment_data: ExperimentData) -> List[ExperimentData]:
        """Initialize component experiment data objects.

        These contain the component metadata, and copy the tags, share level,
        and auto save attributes of the main data.
        """
        # Extract component experiment types and metadata so they can be
        # added to the component experiment data containers
        metadata = experiment_data.metadata
        num_components = len(self._analyses)
        experiment_types = metadata.get("component_types", [None] * num_components)
        component_metadata = metadata.get("component_metadata", [{}] * num_components)

        # Create component experiments and copy backend, tags, share level
        # and auto save from the parent experiment data
        component_expdata = []
        for i, _ in enumerate(self._analyses):
            subdata = ExperimentData(backend=experiment_data.backend)
            subdata._type = experiment_types[i]
            subdata.metadata.update(component_metadata[i])
            subdata.tags = experiment_data.tags
            subdata.share_level = experiment_data.share_level
            subdata.auto_save = experiment_data.auto_save
            component_expdata.append(subdata)

        return component_expdata
