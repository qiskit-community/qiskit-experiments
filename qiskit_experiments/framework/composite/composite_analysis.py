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

from typing import List, Dict, Union, Optional, Iterator
import warnings

from qiskit.result import marginal_distribution, marginal_memory
from qiskit_experiments.framework import BaseAnalysis, ExperimentData
from qiskit_experiments.framework.analysis_result_data import AnalysisResultData
from qiskit_experiments.framework.base_analysis import _requires_copy


class CompositeAnalysis(BaseAnalysis):
    """Run analysis for composite experiments.

    Composite experiments consist of several component experiments
    run together in a single execution, the results of which are returned
    as a single list of circuit result data in the :class:`ExperimentData`
    container.

    Analysis of this composite circuit data involves constructing
    a list of experiment data containers for each component experiment containing
    the marginalized circuit result data for that experiment. These are saved as
    :meth:`~.ExperimentData.child_data` in the main :class:`.ExperimentData` container.
    Each component experiment data is then analyzed using the analysis class from
    the corresponding component experiment.

    .. note::

        If the composite :class:`ExperimentData` does not already contain child
        experiment data containers for the component experiments they will be
        initialized and added to the experiment data when
        :meth:`~.CompositeAnalysis.run` is called on the composite data.

        When calling :meth:`~.CompositeAnalysis.run` on experiment data already
        containing initialized component experiment data, any previously stored circuit
        data will be cleared and replaced with the marginalized data from the composite
        experiment data.
    """

    def __init__(self, analyses: List[BaseAnalysis], flatten_results: bool = None):
        """Initialize a composite analysis class.

        Args:
            analyses: a list of component experiment analysis objects.
            flatten_results: If True flatten all component experiment results
                             into a single ExperimentData container, including
                             nested composite experiments. If False save each
                             component experiment results as a separate child
                             ExperimentData container.
        """
        if flatten_results is None:
            # Backward compatibility for 0.6
            # This if-clause will be removed in 0.7 and flatten_result=True is set in arguments.
            warnings.warn(
                "Default value of flatten_results will be turned to True in Qiskit Experiments 0.7. "
                "If you want child experiment data for each subset experiment, "
                "set 'flatten_results=False' explicitly.",
                DeprecationWarning,
            )
            flatten_results = False
        super().__init__()
        self._analyses = analyses
        self._flatten_results = False
        if flatten_results:
            self._set_flatten_results()

    def component_analysis(
        self, index: Optional[int] = None
    ) -> Union[BaseAnalysis, List[BaseAnalysis]]:
        """Return the component experiment Analysis instance.

        Args:
            index: Optional, the component index to return analysis for.
                   If None return a list of all component analysis instances.

        Returns:
            The analysis instance for the specified index, or a list of all
            analysis instances if index is None.
        """
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

        # Run analysis with replace_results = True since we have already
        # created the copy if it was required
        return super().run(experiment_data, replace_results=True, **options)

    def _run_analysis(self, experiment_data: ExperimentData):

        component_exp_data = []
        iter_components = self._initialize_component_experiment_data(experiment_data)
        for i, sub_exp_data in enumerate(iter_components):
            # Since copy for replace result is handled at the parent level
            # we always run with replace result on component analysis
            self._analyses[i].run(sub_exp_data, replace_results=True)
            component_exp_data.append(sub_exp_data)

        # Analysis is running in parallel so we add loop to wait
        # for all component analysis to finish before returning
        # the parent experiment analysis results
        analysis_results = []
        figures = []
        for i, sub_exp_data in enumerate(component_exp_data):
            sub_exp_data.block_for_results()

            if not self._flatten_results:
                experiment_data.add_child_data(sub_exp_data)
                continue

            # Convert table to AnalysisResultData lists for backward compatibility.
            # In principle this is not necessary because data can be directly concatenated to
            # the table of outer container, i.e. experiment_data._analysis_results, however
            # some custom composite analysis class, such as TphiAnalysis overrides
            # the _run_analysis method to perform further analysis on
            # sub-analysis outcomes. This is indeed an overhead,
            # and at some point we should restrict such subclass implementation.
            analysis_table = sub_exp_data.analysis_results(columns="all", dataframe=True)
            for _, series in analysis_table.iterrows():
                data = AnalysisResultData.from_table_element(**series.to_dict())
                data.experiment_id = experiment_data.experiment_id
                analysis_results.append(data)

            for fig_key in sub_exp_data.figure_names:
                figures.append(sub_exp_data.figure(figure_key=fig_key))

            del sub_exp_data

        return analysis_results, figures

    def _marginalize_data(
        self,
        composite_data: List[Dict],
        component_index: int,
    ) -> List[Dict]:
        """Return marginalized data for component with particular index.

        Args:
            composite_data: a list of composite experiment circuit data.
            component_index: an index of component to return.

        Returns:
            A lists of marginalized circuit data for each component
            experiment in the composite experiment.
        """
        out = []
        for datum in composite_data:
            metadata = datum.get("metadata", {})

            if component_index not in metadata["composite_index"]:
                # This circuit is not tied to the component experiment at "component_index".
                continue
            index = metadata["composite_index"].index(component_index)

            if "composite_clbits" in metadata:
                composite_clbits = metadata["composite_clbits"]
            else:
                composite_clbits = None

            component_data = {"metadata": metadata["composite_metadata"][index]}

            # Use terra result marginalization utils.
            # These functions support parallel execution and are implemented in Rust.
            if "counts" in datum:
                if composite_clbits is not None:
                    component_data["counts"] = marginal_distribution(
                        counts=datum["counts"],
                        indices=composite_clbits[index],
                    )
                else:
                    component_data["counts"] = datum["counts"]
            if "memory" in datum:
                if composite_clbits is not None:
                    component_data["memory"] = marginal_memory(
                        memory=datum["memory"],
                        indices=composite_clbits[index],
                    )
                else:
                    component_data["memory"] = datum["memory"]
            out.append(component_data)
        return out

    def _initialize_component_experiment_data(
        self,
        experiment_data: ExperimentData,
    ) -> Iterator[ExperimentData]:
        """Initialize empty experiment data containers for component experiments.

        Args:
            experiment_data: a composite experiment data container.

        Yields:
            Experiment data containers for each component experiment
            containing the component metadata, and tags, share level.
        """
        metadata = experiment_data.metadata

        # Extract component experiment types and metadata so they can be
        # added to the component experiment data containers
        num_components = len(self._analyses)
        experiment_types = metadata.get("component_types", [None] * num_components)
        component_metadata = metadata.get("component_metadata", [{}] * num_components)

        # Create component experiments and set the backend and
        # metadata for the components
        composite_data = experiment_data.data()
        child_data_ids = []
        for i in range(num_components):
            # Create empty container with metadata
            sub_exp_data = ExperimentData(backend=experiment_data.backend)
            sub_exp_data.experiment_type = experiment_types[i]
            sub_exp_data.metadata.update(component_metadata[i])
            sub_exp_data.auto_save = False

            # Add marginalized experiment data
            sub_exp_data.add_data(self._marginalize_data(composite_data, i))
            child_data_ids.append(sub_exp_data.experiment_id)

            yield sub_exp_data

    def _set_flatten_results(self):
        """Recursively set flatten_results to True for all composite components."""
        self._flatten_results = True
        for analysis in self._analyses:
            if isinstance(analysis, CompositeAnalysis):
                analysis._set_flatten_results()
