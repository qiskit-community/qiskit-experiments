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
Composite Experiment data class.
"""

from typing import Optional, Union, List
from qiskit.result import marginal_counts
from qiskit.exceptions import QiskitError
from qiskit_experiments.framework.experiment_data import ExperimentData
from qiskit_experiments.database_service import DatabaseServiceV1


class CompositeExperimentData(ExperimentData):
    """Composite experiment data class"""

    def __init__(self, experiment, backend=None, parent_id=None, job_ids=None):
        """Initialize experiment data.

        Args:
            experiment (CompositeExperiment): experiment object that generated the data.
            backend (Backend): Optional, Backend the experiment runs on. It can either be a
                :class:`~qiskit.providers.Backend` instance or just backend name.
            parent_id (str): Optional, ID of the parent experiment data
                in the setting of a composite experiment.
            job_ids (list[str]): Optional, IDs of jobs submitted for the experiment.
        """

        super().__init__(experiment, backend=backend, parent_id=parent_id, job_ids=job_ids)

        # Initialize sub experiments
        self._components = [
            expr.__experiment_data__(
                expr, backend=backend, parent_id=self.experiment_id, job_ids=job_ids
            )
            for expr in experiment._experiments
        ]

        self.metadata["component_ids"] = [comp.experiment_id for comp in self._components]
        self.metadata["component_classes"] = [comp.__class__.__name__ for comp in self._components]

    def __str__(self):
        line = 51 * "-"
        n_res = len(self._analysis_results)
        status = self.status()
        ret = line
        ret += f"\nExperiment: {self.experiment_type}"
        ret += f"\nExperiment ID: {self.experiment_id}"
        ret += f"\nStatus: {status}"
        if status == "ERROR":
            ret += "\n  "
            ret += "\n  ".join(self._errors)
        ret += f"\nComponent Experiments: {len(self._components)}"
        ret += f"\nCircuits: {len(self._data)}"
        ret += f"\nAnalysis Results: {n_res}"
        ret += "\n" + line
        if n_res:
            ret += "\nLast Analysis Result:"
            ret += f"\n{str(self._analysis_results.values()[-1])}"
        return ret

    def component_experiment_data(
        self, index: Optional[Union[int, slice]] = None
    ) -> Union[ExperimentData, List[ExperimentData]]:
        """Return component experiment data"""
        if index is None:
            return self._components
        if isinstance(index, (int, slice)):
            return self._components[index]
        raise QiskitError(f"Invalid index type {type(index)}.")

    def _add_single_data(self, data):
        """Add data to the experiment"""
        # TODO: Handle optional marginalizing IQ data
        metadata = data.get("metadata", {})
        if metadata.get("experiment_type") == self._type:

            # Add composite data
            self._data.append(data)

            # Add marginalized data to sub experiments
            if "composite_clbits" in metadata:
                composite_clbits = metadata["composite_clbits"]
            else:
                composite_clbits = None
            for i, index in enumerate(metadata["composite_index"]):
                sub_data = {"metadata": metadata["composite_metadata"][i]}
                if "counts" in data:
                    if composite_clbits is not None:
                        sub_data["counts"] = marginal_counts(data["counts"], composite_clbits[i])
                    else:
                        sub_data["counts"] = data["counts"]

                self._components[index]._jobs = self._jobs
                self._components[index]._add_single_data(sub_data)

    def save(self) -> None:
        super().save()
        for comp in self._components:
            original_verbose = comp.verbose
            comp.verbose = False
            comp.save()
            comp.verbose = original_verbose

    def save_metadata(self) -> None:
        super().save_metadata()
        for comp in self._components:
            comp.save_metadata()

    @classmethod
    def load(cls, experiment_id: str, service: DatabaseServiceV1) -> "CompositeExperimentData":
        expdata = ExperimentData.load(experiment_id, service)
        expdata.__class__ = CompositeExperimentData
        expdata._components = []
        for comp_id, comp_class in zip(
            expdata.metadata["component_ids"], expdata.metadata["component_classes"]
        ):
            load_class = globals()[comp_class]
            load_func = getattr(load_class, "load")
            loaded_comp = load_func(comp_id, service)
            expdata._components.append(loaded_comp)

        return expdata

    def _set_service(self, service: DatabaseServiceV1) -> None:
        """Set the service to be used for storing experiment data.

        Args:
            service: Service to be used.

        Raises:
            DbExperimentDataError: If an experiment service is already being used.
        """
        super()._set_service(service)
        for comp in self._components:
            comp._set_service(service)

    @ExperimentData.share_level.setter
    def share_level(self, new_level: str) -> None:
        """Set the experiment share level.

        Args:
            new_level: New experiment share level. Valid share levels are provider-
                specified. For example, IBM Quantum experiment service allows
                "public", "hub", "group", "project", and "private".
        """
        self._share_level = new_level
        for comp in self._components:
            original_auto_save = comp.auto_save
            comp.auto_save = False
            comp.share_level = new_level
            comp.auto_save = original_auto_save
        if self.auto_save:
            self.save_metadata()

    def _copy_metadata(
        self, new_instance: Optional["CompositeExperimentData"] = None
    ) -> "CompositeExperimentData":
        """Make a copy of the composite experiment metadata.

        Note:
            This method only copies experiment data and metadata, not its
            figures nor analysis results. The copy also contains a different
            experiment ID.

        Returns:
            A copy of the ``CompositeExperimentData`` object with the same data
            and metadata but different ID.
        """
        new_instance = super()._copy_metadata(new_instance)

        for original_comp, new_comp in zip(
            self.component_experiment_data(), new_instance.component_experiment_data()
        ):
            original_comp._copy_metadata(new_comp)

        new_instance.metadata["component_ids"] = [
            comp.experiment_id for comp in new_instance.component_experiment_data()
        ]
        return new_instance
