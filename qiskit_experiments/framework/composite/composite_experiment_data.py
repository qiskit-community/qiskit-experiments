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
from qiskit_experiments.database_service import DbExperimentDataV1, DatabaseServiceV1


class CompositeExperimentData(ExperimentData):
    """Composite experiment data class"""

    def __init__(
        self, experiment, backend=None, job_ids=None, parent_id=None, root_id=None, components=None
    ):
        """Initialize experiment data.

        Args:
            experiment (CompositeExperiment): experiment object that generated the data.
            backend (Backend): Optional, Backend the experiment runs on. It can either be a
                :class:`~qiskit.providers.Backend` instance or just backend name.
            job_ids (list[str]): Optional, IDs of jobs submitted for the experiment.
            parent_id (str): Optional, ID of the parent experiment data in a composite experiment
            root_id (str): Optional, ID of the root experiment data in a composite experiment
            components (list[ExperimentData]): Optional, a list of already prepared experiment
                data objects of the components. Applicable only if ``experiment`` is ``None``,
                otherwise the components are created from the experiment's components.

        Raises:
            ValueError: If both ``experiment`` and ``components`` are not ``None``.
        """

        if experiment is not None and components is not None:
            raise ValueError(
                "CompositeExperimentData initialization does not accept experiment "
                "and component parameters that are both not None"
            )

        super().__init__(
            experiment, backend=backend, job_ids=job_ids, parent_id=parent_id, root_id=root_id
        )

        # In a composite setting, an experiment is tagged with its direct parent and with the root.
        # This is done in the ExperimentData constructir, except for the root experiment,
        # for whom this is done here
        root_id = root_id if root_id is not None else self.experiment_id
        if root_id not in self.tags:
            self.tags.append(root_id)

        # Initialize sub experiments
        if components:
            self._components = components
        else:
            self._components = [
                expr.__experiment_data__(expr, backend, job_ids, self.experiment_id, root_id)
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

            # Add parallel data
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
                self._components[index].add_data(sub_data)

    def save(self) -> None:
        super().save()
        for comp in self._components:
            comp.save()

    def save_metadata(self) -> None:
        super().save_metadata()
        for comp in self._components:
            comp.save_metadata()

    @classmethod
    def load(cls, experiment_id: str, service: DatabaseServiceV1) -> "CompositeExperimentData":
        expdata1 = DbExperimentDataV1.load(experiment_id, service)
        components = []
        for comp_id, comp_class in zip(
            expdata1.metadata["component_ids"], expdata1.metadata["component_classes"]
        ):
            load_class = globals()[comp_class]
            load_func = getattr(load_class, "load")
            components.append(load_func(experiment_id, service))

        expdata2 = CompositeExperimentData(
            experiment=None,
            backend=expdata1.backend,
            job_ids=expdata1.job_ids,
            components=components,
        )

        return expdata2

    def _set_service(self, service: DatabaseServiceV1) -> None:
        """Set the service to be used for storing experiment data.

        Args:
            service: Service to be used.

        Raises:
            DbExperimentDataError: If an experiment service is already being used.
        """
        DbExperimentDataV1._set_service(self, service)
        for comp in self._components:
            comp._set_service(service)
