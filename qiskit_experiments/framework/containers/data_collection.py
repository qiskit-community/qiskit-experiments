# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from functools import singledispatchmethod
from typing import Any

import pandas as pd
from qiskit.result import Result

from qiskit_experiments.database_service.device_component import DeviceComponent
from qiskit_experiments.database_service.exceptions import ExperimentEntryNotFound
from qiskit_experiments.database_service.utils import ThreadSafeOrderedDict, ThreadSafeList
from qiskit_experiments.framework.analysis_result_table import AnalysisResultTable
from qiskit_experiments.framework.artifact_data import ArtifactData
from .elements import _FigureT, FigureData, CanonicalResult

LOG = logging.getLogger(__name__)


class DataCollection:
    """Collection of multiple data kinds that experiment produces through execution.

    Running an experiment produces following data:

    * Analysis results: Data that the analysis produces by digesting the job results.
    * Figures: Visual data that helps human experimentalist with understanding the result.
    * Artifacts: Supplemental data.
    """

    def __init__(
        self,
        experiment_id: str | None = None,
        experiment_type: str | None = None,
        backend_name: str | None = None,
        child_data: list["DataCollection"] | None = None,
        parent_id: str | None = None,
    ):
        self.experiment_type: str = experiment_type
        self.backend_name: str = backend_name
        self.experiment_id: str = experiment_id or str(uuid.uuid4())
        self.metadata: dict[str, Any] = {
            "child_data_ids": [],
        }

        # Data storage
        self._result_data: ThreadSafeList = ThreadSafeList()
        self._figures: ThreadSafeOrderedDict = ThreadSafeOrderedDict()
        self._analysis_results: AnalysisResultTable = AnalysisResultTable()
        self._artifacts: ThreadSafeOrderedDict = ThreadSafeOrderedDict()

        # Child data containers
        self.parent_id: str | None = parent_id
        self._child_data: ThreadSafeOrderedDict = ThreadSafeOrderedDict()
        if child_data is not None:
            self._set_child_data(child_data)

    @property
    def figure_names(self) -> list[str]:
        """All figure names stored in this experiment data."""
        return list(self._figures.keys())

    def add_data(
        self,
        data: Result | list[Result] | dict | list[dict],
    ):
        """Add canonical experiment result.

        Args:
            data: Experiment result to add.
        """
        if not isinstance(data, list):
            data = [data]
        for datum in data:
            self._add_single_data_dispatch(datum)

    @singledispatchmethod
    def _add_single_data_dispatch(
        self,
        data,
    ):
        """Save single result data."""
        raise TypeError(f"Invalid experiment result data type {type(data)}.")

    @_add_single_data_dispatch.register
    def _add_dict_data(
        self,
        data: dict,
    ):
        """Save single canonical result."""
        self._result_data.append(CanonicalResult(**data))

    @_add_single_data_dispatch.register
    def _add_result_object_data(
        self,
        data: Result,
    ):
        """Format Qiskit Result object into experiment canonical result and save."""
        for i in range(len(data.results)):
            expr_result = data.results[i]
            header = data.header
            self._result_data.append(
                CanonicalResult(
                    job_id=data.job_id,
                    metadata=getattr(header, "metadata", None),
                    shots=expr_result.shots,
                    meas_level=expr_result.meas_level,
                    meas_return=getattr(expr_result, "meas_return", None),
                    creg_sizes=getattr(header, "creg_sizes", None),
                    memory_slots=getattr(header, "memory_slots", None),
                    **data.data(i),
                )
            )

    def data(
        self,
        index: int | slice | str | None = None,
    ) -> CanonicalResult | list[CanonicalResult]:
        """Return the experiment result data at the specified index.

        Args:
            index: Index of the data to be returned.
                Several types are accepted for convenience:

                    * None: Return all experiment data.
                    * int: Specific index of the data.
                    * slice: A list slice of data indexes.
                    * str: ID of the job that produced the data.

        Returns:
            Experiment result data.

        Raises:
            TypeError: If the input `index` has an invalid type.
        """
        # self._retrieve_data()
        if index is None:
            return self._result_data.copy()
        if isinstance(index, (int, slice)):
            return self._result_data[index]
        if isinstance(index, str):
            return [d for d in self._result_data if d.get("job_id", None) == index]
        raise TypeError(f"Invalid index type {type(index)}.")

    def add_figures(
        self,
        figures: _FigureT | list[_FigureT],
        figure_names: str | list[str] | None = None,
        overwrite: bool = False,
    ) -> list[str]:
        """Add experiment data plots.

        Args:
            figures: Figure data. This can be either a path to figure files or figure object
                in matplotlib figure or any binary data, or list of them.
            figure_names: Names of the figures. If ``None``, use the figure file
                names, if given, or a generated name of the format ``experiment_type``, figure
                index, first 5 elements of ``device_components``, and first 8 digits of the
                experiment ID connected by underscores, such as ``T1_Q0_0123abcd.svg``. If `figures`
                is a list, then `figure_names` must also be a list of the same length or ``None``.
            overwrite: Whether to overwrite the figure if one already exists with
                the same name. By default, overwrite is ``False`` and the figure will be renamed
                with an incrementing numerical suffix. For example, trying to save ``figure.svg`` when
                ``figure.svg`` already exists will save it as ``figure-1.svg``, and trying to save
                ``figure-1.svg`` when ``figure-1.svg`` already exists will save it as ``figure-2.svg``.

        Returns:
            Figure names with extension of ``.svg``.

        Raises:
            ValueError: If an input parameter has an invalid value.
        """
        if figure_names is not None and not isinstance(figure_names, list):
            figure_names = [figure_names]
        if not isinstance(figures, list):
            figures = [figures]
        if figure_names is not None and len(figures) != len(figure_names):
            raise ValueError(
                "The parameter figure_names must be None or a list of "
                "the same size as the parameter figures."
            )

        formatted_figure_names = []
        for idx, figure in enumerate(figures):
            if figure_names is None:
                if isinstance(figure, str):
                    # figure is a filename, so we use it as the name
                    fig_name = figure
                elif not isinstance(figure, FigureData):
                    # Generate a name in the form StandardRB_Q0_Q1_Q2_b4f1d8ad-1.svg
                    fig_name = (
                        f"{self.experiment_type}_"
                        f'{"_".join(str(i) for i in self.metadata.get("device_components", [])[:5])}_'
                        f"{self.experiment_id[:8]}.svg"
                    )
                else:
                    # Keep the existing figure name if there is one
                    fig_name = figure.name
            else:
                fig_name = figure_names[idx]
            if not fig_name.endswith(".svg"):
                LOG.info("File name %s does not have an SVG extension. A '.svg' is added.")
                fig_name += ".svg"

            is_existing = fig_name in self._figures
            if is_existing and not overwrite:
                # Remove any existing suffixes then generate new figure name
                # StandardRB_Q0_Q1_Q2_b4f1d8ad.svg becomes StandardRB_Q0_Q1_Q2_b4f1d8ad
                fig_name_chunked = fig_name.rsplit("-", 1)
                if len(fig_name_chunked) != 1:  # Figure name already has a suffix
                    # This extracts StandardRB_Q0_Q1_Q2_b4f1d8ad as the prefix from
                    # StandardRB_Q0_Q1_Q2_b4f1d8ad-1.svg
                    fig_name_prefix = fig_name_chunked[0]
                    try:
                        fig_name_suffix = int(fig_name_chunked[1].rsplit(".", 1)[0])
                    except ValueError:  # the suffix is not an int, add our own suffix
                        # my-custom-figure-name will be the prefix of my-custom-figure-name.svg
                        fig_name_prefix = fig_name.rsplit(".", 1)[0]
                        fig_name_suffix = 0
                else:
                    # StandardRB_Q0_Q1_Q2_b4f1d8ad.svg has no hyphens so
                    # StandardRB_Q0_Q1_Q2_b4f1d8ad would be its prefix
                    fig_name_prefix = fig_name.rsplit(".", 1)[0]
                    fig_name_suffix = 0
                fig_name = f"{fig_name_prefix}-{fig_name_suffix + 1}.svg"
                while fig_name in self._figures:  # Increment suffix until the name isn't taken
                    # If StandardRB_Q0_Q1_Q2_b4f1d8ad-1.svg already exists,
                    # StandardRB_Q0_Q1_Q2_b4f1d8ad-2.svg will be the name of this figure
                    fig_name_suffix += 1
                    fig_name = f"{fig_name_prefix}-{fig_name_suffix + 1}.svg"

            # figure_data = None
            if isinstance(figure, str):
                with open(figure, "rb") as file:
                    figure = file.read()

            # check whether the figure is already wrapped, meaning it came from a sub-experiment
            if isinstance(figure, FigureData):
                figure_data = figure.copy(new_name=fig_name)
            else:
                figure_metadata = {
                    "qubits": self.metadata.get("physical_qubits"),
                    "device_components": self.metadata.get("device_components"),
                    "experiment_type": self.experiment_type,
                }
                figure_data = FigureData(figure=figure, name=fig_name, metadata=figure_metadata)

            self._figures[fig_name] = figure_data
            formatted_figure_names.append(fig_name)

        return formatted_figure_names

    def delete_figure(
        self,
        figure_key: str | int,
    ) -> str:
        """Delete specified experiment data plot.

        Args:
            figure_key: Name or index of the figure.

        Returns:
            Deleted figure name.
        """
        figure_key = self._find_figure_key(figure_key)
        del self._figures[figure_key]

        return figure_key

    def figure(
        self,
        figure_key: str | int,
        file_name: str | None = None,
    ) -> int | FigureData:
        """Return the specified experiment figure.

        Args:
            figure_key: Name or index of the figure.
            file_name: Name of the local file to save the figure to. If ``None``,
                the content of the figure is returned instead.

        Returns:
            The size of the figure if `file_name` is specified. Otherwise, the
            content of the figure as a :class:`.FigureData` object.

        Raises:
            ExperimentEntryNotFound: If the figure cannot be found.
        """
        figure_key = self._find_figure_key(figure_key)

        figure_data = self._figures[figure_key]
        if figure_data is None:
            raise ExperimentEntryNotFound(f"Figure {figure_key} not found.")

        if file_name:
            with open(file_name, "wb") as output:
                num_bytes = output.write(figure_data.figure)
                return num_bytes
        return figure_data

    def _find_figure_key(
        self,
        figure_key: int | str,
    ) -> str:
        """A helper method to find figure key."""
        if isinstance(figure_key, int):
            if figure_key < 0 or figure_key >= len(self._figures):
                raise ExperimentEntryNotFound(f"Figure index {figure_key} out of range.")
            return self._figures.keys()[figure_key]

        if figure_key not in self._figures:
            raise ExperimentEntryNotFound(f"Figure key {figure_key} not found.")
        return figure_key

    def add_artifacts(
        self,
        artifacts: ArtifactData | list[ArtifactData],
    ):
        """Add artifacts of experiment.

        Args:
            artifacts: Artifact data to be added.
        """
        if isinstance(artifacts, ArtifactData):
            artifacts = [artifacts]

        for artifact in artifacts:
            self._artifacts[artifact.artifact_id] = artifact

    def delete_artifact(
        self,
        artifact_key: int | str,
    ) -> str | list[str]:
        """Delete specified artifact data.

        Args:
            artifact_key: UID, name or index of the figure.

        Returns:
            Deleted artifact ids.
        """
        artifact_keys = self._find_artifact_keys(artifact_key)

        for key in artifact_keys:
            del self._artifacts[key]

        if len(artifact_keys) == 1:
            return artifact_keys[0]
        return artifact_keys

    def artifacts(
        self,
        artifact_key: int | str,
    ) -> ArtifactData | list[ArtifactData]:
        """Return specified artifact data.

        Args:
            artifact_key: UID, name or index of the figure.

        Returns:
            A list of specified artifact data.
        """
        artifact_keys = self._find_artifact_keys(artifact_key)

        out = []
        for key in artifact_keys:
            artifact_data = self._artifacts[key]
            if artifact_data is None:
                continue
            out.append(artifact_data)

        if len(out) == 1:
            return out[0]
        return out

    def _find_artifact_keys(
        self,
        artifact_key: int | str,
    ) -> list[str]:
        """A helper method to find artifact key."""
        if isinstance(artifact_key, int):
            if artifact_key < 0 or artifact_key >= len(self._artifacts):
                raise ExperimentEntryNotFound(f"Artifact index {artifact_key} out of range.")
            return [self._artifacts.keys()[artifact_key]]

        if artifact_key not in self._artifacts:
            name_mathed = [k for k, d in self._artifacts.items() if d.name == artifact_key]
            if len(name_mathed) == 0:
                raise ExperimentEntryNotFound(f"Artifact key {artifact_key} not found.")
            return name_mathed
        return [artifact_key]

    def add_analysis_results(
        self,
        *,
        name: str | None = None,
        value: Any | None = None,
        quality: str | None = None,
        components: list[DeviceComponent] | None = None,
        experiment: str | None = None,
        experiment_id: str | None = None,
        result_id: str | None = None,
        tags: list[str] | None = None,
        backend: str | None = None,
        run_time: datetime | None = None,
        created_time: datetime | None = None,
        **extra_values,
    ):
        """Save the analysis result.

        Args:
            name: Name of the result entry.
            value: Analyzed quantity.
            quality: Quality of the data.
            components: Associated device components.
            experiment: String identifier of the associated experiment.
            experiment_id: ID of the associated experiment.
            result_id: ID of this analysis entry. If not set a random UUID is generated.
            tags: List of arbitrary tags.
            backend: Name of associated backend.
            run_time: The date time when the experiment started to run on the device.
            created_time: The date time when this analysis is performed.
            extra_values: Arbitrary keyword arguments for supplementary information.
                New dataframe columns are created in the analysis result table with added keys.
        """
        experiment = experiment or self.experiment_type
        experiment_id = experiment_id or self.experiment_id
        tags = tags or []
        backend = backend or self.backend_name

        self._analysis_results.add_entry(
            result_id=result_id,
            name=name,
            value=value,
            quality=quality,
            components=components,
            experiment=experiment,
            experiment_id=experiment_id,
            tags=tags,
            backend=backend,
            run_time=run_time,
            created_time=created_time,
            **extra_values,
        )

    def delete_analysis_result(
        self,
        result_key: int | str,
    ) -> str:
        """Delete the analysis result.

        Args:
            result_key: ID or index of the analysis result to be deleted.

        Returns:
            Analysis result ID.
        """
        result_ids = self._find_result_key(result_key)
        if len(result_ids) > 1:
            raise ExperimentEntryNotFound(
                f"Multiple entries are found with result_key = {result_key}. "
                "Try another key that can uniquely determine entry to delete."
            )
        self._analysis_results.drop_entry(result_ids[0])
        return result_ids[0]

    def analysis_results(
        self,
        index: int | slice | str | None = None,
        columns: str | list[str] = "default",
    ) -> pd.DataFrame | pd.Series:
        """Return specified analysis results.

        Args:
            index: Index of the analysis result to be returned.
                Several types are accepted for convenience:

                    * None: Return all analysis results.
                    * int: Specific index of the analysis results.
                    * slice: A list slice of indexes.
                    * str: ID or name of the analysis result.

            columns: Specifying a set of columns to return. You can pass a list of each
                column name to return, otherwise builtin column groups are available.

                    * "all": Return all columns, including metadata to communicate
                        with experiment service, such as entry IDs.
                    * "default": Return columns including analysis result with supplementary
                        information about experiment.
                    * "minimal": Return only analysis subroutine returns.

        Returns:
            Matched analysis results in the table format.
        """
        out = self._analysis_results.copy()

        if index is not None:
            result_ids = self._find_result_key(index)
            out = out[out.index.isin(result_ids)]

        valid_columns = self._analysis_results.filter_columns(columns)
        out = out[valid_columns]

        if len(out) == 1 and index is not None:
            # For backward compatibility.
            # One can directly access attributes with Series. e.g. out.value
            return out.iloc[0]
        return out

    def _find_result_key(
        self,
        index: int | slice | str,
    ) -> list[str]:
        """A helper method to find analysis result table index."""
        dataframe = self._analysis_results.copy()
        if isinstance(index, int):
            if index < 0 or index >= len(self._analysis_results):
                raise ExperimentEntryNotFound(f"Analysis result index {index} out of range.")
            return [dataframe.iloc[index].name]
        if isinstance(index, slice):
            out = dataframe[slice]
            if len(out) == 0:
                raise ExperimentEntryNotFound(f"Analysis result slice {index} out of range.")
            return out.index.to_list()
        if isinstance(index, str):
            if index not in dataframe.index:
                name_matched = dataframe[dataframe["name"] == index]
                if len(name_matched) == 0:
                    raise ExperimentEntryNotFound(f"Analysis result key {index} not found.")
                return name_matched
        return [index]

    def _set_child_data(
        self,
        child_data: list[DataCollection],
    ):
        """Initialize the child data container."""
        self._child_data = ThreadSafeOrderedDict()
        for datum in child_data:
            self.add_child_data(datum)

    def add_child_data(
        self,
        experiment_data: DataCollection,
    ):
        """Add child experiment data to the current experiment data.

        Args:
            experiment_data: Child data to add.
        """
        experiment_data.parent_id = self.experiment_id
        child_id = experiment_data.experiment_id
        self._child_data[child_id] = experiment_data
        self.metadata["child_data_ids"].append(child_id)

    def child_data(
        self,
        index: int | slice | str | None = None,
    ) -> "DataCollection" | list["DataCollection"]:
        """Return child data container.

        Args:
            index: Index of the child experiment data to be returned.
                Several types are accepted for convenience:

                    * None: Return all child data.
                    * int: Specific index of the child data.
                    * slice: A list slice of indexes.
                    * str: experiment ID of the child data.

        Returns:
            The requested single or list of child experiment data.
        """
        if index is None:
            return list(self._child_data.values())
        out = []
        for key in self._find_child_key(index):
            out.append(self._child_data[key])
        return out

    def _find_child_key(
        self,
        index: int | slice | str,
    ) -> list[str]:
        """A helper method to find child data container index."""
        if isinstance(index, slice):
            return list(self._child_data.keys())[index]
        if isinstance(index, int):
            if index < 0 or index >= len(self._child_data):
                raise ExperimentEntryNotFound(f"Child data index {index} out of range.")
            return [self._child_data.keys()[index]]
        if index not in self._child_data:
            raise ExperimentEntryNotFound(f"Child data UID {index} not found.")
        return [index]

    def _clear_results(self):
        """Delete all currently stored results."""
        self._analysis_results.clear()
        self._figures = ThreadSafeOrderedDict()
        self._artifacts = ThreadSafeOrderedDict()
