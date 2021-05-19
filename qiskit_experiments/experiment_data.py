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
Experiment Data class
"""
import logging
from typing import Optional, Union, List, Dict, Tuple
import os
import uuid
from collections import OrderedDict

from qiskit.result import Result
from qiskit.providers import Backend
from qiskit.exceptions import QiskitError
from qiskit.providers import Job, BaseJob
from qiskit.providers.exceptions import JobError

from qiskit_experiments.matplotlib import pyplot, HAS_MATPLOTLIB


LOG = logging.getLogger(__name__)


class AnalysisResult(dict):
    """Placeholder class"""


class ExperimentData:
    """Qiskit Experiments Data container class"""

    def __init__(
        self,
        experiment=None,
        backend=None,
        job_ids=None,
    ):
        """Initialize experiment data.

        Args:
            experiment (BaseExperiment): Optional, experiment object that generated the data.
            backend (Backend): Optional, Backend the experiment runs on.
            job_ids (list[str]): Optional, IDs of jobs submitted for the experiment.

        Raises:
            ExperimentError: If an input argument is invalid.
        """
        # Experiment class object
        self._experiment = experiment

        # Terra ExperimentDataV1 attributes
        self._backend = backend
        self._id = str(uuid.uuid4())
        if experiment is not None:
            self._type = experiment._type
        else:
            self._type = None
        job_ids = job_ids or []
        self._jobs = OrderedDict((k, None) for k in job_ids)
        self._data = []
        self._figures = OrderedDict()
        self._figure_names = []
        self._analysis_results = []

    @property
    def experiment(self):
        """Return Experiment object.

        Returns:
            BaseExperiment: the experiment object.
        """
        return self._experiment

    @property
    def experiment_type(self) -> str:
        """Return the experiment type."""
        return self._type

    @property
    def experiment_id(self) -> str:
        """Return the experiment id."""
        return self._id

    @property
    def job_ids(self) -> List[str]:
        """Return experiment job IDs.

        Returns:
            IDs of jobs submitted for this experiment.
        """
        return list(self._jobs.keys())

    @property
    def backend(self) -> Backend:
        """Return backend.

        Returns:
            Backend this experiment is for.
        """
        return self._backend

    def add_data(
        self,
        data: Union[Result, List[Result], Job, List[Job], Dict, List[Dict]],
    ):
        """Add experiment data.
        Args:
            data: Experiment data to add. Several types are accepted for convenience:

                * Result: Add data from this ``Result`` object.
                * List[Result]: Add data from the ``Result`` objects.
                * Job: Add data from the job result.
                * List[Job]: Add data from the job results.
                * Dict: Add this data.
                * List[Dict]: Add this list of data.

        Raises:
            QiskitError: if data format is invalid.
            KeyboardInterrupt: when job is cancelled by users.
        """
        # Set backend from the job, this could be added to base class
        if isinstance(data, (Job, BaseJob)):
            backend = data.backend()
            if self.backend is not None and str(self.backend) != str(backend):
                LOG.warning(
                    "Adding a job from a backend (%s) that is different than"
                    " the current ExperimentData backend (%s).",
                    backend,
                    self.backend,
                )
            self._backend = backend
            self._jobs[data.job_id()] = data
            self._add_result_data(data.result())
        elif isinstance(data, dict):
            self._add_single_data(data)
        elif isinstance(data, (Job, BaseJob)):
            try:
                result = data.result()
            except JobError as ex:
                if hasattr(data, "error_message"):
                    msg = data.error_message
                else:
                    msg = "Please contact to administrator of your provider."
                raise QiskitError(f"Execution of experiment failed. {msg}") from ex
            except KeyboardInterrupt as ex:
                # remove job from queue list and return the empty result
                data.cancel()
                raise KeyboardInterrupt(ex) from ex
            self._add_result_data(result)
        elif isinstance(data, Result):
            self._add_result_data(data)
        elif isinstance(data, list):
            for dat in data:
                self.add_data(dat)
        else:
            raise QiskitError(f"Invalid data type {type(data)}.")

    def _add_result_data(self, result: Result) -> None:
        """Add data from a Result object

        Args:
            result: Result object containing data to be added.
        """
        num_data = len(result.results)
        for i in range(num_data):
            metadata = result.results[i].header.metadata
            if metadata.get("experiment_type") == self._type:
                data = result.data(i)
                data["metadata"] = metadata
                if "counts" in data:
                    # Format to Counts object rather than hex dict
                    data["counts"] = result.get_counts(i)
                self._add_single_data(data)

    def _add_single_data(self, data: Dict[str, any]) -> None:
        """Add a single data dictionary to the experiment.
        Args:
            data: Data to be added.
        """
        self._data.append(data)

    def data(self, index: Optional[Union[int, slice, str]] = None) -> Union[Dict, List[Dict]]:
        """Return the experiment data at the specified index.

        Args:
            index: Index of the data to be returned.
                Several types are accepted for convenience:

                    * None: Return all experiment data.
                    * int: Specific index of the data.
                    * slice: A list slice of data indexes.
                    * str: ID of the job that produced the data.

        Returns:
            Experiment data.

        Raises:
            QiskitError: if index is invalid.
        """
        if index is None:
            return self._data
        if isinstance(index, (int, slice)):
            return self._data[index]
        if isinstance(index, str):
            return [data for data in self._data if data.get("job_id") == index]
        raise QiskitError(f"Invalid index type {type(index)}.")

    def add_figure(
        self,
        figure,
        figure_name: Optional[str] = None,
        overwrite: bool = False,
    ) -> Tuple[str, int]:
        """Save the experiment figure.

        Args:
            figure (Union[str, bytes, :class:`~matplotlib.figure.Figure`]): Name of the figure file
            or figure data to store. This can either be a ``str`` (for a filename to load),
            ``bytes`` (for the raw image data), or a :class:`~matplotlib.figure.Figure` object.
            figure_name: Name of the figure. If ``None``, use the figure file name, if
                given, or a generated name.
            overwrite: Whether to overwrite the figure if one already exists with
                the same name.

        Returns:
            A tuple of the name and size of the saved figure. Returned size
            is 0 if there is no experiment service to use.

        Raises:
            QiskitError: If the figure with the same name already exists,
                         and `overwrite=True` is not specified.
        """
        if not figure_name:
            if isinstance(figure, str):
                figure_name = figure
            else:
                figure_name = f"figure_{self.experiment_id}_{len(self.figure_names)}"

        existing_figure = figure_name in self._figure_names
        if existing_figure and not overwrite:
            raise QiskitError(
                f"A figure with the name {figure_name} for this experiment "
                f"already exists. Specify overwrite=True if you "
                f"want to overwrite it."
            )
        out = [figure_name, 0]
        self._figures[figure_name] = figure
        self._figure_names.append(figure_name)
        return out

    def figure(self, figure_name: Union[str, int], file_name: Optional[str] = None):
        """Retrieve the specified experiment figure.

        Args:
            figure_name: Name of the figure or figure position.
            file_name: Name of the local file to save the figure to. If ``None``,
                the content of the figure is returned instead.

        Returns:
            Union[int, bytes, :class:`~matplotlib.figure.Figure`]:

                The size of the figure as an ``int`` if ``file_name`` is specified. Otherwise
                the content of the figure as ``bytes`` object or a
                :class:`~matplotlib.figure.Figure` depending on how the image was loaded.

        Raises:
            QiskitError: If the figure cannot be found.
        """
        if isinstance(figure_name, int):
            figure_name = self._figure_names[figure_name]

        figure_data = self._figures.get(figure_name, None)
        if figure_data is not None:
            if isinstance(figure_data, str):
                with open(figure_data, "rb") as file:
                    figure_data = file.read()
            if file_name:
                with open(file_name, "wb") as output:
                    if HAS_MATPLOTLIB and isinstance(figure_data, pyplot.Figure):
                        figure_data.savefig(output, format="svg")
                        num_bytes = os.path.getsize(file_name)
                    else:
                        num_bytes = output.write(figure_data)
                    return num_bytes
            return figure_data
        raise QiskitError(f"Figure {figure_name} not found.")

    @property
    def figure_names(self) -> List[str]:
        """Return names of the figures associated with this experiment.
        Returns:
            Names of figures associated with this experiment.
        """
        return self._figure_names

    def add_analysis_result(self, result: AnalysisResult) -> None:
        """Save the analysis result.
        Args:
            result: Analysis result to be saved.
        """
        self._analysis_results.append(result)

    def analysis_result(
        self, index: Optional[Union[int, slice, str]]
    ) -> Union[AnalysisResult, List[AnalysisResult]]:
        """Return analysis results associated with this experiment.

        Args:
            index: Index of the analysis result to be returned.
                Several types are accepted for convenience:

                    * None: Return all analysis results.
                    * int: Specific index of the analysis results.
                    * slice: A list slice of indexes.
                    * str: ID of the analysis result.

        Returns:
            Analysis results for this experiment.

        Raises:
            QiskitError: if index is invalid.
        """
        if index is None:
            return self._analysis_results
        if isinstance(index, (int, slice)):
            return self._analysis_results[index]
        if isinstance(index, str):
            for res in self._analysis_results:
                if res.id == index:
                    return res
            raise QiskitError(f"Analysis result {index} not found.")
        raise QiskitError(f"Invalid index type {type(index)}.")

    def status(self) -> str:
        """Return the data processing status.

        Returns:
            Data processing status.
        """
        # TODO: Figure out what statuses should be returned including
        # execution and analysis status
        if not self._jobs and not self._data:
            return "EMPTY"
        return "DONE"

    def __str__(self):
        line = 51 * "-"
        n_res = len(self._analysis_results)
        ret = line
        ret += f"\nExperiment: {self.experiment_type}"
        ret += f"\nExperiment ID: {self.experiment_id}"
        ret += f"\nStatus: {self.status()}"
        ret += f"\nCircuits: {len(self._data)}"
        ret += f"\nAnalysis Results: {n_res}"
        ret += "\n" + line
        if n_res:
            ret += "\nLast Analysis Result"
            for key, value in self._analysis_results[-1].items():
                ret += f"\n- {key}: {value}"
        return ret
