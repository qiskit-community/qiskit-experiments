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

import uuid

from qiskit.result import Result
from qiskit.providers import Job, BaseJob
from qiskit.exceptions import QiskitError


class AnalysisResult(dict):
    """Placeholder class"""


class ExperimentData:
    """ExperimentData container class"""

    def __init__(self, experiment):
        """Initialize the analysis object.

        Args:
            experiment (BaseExperiment): experiment object that
                                         generated the data.
        """
        # Experiment identification metadata
        self._id = str(uuid.uuid4())
        self._experiment = experiment

        # Experiment Data
        self._data = []

        # Figures
        self._figures = []

        # Analysis
        self._analysis_results = []

    def __repr__(self):
        line = 51 * "-"
        n_res = len(self._analysis_results)
        ret = line
        ret += f"\nExperiment: {self._experiment._type}"
        ret += f"\nExperiment ID: {self.experiment_id}"
        ret += "\nStatus: COMPLETE"
        ret += f"\nCircuits: {len(self._data)}"
        ret += f"\nAnalysis Results: {n_res}"
        ret += "\n" + line
        if n_res:
            ret += "\nLast Analysis Result"
            for key, value in self._analysis_results[-1].items():
                ret += f"\n- {key}: {value}"
        return ret

    @property
    def experiment_id(self):
        """Return the experiment id"""
        return self._id

    def experiment(self):
        """Return Experiment object"""
        return self._experiment

    def analysis_result(self, index):
        """Return stored analysis results

        Args:
            index (int or slice): the result or range of results to return.

        Returns:
            AnalysisResult: the result for an integer index.
            List[AnalysisResult]: a list of results for slice index.
        """
        return self._analysis_results[index]

    def add_analysis_result(self, result):
        """Add an Analysis Result

        Args:
            result (AnalysisResult): the analysis result to add.
        """
        self._analysis_results.append(result)

    @property
    def data(self):
        """Return stored experiment data"""
        return self._data

    @property
    def figures(self):
        """Return the figures."""
        return self._figures

    def add_figure(self, figure):
        """Add a figure to the experiment data."""
        self._figures.append(figure)

    def add_data(self, data):
        """Add data to the experiment.

        Args:
            data (Result or Job or dict or list): the circuit execution data
                to add. This can be a Result, Job, or dict object, or a list
                of Result, Job, or dict objects.

        Raises:
            QiskitError: if the data is not a valid format.
        """
        if isinstance(data, dict):
            self._add_single_data(data)
        elif isinstance(data, Result):
            self._add_result_data(data)
        elif isinstance(data, (Job, BaseJob)):
            self._add_result_data(data.result())
        elif isinstance(data, list):
            for dat in data:
                self.add_data(dat)
        else:
            raise QiskitError("Invalid data format.")

    def _add_result_data(self, result: Result):
        """Add data from qiskit Result object"""
        num_data = len(result.results)
        for i in range(num_data):
            metadata = result.results[i].header.metadata
            if metadata.get("experiment_type") == self._experiment._type:
                data = result.data(i)
                data["metadata"] = metadata
                if "counts" in data:
                    # Format to Counts object rather than hex dict
                    data["counts"] = result.get_counts(i)
                self._add_single_data(data)

    def _add_single_data(self, data):
        """Add a single data dictionary to the experiment.

        Args:
            data (dict): a data dictionary for a single circuit exection.
        """
        # This method is intended to be overriden by subclasses when necessary.
        if data.get("metadata", {}).get("experiment_type") == self._experiment._type:
            self._data.append(data)
