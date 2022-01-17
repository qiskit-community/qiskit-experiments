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
Correlated readout mitigation calibration analysis classes
"""
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from qiskit.result import CorrelatedReadoutMitigator
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.framework.matplotlib import get_non_gui_ax
from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData, Options


class CorrelatedReadoutErrorAnalysis(BaseAnalysis):
    """
    Measurement correction analysis for a full calibration
    """

    def _run_analysis(
        self, experiment_data: ExperimentData, **options
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        data = experiment_data.data()
        qubits = experiment_data.metadata["physical_qubits"]
        labels = [datum["metadata"]["label"] for datum in data]
        matrix = self._generate_matrix(data, labels)
        result_mitigator = CorrelatedReadoutMitigator(matrix, qubits=qubits)
        analysis_results = [AnalysisResultData("Correlated Readout Mitigator", result_mitigator)]
        ax = options.get("ax", None)
        figures = [self._plot_calibration(matrix, labels, ax)]
        return analysis_results, figures

    def _generate_matrix(self, data, labels) -> np.array:
        list_size = len(labels)
        matrix = np.zeros([list_size, list_size], dtype=float)
        # matrix[i][j] is the probability of counting i for expected j
        for datum in data:
            expected_outcome = datum["metadata"]["label"]
            j = labels.index(expected_outcome)
            total_counts = sum(datum["counts"].values())
            for measured_outcome, count in datum["counts"].items():
                i = labels.index(measured_outcome)
                matrix[i][j] = count / total_counts
        return matrix

    def _plot_calibration(self, matrix, labels, ax=None) -> "matplotlib.figure.Figure":
        """
        Plot the calibration matrix (2D color grid plot).

        Args:
            matrix: calibration matrix to plot
            ax (matplotlib.axes): settings for the graph

        Returns:
            The generated plot of the calibration matrix

        Raises:
            QiskitError: if _cal_matrices was not set.

            ImportError: if matplotlib was not installed.

        """

        if ax is None:
            ax = get_non_gui_ax()
        figure = ax.get_figure()
        ax.matshow(matrix, cmap=plt.cm.binary, clim=[0, 1])
        ax.set_xlabel("Prepared State")
        ax.xaxis.set_label_position("top")
        ax.set_ylabel("Measured State")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        return figure