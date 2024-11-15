# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Analysis class to characterize correlated readout error
"""
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from qiskit_experiments.data_processing import CorrelatedReadoutMitigator
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.framework.matplotlib import get_non_gui_ax
from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData, Options


class CorrelatedReadoutErrorAnalysis(BaseAnalysis):
    r"""An analysis to characterize correlated readout error.

    # section: overview

        This class generates the full assignment matrix :math:`A` characterizing the
        readout error for the given qubits from the experiment results
        and returns the resulting :class:`~qiskit.result.CorrelatedReadoutMitigator`

        :math:`A` is a :math:`2^n\times 2^n` matrix :math:`A` such that :math:`A_{y,x}`
        is the probability to observe :math:`y` given the true outcome should be :math:`x`.

        In the experiment, for each :math:`x` a circuit is constructed whose expected
        outcome is :math:`x`. From the observed results on the circuit, the probability for
        each :math:`y` is determined, and :math:`A_{y,x}` is set accordingly.

        Analysis Results:
           * "Local Readout Mitigator": The :class:`~qiskit.result.LocalReadoutMitigator`.

        Analysis Figures:
           * (Optional) A figure of the assignment matrix.

    # section: reference
        .. ref_arxiv:: 1 2006.14044
    """

    @classmethod
    def _default_options(cls) -> Options:
        """Return default analysis options.

        Analysis Options:
            plot (bool): Set ``True`` to create figure for fit result.
            ax (AxesSubplot): Optional. A matplotlib axis object to draw.
        """
        options = super()._default_options()
        options.plot = True
        options.ax = None
        return options

    def _run_analysis(
        self, experiment_data: ExperimentData, **options
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        data = experiment_data.data()
        qubits = experiment_data.metadata["physical_qubits"]
        labels = [datum["metadata"]["state_label"] for datum in data]
        matrix = self._generate_matrix(data, labels)
        result_mitigator = CorrelatedReadoutMitigator(matrix, qubits=qubits)
        analysis_results = [AnalysisResultData("Correlated Readout Mitigator", result_mitigator)]
        if self.options.plot:
            ax = options.get("ax", None)
            figures = [self._assignment_matrix_visualization(matrix, labels, ax)]
        else:
            figures = []
        return analysis_results, figures

    def _generate_matrix(self, data, labels) -> np.array:
        list_size = len(labels)
        matrix = np.zeros([list_size, list_size], dtype=float)
        # matrix[i][j] is the probability of counting i for expected j
        for datum in data:
            expected_outcome = datum["metadata"]["state_label"]
            j = labels.index(expected_outcome)
            total_counts = sum(datum["counts"].values())
            for measured_outcome, count in datum["counts"].items():
                i = labels.index(measured_outcome)
                matrix[i][j] = count / total_counts
        return matrix

    def _assignment_matrix_visualization(
        self, matrix, labels, ax=None
    ) -> "matplotlib.figure.Figure":
        """
        Plot the assignment matrix (2D color grid plot).

        Args:
            matrix: assignment matrix to plot
            ax (matplotlib.axes): settings for the graph

        Returns:
            The generated plot of the assignment matrix

        Raises:
            QiskitError: If _cal_matrices was not set.

            ImportError: If matplotlib was not installed.

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
