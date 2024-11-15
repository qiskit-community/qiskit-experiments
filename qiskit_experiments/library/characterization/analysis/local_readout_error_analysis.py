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
Analysis class to characterize local readout error
"""
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from qiskit.result import marginal_distribution
from qiskit_experiments.data_processing import LocalReadoutMitigator
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.framework.matplotlib import get_non_gui_ax
from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData, Options


class LocalReadoutErrorAnalysis(BaseAnalysis):
    r"""
    Local readout error characterization analysis
    # section: overview

       This class generates the assignment matrices characterizing the
       readout error for each of the given qubits from the experiment result,
       and returns the resulting :class:`~qiskit.result.LocalReadoutMitigator`

       Each such matrix is a :math:`2\times 2` matrix :math:`A`. Such that :math:`A_{y,x}`
       is the probability to observe :math:`y` given the true outcome should be :math:`x`,
       where :math:`x,y \in \left\{0,1\right\}` can be 0 and 1.

       In the experiment, two circuits are constructed - one for 0 outcome for all
       qubits and one for 1 outcome. From the observed results on the circuit, the
       probability for each :math:`x,y` is determined, and :math:`A_{y,x}` is set accordingly.

       Analysis Results:
           * "Local Readout Mitigator": The :class:`~qiskit.result.LocalReadoutMitigator`.

       Analysis Figures:
           * (Optional) A figure of the assignment matrix.
             Note: producing this figure scales exponentially with the number of qubits.

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
        # since the plot size grows exponentially with the number of qubits, plotting is off by default
        options.plot = False
        options.ax = None
        return options

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        data = experiment_data.data()
        qubits = experiment_data.metadata["physical_qubits"]
        matrices = self._generate_matrices(data)
        result_mitigator = LocalReadoutMitigator(matrices, qubits=qubits)
        analysis_results = [AnalysisResultData("Local Readout Mitigator", result_mitigator)]
        if self.options.plot:
            figure = assignment_matrix_visualization(
                result_mitigator.assignment_matrix(), ax=self.options.ax
            )
            figures = [figure]
        else:
            figures = None
        return analysis_results, figures

    def _generate_matrices(self, data) -> List[np.array]:
        num_qubits = len(data[0]["metadata"]["state_label"])
        counts = [None, None]
        for result in data:
            for i in range(2):
                if result["metadata"]["state_label"] == str(i) * num_qubits:
                    counts[i] = result["counts"]
        matrices = []
        for k in range(num_qubits):
            matrix = np.zeros([2, 2], dtype=float)
            marginalized_counts = []
            shots = []
            for i in range(2):
                marginal_cts = marginal_distribution(counts[i], [k])
                marginalized_counts.append(marginal_cts)
                shots.append(sum(marginal_cts.values()))

            # matrix[i][j] is the probability of counting i for expected j
            for i in range(2):
                for j in range(2):
                    matrix[i][j] = marginalized_counts[j].get(str(i), 0) / shots[j]
            matrices.append(matrix)
        return matrices


def assignment_matrix_visualization(assignment_matrix, ax=None):
    """Displays a visualization of the assignment matrix compared to the identity"""
    if ax is None:
        ax = get_non_gui_ax()
    figure = ax.get_figure()
    n = len(assignment_matrix)
    diff = np.abs(assignment_matrix - np.eye(n))
    im2 = ax.matshow(diff, cmap=plt.cm.Reds, vmin=0, vmax=0.2)
    ax.set_yticks(np.arange(n))
    ax.set_xticks(np.arange(n))
    ax.set_yticklabels(n * [""])
    ax.set_xticklabels(n * [""])
    ax.set_title(r"$|A - I  |$", fontsize=16)
    ax.set_xlabel("Prepared State")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Measured State")
    figure.colorbar(im2, ax=ax)
    return figure
