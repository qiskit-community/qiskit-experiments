# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Multi state discrimination analysis."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from typing import List, Tuple

from qiskit.providers.options import Options
from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData, ExperimentData
from qiskit_experiments.framework.matplotlib import get_non_gui_ax
from qiskit_experiments.data_processing import SkCLF

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import SGDClassifier


class MultiStateDiscriminationAnalysis(BaseAnalysis):
    """This class fits a multi-state discriminator to the data."""

    @classmethod
    def _default_options(cls) -> Options:
        """Return default analysis options.

        Analysis Options:
            plot (bool): Set ``True`` to create figure for fit result.
            ax(AxesSubplot): Optional. A matplotlib axis object to draw.
            discriminator: The discriminator to classify the data. The default is a stochastic
            gradient descent (SGD) classifier.
        """
        options = super()._default_options()
        options.plot = True
        options.ax = None
        options.discriminator = SkCLF(SGDClassifier(loss="modified_huber", max_iter=1000,
                                                    tol=1e-3))
        return options

    def _run_analysis(
            self,
            experiment_data: ExperimentData,
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        """Train a discriminator based on the experiment data.

        Args:
            experiment_data: the data obtained from the experiment

        Returns:
            The configuration of the trained discriminator and the IQ plot.
        """

        # number of states and shots
        n_states = len(experiment_data.data())
        NUM_SHOTS = len(experiment_data.data()[0]['memory'])

        # Process the data and get labels
        result_data, fit_state = [], []
        for i in range(n_states):
            state_data = []
            for j in range(NUM_SHOTS):
                re, im = experiment_data.data()[i]['memory'][j][0]
                state_data.append(re + 1j * im)
                fit_state.append(experiment_data.data()[i]['metadata']['label'])
            result_data.append(np.array(state_data))
        data = np.concatenate([self._reshape_complex_vec(result) for result in result_data])

        # Train a discriminator on the processed data
        discriminator = self.options.discriminator
        discriminator.fit(data, fit_state)

        # Crate analysis results from the discriminator configuration
        analysis_results = [AnalysisResultData(name="discriminator_config",
                                               value=discriminator.config())]
        # Create figure
        if self.options.plot:
            ax = self.options.get("ax", None)
            figures = [self._levels_plot(discriminator.discriminator, data,
                                         n_states, NUM_SHOTS, ax).get_figure()]
        else:
            figures = []

        return analysis_results, figures

    def _reshape_complex_vec(self, vec: List[complex]) -> List[List[float]]:
        """Take in complex vector vec and return 2D array with real, imaginary entries.
        This is needed for the learning.

        Args:
            vec (List): complex vector of data

        Returns:
            List: vector with entries given by [real(vec[i]), imag(vec[i])]
        """
        length = len(vec)
        vec_reshaped = np.zeros((length, 2))
        for i in range(len(vec)):
            vec_reshaped[i] = [np.real(vec[i]), np.imag(vec[i])]
        return vec_reshaped

    def _levels_plot(self, clf, data, n_states, NUM_SHOTS, ax=None):
        """Helper function for plotting IQ plane for different energy levels.
        Args:
            clf: the trained discriminator
            data: the training data
            n_states: the number of energy levels
            NUM_SHOTS: the number of shots used in the experiment
        Returns:
            The plotted IQ data.
        """
        if ax is None:
            ax = get_non_gui_ax()
        figure = ax.get_figure()

        # get the colour map, the colour list needs to be trimmed to match the scatter plot
        colours = sns.color_palette('muted')[:n_states]
        cmap = matplotlib.colors.ListedColormap(colours)

        # Get the decision boundary plot
        DecisionBoundaryDisplay.from_estimator(
            clf,
            data,
            cmap=cmap,
            alpha=0.3,
            ax=ax,
            response_method="predict",
            plot_method="pcolormesh",
            shading="auto",
        )

        # One cloud of points per level
        for i in range(n_states):
            zorder = 1
            # plot with priority cloud for |1> state
            if i == 1:
                zorder = 2
            ax.scatter(data[i*NUM_SHOTS:NUM_SHOTS*(i+1), 0], data[i*NUM_SHOTS:NUM_SHOTS*(i+1), 1],
                       color=colours[i], cmap=plt.cm.Paired, edgecolors="k",
                       label=r'$|%s\rangle$' % i, zorder=zorder)
        ax.legend()
        ax.set_ylabel('I', fontsize=15)
        ax.set_xlabel('Q', fontsize=15)
        ax.set_title("Energy levels discrimination", fontsize=15)

        return figure