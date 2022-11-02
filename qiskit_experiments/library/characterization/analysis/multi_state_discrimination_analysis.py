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

from typing import List, Tuple

import matplotlib
import numpy as np
from sklearn.linear_model import SGDClassifier

from qiskit.providers.options import Options
from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData, ExperimentData
from qiskit_experiments.data_processing import SkCLF
from qiskit_experiments.visualization import IQPlotter, MplDrawer, PlotStyle


class MultiStateDiscriminationAnalysis(BaseAnalysis):
    """This class fits a multi-state discriminator to the data.
    
    The class will report the configuration of the discriminator in the analysis result as well as
    the fidelity of the discrimination reported as
    
    .. math::
    
        F = 1 - \frac{1}{d}\sum{i\neq j}P(i|j)
        
    Here, :math:`d` is the number of levels that were discriminated while :math:`P(i|j)` is the probability
    of measuring outcome :math:`i` given that state :math:`j` was prepared.
    """

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
        options.discriminator = SkCLF(SGDClassifier(loss="modified_huber", max_iter=1000, tol=1e-3))
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
        num_shots = len(experiment_data.data()[0]["memory"])

        # Process the data and get labels
        data, fit_state = [], []
        for i in range(n_states):
            state_data = []
            for j in range(num_shots):
                state_data.append(experiment_data.data()[i]["memory"][j][0])
            data.append(np.array(state_data))
            fit_state.append(experiment_data.data()[i]["metadata"]["label"])

        # Train a discriminator on the processed data
        discriminator = self.options.discriminator
        discriminator.fit(
            np.concatenate(data),
            np.asarray([[label] * num_shots for label in fit_state]).flatten().transpose(),
        )

        # Crate analysis results from the discriminator configuration
        analysis_results = [
            AnalysisResultData(name="discriminator_config", value=discriminator.config())
        ]
        # Create figure
        if self.options.plot:
            figures = [self._levels_plot(discriminator, data, fit_state).get_figure()]
        else:
            figures = []

        return analysis_results, figures

    def _levels_plot(self, discriminator, data, fit_state) -> matplotlib.figure.Figure:
        """Helper function for plotting IQ plane for different energy levels.

        Args:
            discriminator: the trained discriminator
            data: the training data
            fit_state: the labels

        Returns:
            The plotted IQ data.
        """
        # Create IQPlotter and generate figure.
        plotter = IQPlotter(MplDrawer())
        plotter.set_options(
            discriminator_max_resolution=64,
            style=PlotStyle(figsize=(6, 4), legend_loc=None),
        )

        # create figure labels
        params_dict = {}
        for i, label in enumerate(fit_state):
            params_dict[label] = {"label": "$|%s\\rangle$" % i}
        plotter.set_figure_options(series_params=params_dict)

        # calculate centroids
        centroids = [np.mean(x, axis=0) for x in data]

        for p, c, n in zip(data, centroids, fit_state):
            _name = f"{n}"
            plotter.set_series_data(_name, points=p, centroid=c)
        plotter.set_supplementary_data(discriminator=discriminator)

        return plotter.figure()
