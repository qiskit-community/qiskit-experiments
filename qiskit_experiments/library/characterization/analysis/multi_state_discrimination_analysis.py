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

from typing import List, Tuple, TYPE_CHECKING

import matplotlib
import numpy as np

from qiskit.providers.options import Options
from qiskit_experiments.framework import BaseAnalysis, AnalysisResultData, ExperimentData
from qiskit_experiments.data_processing import SkQDA
from qiskit_experiments.visualization import BasePlotter, IQPlotter, MplDrawer, PlotStyle
from qiskit_experiments.framework.package_deps import HAS_SKLEARN

if TYPE_CHECKING:
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class MultiStateDiscriminationAnalysis(BaseAnalysis):
    r"""This class fits a multi-state discriminator to the data.

    The class will report the configuration of the discriminator in the analysis result as well as
    the fidelity of the discrimination reported as

    .. math::

        F = 1 - \frac{1}{d}\sum{i\neq j}P(i|j)

    Here, :math:`d` is the number of levels that were discriminated while :math:`P(i|j)` is the
    probability of measuring outcome :math:`i` given that state :math:`j` was prepared.

    .. note::
        This class requires that scikit-learn is installed.
    """

    @classmethod
    @HAS_SKLEARN.require_in_call
    def _default_options(cls) -> Options:
        """Return default analysis options.

        Analysis Options:
            plot (bool): Set ``True`` to create figure for fit result.
            plotter (BasePlotter): A plotter instance to visualize the analysis result.
            ax (AxesSubplot): Optional. A matplotlib axis object in which to draw.
            discriminator (BaseDiscriminator): The sklearn discriminator to classify the data.
                The default is a quadratic discriminant analysis.
        """
        options = super()._default_options()
        options.plotter = IQPlotter(MplDrawer())
        options.plotter.set_options(
            discriminator_max_resolution=64,
            style=PlotStyle(figsize=(6, 4), legend_loc=None),
        )
        options.plot = True
        options.ax = None
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        options.discriminator = SkQDA(QuadraticDiscriminantAnalysis())
        return options

    @property
    def plotter(self) -> BasePlotter:
        """A short-cut to the IQ plotter instance."""
        return self._options.plotter

    def _run_analysis(
        self,
        experiment_data: ExperimentData,
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        """Train a discriminator based on the experiment data.

        Args:
            experiment_data: the data obtained from the experiment

        Returns:
            The configuration of the trained discriminator and the IQ plot and the fidelity of the
            discrimination.
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

        # Calculate fidelity. First we need to calculate P(i|j):= prob. measuring outcome i given
        # state j was prepared
        predicted_data = [discriminator.predict(state_data) for state_data in data]
        # count per prepared state the number of measured states of each kind and calculate the
        # probability of measuring the wrong state
        prob_wrong = 0
        for i in range(n_states):
            counts = [0] * n_states
            for point in predicted_data[i]:
                counts[point] += 1
            for j in range(n_states):
                if j != i:
                    prob_wrong += counts[j] / num_shots

        # calculate the fidelity
        fidelity = 1 - (1 / n_states) * prob_wrong

        # Crate analysis results from the discriminator configuration
        analysis_results = [
            AnalysisResultData(name="discriminator_config", value=discriminator.config()),
            AnalysisResultData(name="fidelity", value=fidelity),
        ]

        figures = []
        if self.options.plot:
            figures.append(self._levels_plot(discriminator, data, fit_state, fidelity).get_figure())

        return analysis_results, figures

    def _levels_plot(self, discriminator, data, fit_state, fidelity) -> matplotlib.figure.Figure:
        """Helper function for plotting IQ plane for different energy levels.

        Args:
            discriminator: the trained discriminator
            data: the training data
            fit_state: the labels
            fidelity: the fidelity of the classification

        Returns:
            The plotted IQ data.
        """
        # create figure labels
        params_dict = {}
        for state in fit_state:
            params_dict[state] = {"label": f"$|{state}\\rangle$"}
        # Update params_dict to contain any existing series_params values,
        # where they have priority over params_dict.
        params_dict.update(self.plotter.figure_options.series_params)
        self.plotter.set_figure_options(series_params=params_dict)

        # calculate centroids
        centroids = [np.mean(x, axis=0) for x in data]

        for p, c, n in zip(data, centroids, fit_state):
            self.plotter.set_series_data(n, points=p, centroid=c)
        self.plotter.set_supplementary_data(discriminator=discriminator, fidelity=fidelity)

        return self.plotter.figure()
