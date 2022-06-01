## This code is part of Qiskit.
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
"""
Standard discriminator analysis class.
"""
from typing import List, Optional, Tuple
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from qiskit_experiments.framework import AnalysisResultData, ExperimentData

from qiskit_experiments.framework.matplotlib import get_non_gui_ax
from qiskit_experiments.curve_analysis.visualization import plot_contourf, plot_scatter
from qiskit_experiments.framework import BaseAnalysis, Options, AnalysisResultData
from qiskit_experiments.framework.sklearn import requires_sklearn


class DiscriminatorAnalysis(BaseAnalysis):
    """A class to analyze discriminator experiments.
    LDA, QDA, and Gaussian mixture methods are supported.
    """

    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        options.ax = None
        options.discriminator_type = "LDA"

        return options

    # @requires_sklearn
    # def LDA(self, xdata, ydata):
    #     discriminator.
    #     return analysis_results

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[AnalysisResultData, List["matplotlib.figure.Figure"]]:
        """Run analysis on discriminator data.
        Args:
            experiment_data (ExperimentData): The experiment data to analyze.
            discriminator_type (str): Type of discriminator to use in analysis. Default is
                Linear Discriminant Analysis, which fits a Gaussian density to each class
                with the assumption that all classes have the same covariance, generating
                a linear decision boundary.
            options: kwarg options for analysis function.
        Returns:
            tuple: A pair ``(analysis_results, figures)`` where
                ``analysis_results`` may be a single or list of
                AnalysisResult objects, and ``figures`` may be
                None, a single figure, or a list of figures.
        """
        data = experiment_data.data()

        _xdata, _ydata = self._process_data(data)

        if self.options.discriminator_type == "LDA":
            discriminator = LinearDiscriminantAnalysis()
        elif self.options.discriminator_type == "QDA":
            discriminator = QuadraticDiscriminantAnalysis()
        elif self.options.discriminator_type == "GaussianMixture":
            centers = []
            for level in np.unique(_xdata):
                ix = np.where(_xdata == level)
                centers.append(np.average(_ydata[ix], axis=0))
            discriminator = GaussianMixture(
                n_components=2, covariance_type="spherical", means_init=centers
            )
        else:
            raise AttributeError("Unsupported discriminator type")

        discriminator.fit(_ydata, _xdata)
        score = discriminator.score(_ydata, _xdata)

        if self.options.plot:
            spacing_x = (max(_ydata[:, 0]) - min(_ydata[:, 0])) / 10
            spacing_y = (max(_ydata[:, 1]) - min(_ydata[:, 1])) / 10
            xx, yy = np.meshgrid(
                np.arange(
                    min(_ydata[:, 0]) - spacing_x,
                    max(_ydata[:, 0]) + spacing_x,
                    (max(_ydata[:, 0]) - min(_ydata[:, 0])) / 500,
                ),
                np.arange(
                    min(_ydata[:, 1]) - spacing_y,
                    max(_ydata[:, 1]) + spacing_y,
                    (max(_ydata[:, 1]) - min(_ydata[:, 1])) / 500,
                ),
            )

            if self.options.ax is None:
                ax = get_non_gui_ax()
            else:
                ax = self.options.ax
            for level in np.unique(_xdata):
                ix = np.where(_xdata == level)
                ax.scatter(_ydata[ix, 0], _ydata[ix, 1], label=f"|{level}>", marker="x")
            zz = discriminator.predict(np.c_[xx.ravel(), yy.ravel()])
            zz = np.array(zz).astype(float).reshape(xx.shape)
            ax = plot_contourf(xx, yy, zz, ax, alpha=0.2)
            ax.set_xlabel("I data")
            ax.set_ylabel("Q data")
            ax.legend()
            figures = [ax.get_figure()]
        else:
            figures = None

        if self.options.discriminator_type == "LDA":
            analysis_results = [
                AnalysisResultData(
                    "discriminator",
                    value=discriminator,
                ),
                AnalysisResultData(
                    "coef",
                    value=str(discriminator.coef_),
                ),
                AnalysisResultData(
                    "intercept",
                    value=str(discriminator.intercept_),
                ),
                AnalysisResultData(
                    "score",
                    value=score,
                ),
            ]
        elif self.options.discriminator_type == "QDA":
            analysis_results = [
                AnalysisResultData(
                    "discriminator",
                    value=discriminator,
                ),
                AnalysisResultData(
                    "rotations",
                    value=discriminator.rotations_,
                ),
                AnalysisResultData(
                    "score",
                    value=score,
                ),
            ]
        elif self.options.discriminator_type == "GaussianMixture":
            analysis_results = [
                AnalysisResultData(
                    "discriminator",
                    value=discriminator,
                ),
                AnalysisResultData(
                    "means",
                    value=str(discriminator.means_),
                ),
                AnalysisResultData(
                    "covariances",
                    value=str(np.sqrt(discriminator.covariances_)),
                ),
            ]
        return analysis_results, figures

    def _process_data(self, data):
        """Returns x and y data for discriminator on specific qubit."""
        xdata = np.array([int(data[0]["metadata"]["ylabel"])] * len(data[0]["memory"]))
        ydata = np.array(data[0]["memory"])[:, 0, :]
        xdata = np.concatenate(
            (
                xdata,
                [int(data[1]["metadata"]["ylabel"])] * len(data[1]["memory"]),
            )
        )
        ydata = np.concatenate((ydata, np.array(data[1]["memory"])[:, 0, :]))
        return xdata, ydata
