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
from qiskit_experiments.curve_analysis.visualization import (
    plot_contourf,
    plot_scatter,
    plot_ellipse,
)
from qiskit_experiments.framework import BaseAnalysis, Options, AnalysisResultData
from qiskit_experiments.framework.sklearn import requires_sklearn


try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


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

    def gaussian_analysis(self, gmm, ax):
        angles = []
        diameters = []
        for n in range(len(gmm.means_)):
            if gmm.covariance_type == "full":
                covariances = gmm.covariances_[n][:2, :2]
            elif gmm.covariance_type == "tied":
                covariances = gmm.covariances_[:2, :2]
            elif gmm.covariance_type == "diag":
                covariances = np.diag(gmm.covariances_[n][:2])
            elif gmm.covariance_type == "spherical":
                covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            # plot 1- and 2-sigma ellipses
            for i in range(1, 3):
                ax = plot_ellipse(
                    gmm.means_[n, :2],
                    v[0] * i,
                    v[1] * i,
                    180 + angle,
                    ax,
                    alpha=1 - (i - 1) / 2,
                )
            diameters.append(v)
            angles.append(angle + 180)

        analysis_results = [
            AnalysisResultData(
                "discriminator",
                value=gmm,
            ),
            AnalysisResultData(
                "centers",
                value=str(gmm.means_),
            ),
            AnalysisResultData(
                "covariances",
                value=str(gmm.covariances_),
            ),
            AnalysisResultData("diameters", value=str(diameters)),
            AnalysisResultData("angle", value=str(angles)),
        ]
        return ax, analysis_results

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
                n_components=2, covariance_type="full", means_init=centers
            )
        else:
            raise AttributeError("Unsupported discriminator type")

        discriminator.fit(_ydata, _xdata)
        score = discriminator.score(_ydata, _xdata)

        if self.options.plot:
            minxy = np.amin(_ydata)
            maxxy = np.amax(_ydata)

            spacing = (maxxy - minxy) / 20

            side = np.arange(minxy - spacing, maxxy + spacing, (maxxy - minxy) / 500)

            xx, yy = np.meshgrid(side, side)

            if self.options.ax is None:
                ax = get_non_gui_ax()
            else:
                ax = self.options.ax

            zz = discriminator.predict(np.c_[xx.ravel(), yy.ravel()])
            zz = np.array(zz).astype(float).reshape(xx.shape)
            fig = ax.get_figure()
            fig.set_size_inches(5, 5)
            ax.axis("off")
            ax.set_title(f"Qubit {experiment_data._metadata['physical_qubits']}")

            gs = fig.add_gridspec(
                2,
                2,
                width_ratios=(7, 2),
                height_ratios=(2, 7),
                left=0.1,
                right=0.85,
                bottom=0.1,
                top=0.85,
                wspace=0.15,
                hspace=0.15,
            )
            ax = fig.add_subplot(gs[1, 0])

            ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
            ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

            ax.set_xlabel("I data")
            ax.set_ylabel("Q data")

            ax_histx.tick_params(axis="x", labelbottom=False)
            ax_histy.tick_params(axis="y", labelleft=False)

            ax.margins(0)
            ax = plot_contourf(xx, yy, zz, ax, alpha=0.2)
            ax.margins(0)
            for level in np.unique(_xdata):
                ix = np.where(_xdata == level)

                # plot in two layers so higher levels don't cover up lower
                tophalf = ax.scatter(
                    np.array_split(_ydata[ix, 0], 2, 1)[0],
                    np.array_split(_ydata[ix, 1], 2, 1)[0],
                    label=f"|{level}>",
                    marker=".",
                    s=5,
                    alpha=0.6,
                    zorder=level,
                )
                ax.scatter(
                    np.array_split(_ydata[ix, 0], 2, 1)[1],
                    np.array_split(_ydata[ix, 1], 2, 1)[1],
                    marker=".",
                    s=5,
                    alpha=0.6,
                    color=tophalf.get_facecolors()[0],
                    zorder=-level,
                )
                xymax = max(np.max(np.abs(_ydata[ix, 0])), np.max(np.abs(_ydata[ix, 1])))
                binwidth = xymax / 50
                lim = (int(xymax / binwidth) + 1) * binwidth

                bins = np.arange(-lim, lim + binwidth, binwidth)
                ax_histx.hist(_ydata[ix, 0][0], bins=bins, histtype="step", log=True)
                ax_histy.hist(
                    _ydata[ix, 1][0], bins=bins, histtype="step", orientation="horizontal", log=True
                )

            ax.legend()
            ax.tick_params(labelsize=10)
            ax.set_xlim([minxy - spacing, maxxy + spacing])
            ax.set_ylim([minxy - spacing, maxxy + spacing])
            if self.options.discriminator_type == "GaussianMixture":
                ax, analysis_results = self.gaussian_analysis(discriminator, ax)

            figures = [fig]
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
