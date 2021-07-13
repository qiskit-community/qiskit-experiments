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
Standard discriminator analysis class.
"""
from typing import List, Optional, Union, Iterable, Tuple
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from qiskit.providers.options import Options
from qiskit.qobj.utils import MeasLevel

from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.analysis import plotting
from qiskit_experiments import AnalysisResult


class DiscriminatorAnalysis(BaseAnalysis):
    @classmethod
    def _default_options(cls):
        return Options(
            meas_level=MeasLevel.KERNELED,
            meas_return="single",
            discriminator_type="LDA",
        )

    def _run_analysis(
        self,
        experiment_data,
        discriminator_type="LDA",
        data_processor: Optional[callable] = None,
        plot: bool = True,
        **options,
    ) -> Tuple[AnalysisResult, List["matplotlib.figure.Figure"]]:
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

        qubit = data[0]["metadata"]["qubit"]
        _xdata, _ydata = self._process_data(data, qubit)

        if discriminator_type == "LDA":
            discriminator = LinearDiscriminantAnalysis()
        elif discriminator_type == "QDA":
            discriminator = QuadraticDiscriminantAnalysis()
        else:
            raise AttributeError("Unsupported discriminator type")
        discriminator.fit(_ydata, _xdata)
        score = discriminator.score(_ydata, _xdata)

        if plot and plotting.HAS_MATPLOTLIB:
            xx, yy = np.meshgrid(
                np.arange(
                    min(_ydata[:, 0]),
                    max(_ydata[:, 0]),
                    (max(_ydata[:, 0]) - min(_ydata[:, 0])) / 500,
                ),
                np.arange(
                    min(_ydata[:, 1]),
                    max(_ydata[:, 1]),
                    (max(_ydata[:, 1]) - min(_ydata[:, 1])) / 500,
                ),
            )
            ax = plotting.plot_scatter(_ydata[:, 0], _ydata[:, 1], c=_xdata)
            zz = discriminator.predict(np.c_[xx.ravel(), yy.ravel()])
            zz = np.array(zz).astype(float).reshape(xx.shape)
            ax = plotting.plot_contourf(xx, yy, zz, ax, alpha=0.2)
            ax.set_xlabel("I data")
            ax.set_ylabel("Q data")
            figures = [ax.get_figure()]
        else:
            figures = None

        if discriminator_type == "LDA":
            analysis_result = AnalysisResult(
                {
                    "discriminator": discriminator,
                    "coef": discriminator.coef_,
                    "intercept": discriminator.intercept_,
                    "score": score,
                }
            )

        elif discriminator_type == "QDA":
            analysis_result = AnalysisResult(
                {
                    "discriminator": discriminator,
                    "rotations": discriminator.rotations_,
                    "score": score,
                }
            )

        return [analysis_result], figures

    def _process_data(self, data, qubit):
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
