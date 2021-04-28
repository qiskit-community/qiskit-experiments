"""
Standard discriminator analysis class.
"""

import numpy as np
from qiskit_experiments.base_analysis import BaseAnalysis, AnalysisResult
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class DiscriminatorAnalysis(BaseAnalysis):
    def _run_analysis(
        self, experiment_data, discriminator_type="LDA", plot: bool = True, **options
    ):
        """Run analysis on discriminator data.
        Args:
            experiment_data (ExperimentData): The experiment data to analyze.
            discriminator_type (str): Type of discriminator to use in analysis. Default is LDA.
            options: kwarg options for analysis function.
        Returns:
            tuple: A pair ``(analysis_results, figures)`` where
                ``analysis_results`` may be a single or list of
                AnalysisResult objects, and ``figures`` may be
                None, a single figure, or a list of figures.
        """

        nqubits = len(experiment_data.data[0]["metadata"]["ylabel"])
        discriminator = [None] * nqubits
        score = [None] * nqubits
        fig, ax = plt.subplots(nqubits)
        fig.tight_layout()
        if nqubits == 1:
            ax = [ax]

        for q in range(nqubits):
            _xdata, _ydata = self._process_data(experiment_data, q)

            if discriminator_type == "LDA":
                discriminator[q] = LinearDiscriminantAnalysis()
            elif discriminator_type == "QDA":
                discriminator[q] = QuadraticDiscriminantAnalysis()

            discriminator[q].fit(_ydata, _xdata)

            if plot:
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
                scatter = ax[q].scatter(_ydata[:, 0], _ydata[:, 1], c=_xdata)
                zz = discriminator[q].predict(np.c_[xx.ravel(), yy.ravel()])
                zz = np.array(zz).astype(float).reshape(xx.shape)
                ax[q].contourf(xx, yy, zz, alpha=0.2)
                ax[q].set_xlabel("I data")
                ax[q].set_ylabel("Q data")
                ax[q].legend(*scatter.legend_elements())
            score[q] = discriminator[q].score(_ydata, _xdata)

        if discriminator_type == "LDA":
            analysis_result = AnalysisResult(
                {
                    "discriminator": discriminator,
                    "coef": [d.coef_ for d in discriminator],
                    "intercept": [d.intercept_ for d in discriminator],
                    "score": score,
                    "plt": ax,
                }
            )

        elif discriminator_type == "QDA":
            analysis_result = AnalysisResult(
                {
                    "discriminator": discriminator,
                    "rotations": [d.rotations_ for d in discriminator],
                    "score": score,
                    "plt": ax,
                }
            )

        return analysis_result, None

    def _process_data(self, experiment_data, qubit):
        """Returns x and y data for discriminator on specific qubit."""
        xdata = np.array(
            [int(experiment_data.data[0]["metadata"]["ylabel"][qubit])]
            * len(experiment_data.data[0]["memory"])
        )
        ydata = experiment_data.data[0]["memory"][:, qubit, :]
        xdata = np.concatenate(
            (
                xdata,
                [int(experiment_data.data[1]["metadata"]["ylabel"][qubit])]
                * len(experiment_data.data[1]["memory"]),
            )
        )
        ydata = np.concatenate((ydata, experiment_data.data[1]["memory"][:, qubit, :]))
        return xdata, ydata
