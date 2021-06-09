"""
Standard Discriminator Experiment class.
"""

from typing import List, Optional, Union, Iterable, Tuple
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


from qiskit_experiments.base_experiment import BaseExperiment
from qiskit_experiments.base_analysis import BaseAnalysis

from qiskit.circuit import QuantumCircuit
from qiskit.qobj.utils import MeasLevel
from qiskit.providers.options import Options

from qiskit_experiments.analysis import plotting
from qiskit_experiments import AnalysisResult
from qiskit_experiments.data_processing.processor_library import get_to_signal_processor


class DiscriminatorAnalysis(BaseAnalysis):
    @classmethod
    def _default_options(cls):
        return Options(
            meas_level=MeasLevel.KERNELED,
            meas_return="single",
            discriminator_type="LDA",
        )

    def _run_analysis(
        self, experiment_data, discriminator_type="LDA", data_processor: Optional[callable] = None, plot: bool = True, **options
    ) -> Tuple[AnalysisResult, List["matplotlib.figure.Figure"]]:
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
        data = experiment_data.data()

        qubit = data[0]["metadata"]["qubit"]
        _xdata, _ydata = self._process_data(data, qubit)

        if discriminator_type == "LDA":
            discriminator = LinearDiscriminantAnalysis()
        elif discriminator_type == "QDA":
            discriminator = QuadraticDiscriminantAnalysis()
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
            #ax.legend(*scatter.legend_elements())
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
                    "plt": ax,
                }
            )

        elif discriminator_type == "QDA":
            analysis_result = AnalysisResult(
                {
                    "discriminator": discriminator,
                    "rotations": discriminator.rotations_,
                    "score": score,
                    "plt": ax,
                }
            )

        return analysis_result, figures

    def _process_data(self, data, qubit):
        """Returns x and y data for discriminator on specific qubit."""
        # xdata = np.array([int(data[0]["metadata"]["ylabel"][qubit])] * len(data[0]["memory"]))
        # ydata = data[0]["memory"][:, qubit, :]
        # xdata = np.concatenate(
        #     (
        #         xdata,
        #         [int(data[1]["metadata"]["ylabel"][qubit])] * len(data[1]["memory"]),
        #     )
        # )
        # ydata = np.concatenate((ydata, data[1]["memory"][:, qubit, :]))
        # return xdata, ydata
        xdata = np.array([int(data[0]["metadata"]["ylabel"])] * len(data[0]["memory"]))
        ydata = data[0]["memory"][:,0,:]
        xdata = np.concatenate(
            (
                xdata,
                [int(data[1]["metadata"]["ylabel"])] * len(data[1]["memory"]),
            )
        )
        ydata = np.concatenate((ydata, data[1]["memory"][:,0,:]))
        return xdata, ydata

class DiscriminatorExperiment(BaseExperiment):
    """Discriminator Experiment class"""

    # Analysis class for experiment
    __analysis_class__ = DiscriminatorAnalysis

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default options values for the experiment :meth:`run` method."""
        return Options(
            meas_level=MeasLevel.KERNELED,
            meas_return="single",
        )

    def __init__(
        self,
        qubit: int,
    ):
        """Standard discriminator experiment

        Args:
            qubit: The qubit to discriminate on.
        """

        super().__init__([qubit])

    def circuits(self, backend: Optional["Backend"] = None) -> List[QuantumCircuit]:
        """Return a list of discriminator circuits.
        Args:
            backend (Backend): Optional, a backend object.
        Returns:
            List[QuantumCircuit]: A list of :class:`QuantumCircuit`s.
        """
        circuits = []
        for label in (0, 1):
            circ = QuantumCircuit(1, 1)
            if label == 1:
                circ.x(0)
            circ.measure(0, 0)

            circ.metadata = {
                "experiment_type": self._type,
                "ylabel": str(label),
                "qubit": self.physical_qubits[0],
                "meas_level": self.run_options.meas_level,
                "meas_return": self.run_options.meas_return,
            }
            circuits.append(circ)

        return circuits
