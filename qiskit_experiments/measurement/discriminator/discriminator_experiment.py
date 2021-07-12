"""
Standard Discriminator Experiment class.
"""

from typing import List, Optional, Union, Iterable, Tuple
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


from qiskit_experiments.base_experiment import BaseExperiment

from qiskit.circuit import QuantumCircuit
from qiskit.qobj.utils import MeasLevel
from qiskit.providers.options import Options

from qiskit_experiments.analysis import plotting
from qiskit_experiments import AnalysisResult
from qiskit_experiments.data_processing.processor_library import get_to_signal_processor
from .discriminator_analysis import DiscriminatorAnalysis


class Discriminator(BaseExperiment):
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
            circ = QuantumCircuit(1)
            if label == 1:
                circ.x(0)
            circ.measure_all()

            circ.metadata = {
                "experiment_type": self._type,
                "ylabel": str(label),
                "qubit": self.physical_qubits[0],
                "meas_level": self.run_options.meas_level,
                "meas_return": self.run_options.meas_return,
            }
            circuits.append(circ)

        return circuits
