"""
Standard Discriminator Experiment class.
"""

from qiskit_experiments.base_experiment import BaseExperiment

from qiskit.circuit import QuantumCircuit
from .discriminator_analysis import DiscriminatorAnalysis
from typing import List, Optional, Union, Iterable
from qiskit.qobj.utils import MeasLevel


class DiscriminatorExperiment(BaseExperiment):
    """Discriminator Experiment class"""

    # Analysis class for experiment
    __analysis_class__ = DiscriminatorAnalysis

    # default run options
    __run_defaults__ = {"meas_level": MeasLevel.KERNELED, "meas_return": "single"}


    def __init__(
        self,
        qubits: Union[int, Iterable[int]],
    ):
        """Standard discriminator experiment

        Args:
            qubits: the number of qubits or list of
                    physical qubits for the experiment.
        """

        super().__init__(qubits)

    def circuits(self, backend: Optional["Backend"] = None) -> List[QuantumCircuit]:
        """Return a list of discriminator circuits.
        Args:
            backend (Backend): Optional, a backend object.
        Returns:
            List[QuantumCircuit]: A list of :class:`QuantumCircuit`s.
        """
        circuits = []

        for label in (0, 1):
            circ = QuantumCircuit(self.num_qubits)
            if label == 1:
                for qubit in range(self.num_qubits):
                    circ.x(qubit)
            circ.measure_all()

            circ.metadata = {
                "experiment_type": self._type,
                "ylabel": self.num_qubits * str(label),
                "qubits": self.physical_qubits,
            }
            circuits.append(circ)

        return circuits
