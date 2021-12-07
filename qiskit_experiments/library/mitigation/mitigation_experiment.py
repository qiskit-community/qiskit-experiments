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
Measurement calibration experiment classes.
"""
from typing import Iterable, Optional, List
from abc import abstractmethod

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.exceptions import QiskitError
from qiskit_experiments.framework import BaseExperiment
from .mitigation_analysis import CompleteMitigationAnalysis, TensoredMitigationAnalysis


class MeasurementMitigation(BaseExperiment):
    """Interface class for measurement mitigation experiments"""

    METHOD_COMPLETE = "complete"
    METHOD_TENSORED = "tensored"
    ALL_METHODS = [METHOD_COMPLETE, METHOD_TENSORED]

    def __init__(self, qubits: Iterable[int]):
        super().__init__(qubits)

    @abstractmethod
    def labels(self) -> List[str]:
        return None

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        return [self._calibration_circuit(self.num_qubits, label) for label in self.labels()]

    @staticmethod
    def _calibration_circuit(num_qubits: int, label: str) -> QuantumCircuit:
        """Return a calibration circuit.

        This is an N-qubit circuit where N is the length of the label.
        The circuit consists of X-gates on qubits with label bits equal to 1,
        and measurements of all qubits.
        """
        circ = QuantumCircuit(num_qubits, name="meas_mit_cal_" + label)
        for i, val in enumerate(reversed(label)):
            if val == "1":
                circ.x(i)
        circ.measure_all()
        circ.metadata = {"label": label}
        return circ


class CompleteMeasurementMitigation(MeasurementMitigation):
    def __init__(self, qubits: Iterable[int]):
        super().__init__(qubits)
        self.analysis = CompleteMitigationAnalysis()

    def labels(self) -> List[str]:
        return [bin(j)[2:].zfill(self.num_qubits) for j in range(2 ** self.num_qubits)]


class TensoredMeasurementMitigation(MeasurementMitigation):
    def __init__(self, qubits: Iterable[int]):
        super().__init__(qubits)
        self.analysis = TensoredMitigationAnalysis()

    def labels(self) -> List[str]:
        return ["0" * self.num_qubits, "1" * self.num_qubits]


def mitigation_experiment(qubits, method=MeasurementMitigation.METHOD_COMPLETE):
    if method not in MeasurementMitigation.ALL_METHODS:
        raise QiskitError("Method {} not recognized".format(method))
    if method == MeasurementMitigation.METHOD_COMPLETE:
        return CompleteMeasurementMitigation(qubits)
    if method == MeasurementMitigation.METHOD_TENSORED:
        return TensoredMeasurementMitigation(qubits)
