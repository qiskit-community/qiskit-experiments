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
from qiskit_experiments.framework import BaseExperiment, ParallelExperiment, ExperimentData
from .mitigation_analysis import CompleteMitigationAnalysis


class MeasurementMitigation(BaseExperiment):
    """Interface class for measurement mitigation experiments"""

    METHOD_COMPLETE = "complete"
    METHOD_TENSORED = "tensored"
    ALL_METHODS = [METHOD_COMPLETE, METHOD_TENSORED]

    def __init__(self, qubits: Iterable[int], method=METHOD_COMPLETE):
        if method not in self.ALL_METHODS:
            raise QiskitError("Method {} not recognized".format(method))
        self._method = method
        if method == self.METHOD_COMPLETE:
            self._exp = CompleteMeasurementMitigation(qubits)
            qubit_list = qubits
        if method == self.METHOD_TENSORED:
            if not isinstance(qubits, list) or not isinstance(qubits[0], list):
                raise QiskitError(
                    "Tensored experiment requires a list of qubit lists; {} was passed".format(
                        qubits
                    )
                )
            sub_experiments = []
            for qubit_list in qubits:
                exp = CompleteMeasurementMitigation(qubit_list)
                sub_experiments.append(exp)
            self._exp = ParallelExperiment(sub_experiments)
        super().__init__(qubit_list)

    def circuits(self, backend: Optional[Backend] = None) -> List[QuantumCircuit]:
        return self._exp.circuits(backend)

    def run(
        self,
        backend: Backend,
        analysis: bool = True,
        experiment_data: Optional[ExperimentData] = None,
        **run_options,
    ) -> ExperimentData:
        if self._method == self.METHOD_COMPLETE:
            res = self._exp.run(backend, analysis, experiment_data, **run_options)
            return res
        if self._method == self.METHOD_TENSORED:
            res = self._exp.run(backend, analysis, experiment_data, **run_options)
            return res
        return None


class CompleteMeasurementMitigation(BaseExperiment):
    """
    Measurement correction experiment for a full calibration
    """

    __analysis_class__ = CompleteMitigationAnalysis

    def __init__(self, qubits: List[int]):
        super().__init__(qubits)

    def labels(self) -> List[str]:
        return [bin(j)[2:].zfill(self.num_qubits) for j in range(2 ** self.num_qubits)]

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
