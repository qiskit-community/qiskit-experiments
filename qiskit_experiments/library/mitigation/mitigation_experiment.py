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
from typing import Iterable, List

from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit_experiments.framework import BaseExperiment
from .mitigation_analysis import CorrelatedMitigationAnalysis, LocalMitigationAnalysis


class ReadoutMitigationExperiment(BaseExperiment):
    """Class for readout mitigation experiments"""

    METHOD_LOCAL = "local"
    METHOD_CORRELATED = "correlated"
    ALL_METHODS = [METHOD_LOCAL, METHOD_CORRELATED]

    def __init__(self, qubits: Iterable[int], method=METHOD_LOCAL):
        """Initialize a mitigation experiment.

        Args:
            qubits: The qubits being mitigated
            method: A string denoting mitigation method

        Raises:
            QiskitError: if the given mitigation method is not recoginzed

        Additional info:
            The currently supported mitigation methods are:
            * "local": each qubit is mitigated by itself; this is the default method,
            and assumes readout errors are independent for each qubits
            * "correlated": All the qubits are mitigated together; this results in an exponentially
            large mitigation matrix and so is useable only for a small number of qubits,
            but might be more accurate than local mitigation.
        """
        super().__init__(qubits)
        if method not in self.ALL_METHODS:
            raise QiskitError("Method {} not recognized".format(method))
        if method == self.METHOD_LOCAL:
            self.helper = LocalMitigationHelper(self.num_qubits)
        if method == self.METHOD_CORRELATED:
            self.helper = CorrelatedMitigationHelper(self.num_qubits)

        self.analysis = self.helper.analysis()

    def circuits(self) -> List[QuantumCircuit]:
        """Returns the experiment's circuits"""
        return [self._calibration_circuit(self.num_qubits, label) for label in self.helper.labels()]

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


class CorrelatedMitigationHelper:
    """Helper class for correlated mitigation experiment data"""

    def __init__(self, num_qubits: int):
        """Creates the helper class
        Args:
            num_qubits: The number of qubits being mitigated
        """
        self.num_qubits = num_qubits

    def analysis(self):
        """Returns the analysis class for the mitigation"""
        return CorrelatedMitigationAnalysis()

    def labels(self) -> List[str]:
        """Returns the labels dictating the generation of the mitigation circuits"""
        return [bin(j)[2:].zfill(self.num_qubits) for j in range(2**self.num_qubits)]


class LocalMitigationHelper:
    """Helper class for local mitigation experiment data"""

    def __init__(self, num_qubits: int):
        """Creates the helper class
        Args:
            num_qubits: The number of qubits being mitigated
        """
        self.num_qubits = num_qubits

    def analysis(self):
        """Returns the analysis class for the mitigation"""
        return LocalMitigationAnalysis()

    def labels(self) -> List[str]:
        """Returns the labels dictating the generation of the mitigation circuits"""
        return ["0" * self.num_qubits, "1" * self.num_qubits]
