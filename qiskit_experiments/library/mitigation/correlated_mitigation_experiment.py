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
Correlated readout mitigation calibration experiment class.
"""
from typing import Iterable, List
from qiskit import QuantumCircuit
from qiskit_experiments.framework import BaseExperiment
from .correlated_mitigation_analysis import CorrelatedMitigationAnalysis
from .utils import calibration_circuit

class CorrelatedReadoutMitigationExperiment(BaseExperiment):
    """Class for correlated readout mitigation experiment
    # section: overview
        Readout mitigation aims to reduce the effect of errors during the measurement
        of the qubits in a quantum device. It is used both to obtain a more accurate
        distribution of the outputs, and more accurate measurements of expectation
        value for measurables.

        The readout mitigator is generated from an *assignment matrix*:
        a :math:`2^n\times 2^n` matrix :math:`A` such that :math:`A_{y,x}` is the probability
        to observe :math:`y` given the true outcome should be :math:`x`. The assignment matrix is used
        to compute the *mitigation matrix* used in the mitigation process itself.

        A *Correlated readout mitigator* uses the full :math:`2^n\times 2^n` assignment matrix, meaning
        it can only be used for small values of :math:`n`.
        The corresponding class in Qiskit is the `Correlated readout mitigator
        <https://qiskit.org/documentation/stubs/qiskit.result.CorrelatedReadoutMitigator.html>_`
        in `qiskit-terra`.

        The experiment generates :math:`2^n` circuits, for every possible
        :math:`n`-qubit quantum state and constructs
        the assignment matrix and correlated mitigator from the results.

        See :class:`CorrelatedMitigationAnalysis`
        documentation for additional information on correlated readout mitigation experiment analysis.

    # section: analysis_ref
        :py:class:`CorrelatedMitigationAnalysis`

    # section: reference
        .. ref_arxiv:: 1 2006.14044
    """
    def __init__(self, qubits: Iterable[int]):
        """Initialize a correlated readout mitigation calibration experiment.

        Args:
            qubits: The qubits being mitigated

        Additional info:
            The currently supported mitigation methods are:
            * "local": each qubit is mitigated by itself; this is the default method,
            and assumes readout errors are independent for each qubits
            * "correlated": All the qubits are mitigated together; this results in an exponentially
            large mitigation matrix and so is useable only for a small number of qubits,
            but might be more accurate than local mitigation.
        """
        super().__init__(qubits)
        self.analysis = CorrelatedMitigationAnalysis()

    def circuits(self) -> List[QuantumCircuit]:
        """Returns the experiment's circuits"""
        labels = [bin(j)[2:].zfill(self.num_qubits) for j in range(2 ** self.num_qubits)]
        return [calibration_circuit(self.num_qubits, label) for label in self.helper.labels()]