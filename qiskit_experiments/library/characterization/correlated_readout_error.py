# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Correlated readout error calibration experiment class.
"""
from typing import Iterable, List
from qiskit import QuantumCircuit
from qiskit_experiments.framework import BaseExperiment
from qiskit_experiments.library.characterization.analysis.correlated_readout_error_analysis import (
    CorrelatedReadoutErrorAnalysis,
)


def calibration_circuit(num_qubits: int, state_label: str) -> QuantumCircuit:
    """Return a calibration circuit.

    This is an N-qubit circuit where N is the length of the label.
    The circuit consists of X-gates on qubits with label bits equal to 1,
    and measurements of all qubits.
    """
    circ = QuantumCircuit(num_qubits, name="meas_mit_cal_" + state_label)
    for i, val in enumerate(reversed(state_label)):
        if val == "1":
            circ.x(i)
    circ.measure_all()
    circ.metadata = {"state_label": state_label}
    return circ


class CorrelatedReadoutError(BaseExperiment):
    r"""Class for correlated readout error characterization experiment
    # section: overview
        This class constructs the a :class:`~qiskit.result.CorrelatedReadoutMitigator`
        containing the full assignment matrix :math:`A` characterizing the readout error
        for the given qubits from the experiment results accessible via the
        :meth:`~qiskit.result.CorrelatedReadoutMitigator.assignment_matrix` method.

        Readout errors affect quantum computation during the measurement
        of the qubits in a quantum device. By characterizing the readout errors,
        it is possible to construct a *readout error mitigator* that is used both
        to obtain a more accurate distribution of the outputs, and more accurate
        measurements of expectation value for measurables.

        The readout mitigator is generated from an *assignment matrix*:
        a :math:`2^n\times 2^n` matrix :math:`A` such that :math:`A_{y,x}` is the probability
        to observe :math:`y` given the true outcome should be :math:`x`. The assignment matrix is used
        to compute the *mitigation matrix* used in the readout error mitigation process itself.

        A *Correlated readout mitigator* uses the full :math:`2^n \times 2^n` assignment matrix, meaning
        it can only be used for small values of :math:`n`.
        The corresponding class in Qiskit is the
        :class:`~qiskit.result.CorrelatedReadoutMitigator` in :mod:`qiskit.result`.

        The experiment generates :math:`2^n` circuits, for every possible
        :math:`n`-qubit quantum state and constructs
        the assignment matrix and correlated mitigator from the results.

        See :class:`CorrelatedReadoutErrorAnalysis`
        documentation for additional information on correlated readout error experiment analysis.

    # section: analysis_ref
        :py:class:`CorrelatedReadoutErrorAnalysis`

    # section: reference
        .. ref_arxiv:: 1 2006.14044
    """

    def __init__(self, qubits: Iterable[int]):
        """Initialize a correlated readout error characterization experiment.

        Args:
            qubits: The qubits being characterized for readout error
        """
        super().__init__(qubits)
        self.analysis = CorrelatedReadoutErrorAnalysis()

    def circuits(self) -> List[QuantumCircuit]:
        """Returns the experiment's circuits"""
        labels = [bin(j)[2:].zfill(self.num_qubits) for j in range(2**self.num_qubits)]
        return [calibration_circuit(self.num_qubits, label) for label in labels]
