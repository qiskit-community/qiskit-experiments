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
Local readout mitigation calibration experiment class.
"""
from typing import Iterable, List
from qiskit import QuantumCircuit
from qiskit_experiments.framework import BaseExperiment
from .local_readout_error_analysis import LocalReadoutErrorAnalysis
from .utils import calibration_circuit

class LocalReadoutErrorExperiment(BaseExperiment):
    """Class for local readout error characterization experiment
    # section: overview
        Readout mitigation aims to reduce the effect of errors during the measurement
        of the qubits in a quantum device. It is used both to obtain a more accurate
        distribution of the outputs, and more accurate measurements of expectation
        value for measurables.

        The readout mitigator is generated from an *assignment matrix*:
        a :math:`2^n\times 2^n` matrix :math:`A` such that :math:`A_{y,x}` is the probability
        to observe :math:`y` given the true outcome should be :math:`x`. The assignment matrix is used
        to compute the *mitigation matrix* used in the mitigation process itself.

        A *Local readout mitigator* works under the assumption the readout errors are mostly *local*, meaning
        readout errors for different qubits are independent of each other. In this case, the assignment matrix
        is the tensor product of :math:`n` :math:`2\times 2` matrices, one for each qubit, making it practical
        to store the assignment matrix in implicit form, by storing the individual :math:`2\times 2` assignment matrices.
        The corresponding class in Qiskit is the `Local readout mitigator
        <https://qiskit.org/documentation/stubs/qiskit.result.BaseReadoutMitigator.html#qiskit.result.BaseReadoutMitigator.html>_`
        in `qiskit-terra`.

        The experiment generates 2 circuits, corresponding to the states
        :math:`|0^n>` and :math:`|1^n>`, measuring the error in all the qubits at once, and constructs
        the assignment matrix and local mitigator from the results.

        See :class:`LocalMitigationAnalysis`
        documentation for additional information on local readout mitigation experiment analysis.

    # section: analysis_ref
        :py:class:`LocalMitigationAnalysis`

    # section: reference
        .. ref_arxiv:: 1 2006.14044
    """

    def __init__(self, qubits: Iterable[int]):
        """Initialize a local readout mitigation calibration experiment.

        Args:
            qubits: The qubits being characterized for readout error
        """
        super().__init__(qubits)
        self.analysis = LocalReadoutErrorAnalysis()

    def circuits(self) -> List[QuantumCircuit]:
        """Returns the experiment's circuits"""
        labels = ["0" * self.num_qubits, "1" * self.num_qubits]
        return [calibration_circuit(self.num_qubits, label) for label in labels()]