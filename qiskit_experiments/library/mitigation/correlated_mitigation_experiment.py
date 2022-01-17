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


        A readout mitigation experiment aims to determine the assignment matrix
        for the readout error in a given device, and generate a corresponding mitigator
        based on the `qiskit-terra` classes of `Correlated readout mitigator
        <https://qiskit.org/documentation/stubs/qiskit.result.CorrelatedReadoutMitigator.html>_`
        and `Local readout mitigator
        <https://qiskit.org/documentation/stubs/qiskit.result.BaseReadoutMitigator.html#qiskit.result.BaseReadoutMitigator.html>_`

        A *Correlated readout mitigator* uses the full :math:`2^n\times 2^n` assignment matrix, meaning
        it can only be used for small values of :math:`n`.

        A *Local readout mitigator* works under the assumption the readout errors are mostly *local*, meaning
        readout errors for different qubits are independent of each other. In this case, the assignment matrix
        is the tensor product of :math:`n` :math:`2\times 2` matrices, one for each qubit, making it practical
        to store the assignment matrix in implicit form, by storing the individual :math:`2\times 2` assignment matrices.

        The readout mitigation experiment can be passed a `method` parameter to determine
        which method will be used (and which mitigator will be returned): The default `local` method
        or the `correlated` method.

        In the `local` method, the experiment generates 2 circuits, corresponding to the states
        :math:`|0^n>` and :math:`|1^n>`, measuring the error in all the qubits at once.

        In the `correlated` method, the experiment generates :math:`2^n` circuits, for every possible
        :math:`n`-qubit quantum state.

        See :class:`LocalMitigationAnalysis` and :class:`CorrelatedMitigationAnalysis`
        documentation for additional information on readout mitigation experiment analysis.

    # section: analysis_ref
        :py:class:`LocalMitigationAnalysis`
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