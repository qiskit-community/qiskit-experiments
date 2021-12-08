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
Quantum Process Tomography experiment
"""

from typing import Union, Optional, Iterable, List, Tuple, Sequence
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.quantum_info.operators.base_operator import BaseOperator
from .tomography_experiment import TomographyExperiment
from .qpt_analysis import ProcessTomographyAnalysis
from . import basis


class ProcessTomography(TomographyExperiment):
    """Quantum process tomography experiment.

    # section: overview
        Quantum process tomography (QPT) is a method for experimentally
        reconstructing the quantum channel from measurement data.

        A QPT experiment prepares multiple input states, evolves them by the
        circuit, then performs multiple measurements in different measurement
        bases. The resulting measurement data is then post-processed by a
        tomography fitter to reconstruct the quantum channel.

    # section: note
        Performing full process tomography on an `N`-qubit circuit requires
        running :math:`4^N 3^N` measurement circuits when using the default
        preparation and measurement bases.

    # section: analysis_ref
        :py:class:`ProcessTomographyAnalysis`

    # section: see_also
        qiskit_experiments.library.tomography.tomography_experiment.TomographyExperiment

    """

    def __init__(
        self,
        circuit: Union[QuantumCircuit, Instruction, BaseOperator],
        measurement_basis: basis.BaseTomographyMeasurementBasis = basis.PauliMeasurementBasis(),
        measurement_qubits: Optional[Sequence[int]] = None,
        preparation_basis: basis.BaseTomographyPreparationBasis = basis.PauliPreparationBasis(),
        preparation_qubits: Optional[Sequence[int]] = None,
        basis_indices: Optional[Iterable[Tuple[List[int], List[int]]]] = None,
        qubits: Optional[Sequence[int]] = None,
    ):
        """Initialize a quantum process tomography experiment.

        Args:
            circuit: the quantum process circuit. If not a quantum circuit
                it must be a class that can be appended to a quantum circuit.
            measurement_basis: Tomography basis for measurements. If not specified the
                default basis is the :class:`~basis.PauliMeasurementBasis`.
            measurement_qubits: Optional, the qubits to be measured. These should refer
                to the logical qubits in the state circuit. If None all qubits
                in the state circuit will be measured.
            preparation_basis: Tomography basis for measurements. If not specified the
                        default basis is the :class:`~basis.PauliPreparationBasis`.
            preparation_qubits: Optional, the qubits to be prepared. These should refer
                to the logical qubits in the process circuit. If None all qubits
                in the process circuit will be prepared.
            basis_indices: Optional, a list of basis indices for generating partial
                tomography measurement data. Each item should be given as a pair of
                lists of preparation and measurement basis configurations
                ``([p[0], p[1], ..], m[0], m[1], ...])``, where ``p[i]`` is the
                preparation basis index, and ``m[i]`` is the measurement basis index
                for qubit-i. If not specified full tomography for all indices of the
                preparation and measurement bases will be performed.
            qubits: Optional, the physical qubits for the initial state circuit.
        """
        super().__init__(
            circuit,
            measurement_basis=measurement_basis,
            measurement_qubits=measurement_qubits,
            preparation_basis=preparation_basis,
            preparation_qubits=preparation_qubits,
            basis_indices=basis_indices,
            qubits=qubits,
        )
        self.analysis = ProcessTomographyAnalysis()
