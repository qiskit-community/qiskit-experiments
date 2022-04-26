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
Quantum State Tomography experiment
"""

from typing import Union, Optional, Iterable, List, Sequence
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
from qiskit_experiments.exceptions import QiskitError
from .tomography_experiment import TomographyExperiment
from .qst_analysis import StateTomographyAnalysis
from . import basis


class StateTomography(TomographyExperiment):
    """Quantum state tomography experiment.

    # section: overview
        Quantum state tomography (QST) is a method for experimentally
        reconstructing the quantum state from measurement data.

        A QST experiment measures the state prepared by quantum
        circuit in different measurement bases and post-processes the
        measurement data to reconstruct the state.

    # section: note
        Performing full state tomography on an `N`-qubit state requires
        running :math:`3^N` measurement circuits when using the default
        measurement basis.

    # section: analysis_ref
        :py:class:`StateTomographyAnalysis`

    # section: see_also
        qiskit_experiments.library.tomography.tomography_experiment.TomographyExperiment

    """

    def __init__(
        self,
        circuit: Union[QuantumCircuit, Instruction, BaseOperator, Statevector],
        measurement_basis: basis.MeasurementBasis = basis.PauliMeasurementBasis(),
        measurement_qubits: Optional[Sequence[int]] = None,
        basis_indices: Optional[Iterable[List[int]]] = None,
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
            basis_indices: Optional, a list of basis indices for generating partial
                tomography measurement data. Each item should be given as a list of
                measurement basis configurations ``[m[0], m[1], ...]`` where ``m[i]``
                is the measurement basis index for qubit-i. If not specified full
                tomography for all indices of the measurement basis will be performed.
            qubits: Optional, the physical qubits for the initial state circuit.
        """
        if isinstance(circuit, Statevector):
            # Convert to circuit using initialize instruction
            circ = QuantumCircuit(circuit.num_qubits)
            circ.initialize(circuit)
            circuit = circ

        if basis_indices is not None:
            # Add trivial preparation indices for base class
            basis_indices = [([], i) for i in basis_indices]

        super().__init__(
            circuit,
            measurement_basis=measurement_basis,
            measurement_qubits=measurement_qubits,
            basis_indices=basis_indices,
            qubits=qubits,
            analysis=StateTomographyAnalysis(),
        )

        # Set target quantum state
        self.analysis.set_options(target=self._target_quantum_state())

    def _target_quantum_state(self) -> Union[Statevector, DensityMatrix]:
        """Return the state tomography target"""
        # Check if circuit contains measure instructions
        # If so we cannot return target state
        circuit_ops = self._circuit.count_ops()
        if "measure" in circuit_ops:
            return None

        try:
            circuit = self._permute_circuit()
            if "reset" in circuit_ops or "kraus" in circuit_ops or "superop" in circuit_ops:
                state = DensityMatrix(circuit)
            else:
                state = Statevector(circuit)
        except QiskitError:
            # Circuit couldn't be simulated
            return None

        if self._meas_qubits is None:
            return state

        non_meas_qargs = list(range(len(self._meas_qubits), self._circuit.num_qubits))
        if non_meas_qargs:
            # Trace over non-measured qubits
            state = partial_trace(state, non_meas_qargs)

        return state
