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

from typing import Union, Optional, Iterable, List
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import Statevector
from .tomography_experiment import TomographyExperiment, Options
from .qst_analysis import StateTomographyAnalysis
from . import basis


class StateTomography(TomographyExperiment):
    """Quantum state tomography experiment.

    Overview
        Quantum state tomography (QST) is a method for experimentally
        reconstructing the quantum state from measurement data.

        A QST experiment measures the state prepared by quantum
        circuit in different measurement bases and post-processes the
        measurement data to reconstruct the state.

        See :class:`TomographyAnalysis` documentation for additional
        information on tomography experiment analysis.

        .. note::
            Performing full state tomography on an `N`-qubit state requires
            running :math:`3^N` measurement circuits when using the default
            measurement basis.

    Analysis Class
        :class:`~qiskit.experiments.tomography.TomographyAnalysis`

    Experiment Options
        - **measurement_basis** (:class:`~basis.BaseTomographyMeasurementBasis`):
          The Tomography measurement basis to use for the experiment.
          The default basis is the :class:`~basis.PauliMeasurementBasis` which
          performs measurements in the Pauli Z, X, Y bases for each qubit
          measurement.

    Analysis Options
        - **measurement_basis**
          (:class`~basis.BaseFitterMeasurementBasis`):
          A custom measurement basis for analysis. By default the
          :meth:`experiment_options` measurement basis will be used.
        - **fitter** (``str`` or ``Callable``): The fitter function to use for
          reconstruction.
        - **rescale_psd** (``bool``): If True rescale the fitted state to be
          positive-semidefinite (Default: True).
        - **rescale_trace** (``bool``): If True rescale the state returned by the fitter
          have either trace 1 (Default: True).
        - **kwargs**: Additional kwargs will be supplied to the fitter function.
    """

    __analysis_class__ = StateTomographyAnalysis

    @classmethod
    def _default_analysis_options(cls):
        return Options(measurement_basis=basis.PauliMeasurementBasis().matrix)

    def __init__(
        self,
        circuit: Union[QuantumCircuit, Instruction, BaseOperator, Statevector],
        measurement_basis: basis.BaseTomographyMeasurementBasis = basis.PauliMeasurementBasis(),
        measurement_qubits: Optional[Iterable[int]] = None,
        basis_indices: Optional[Iterable[List[int]]] = None,
        qubits: Optional[Iterable[int]] = None,
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
        )
