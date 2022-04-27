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
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import Choi, Operator, Statevector, DensityMatrix, partial_trace
from qiskit_experiments.exceptions import QiskitError
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
        measurement_basis: basis.MeasurementBasis = basis.PauliMeasurementBasis(),
        measurement_qubits: Optional[Sequence[int]] = None,
        preparation_basis: basis.PreparationBasis = basis.PauliPreparationBasis(),
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
            analysis=ProcessTomographyAnalysis(),
        )

        # Set target quantum channel
        self.analysis.set_options(target=self._target_quantum_channel())

    def _target_quantum_channel(self) -> Union[Choi, Operator]:
        """Return the process tomography target"""
        # Check if circuit contains measure instructions
        # If so we cannot return target state
        circuit_ops = self._circuit.count_ops()
        if "measure" in circuit_ops:
            return None

        try:
            circuit = self._permute_circuit()
            if "reset" in circuit_ops or "kraus" in circuit_ops or "superop" in circuit_ops:
                channel = Choi(circuit)
            else:
                channel = Operator(circuit)
        except QiskitError:
            # Circuit couldn't be simulated
            return None

        total_qubits = self._circuit.num_qubits
        num_meas = total_qubits if self._meas_qubits is None else len(self._meas_qubits)
        num_prep = total_qubits if self._prep_qubits is None else len(self._prep_qubits)

        # If all qubits are prepared or measurement we are done
        if num_meas == total_qubits and num_prep == total_qubits:
            return channel

        # Convert channel to a state to project and trace out non-tomography
        # input and output qubits
        if isinstance(channel, Operator):
            chan_state = Statevector(np.ravel(channel, order="F"))
        else:
            chan_state = DensityMatrix(channel.data)

        # Get qargs for non measured and prepared subsystems
        non_meas_qargs = list(range(num_meas, total_qubits))
        non_prep_qargs = list(range(total_qubits + num_prep, 2 * total_qubits))

        # Project non-prepared subsystems on to the zero state
        if non_prep_qargs:
            proj0 = Operator([[1, 0], [0, 0]])
            for qarg in non_prep_qargs:
                chan_state = chan_state.evolve(proj0, [qarg])

        # Trace out indices to remove
        tr_qargs = non_meas_qargs + non_prep_qargs
        chan_state = partial_trace(chan_state, tr_qargs)
        channel = Choi(chan_state.data, input_dims=[2] * num_prep, output_dims=[2] * num_meas)
        return channel
