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
Quantum Tomography experiment
"""

import copy
from typing import Union, Optional, Iterable, List, Tuple, Sequence
from itertools import product
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.circuit.library import Permutation
from qiskit.providers.backend import Backend
from qiskit.quantum_info.operators.base_operator import BaseOperator
import qiskit.quantum_info as qi
from qiskit_experiments.exceptions import QiskitError
from qiskit_experiments.framework import BaseExperiment, Options
from .basis import BaseTomographyMeasurementBasis, BaseTomographyPreparationBasis
from .tomography_analysis import TomographyAnalysis


class TomographyExperiment(BaseExperiment):
    """Base experiment for quantum state and process tomography.

    # section: analysis_ref
        :py:class:`TomographyAnalysis`
    """

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.

        Experiment Options:
            measurement_basis (:class:`~basis.BaseTomographyMeasurementBasis`): The
                Tomography measurement basis to use for the experiment.
                The default basis is the :class:`~basis.PauliMeasurementBasis` which
                performs measurements in the Pauli Z, X, Y bases for each qubit
                measurement.

        """
        options = super()._default_experiment_options()

        options.basis_indices = None

        return options

    def __init__(
        self,
        circuit: Union[QuantumCircuit, Instruction, BaseOperator],
        backend: Optional[Backend] = None,
        measurement_basis: Optional[BaseTomographyMeasurementBasis] = None,
        measurement_qubits: Optional[Sequence[int]] = None,
        preparation_basis: Optional[BaseTomographyPreparationBasis] = None,
        preparation_qubits: Optional[Sequence[int]] = None,
        basis_indices: Optional[Iterable[Tuple[List[int], List[int]]]] = None,
        qubits: Optional[Sequence[int]] = None,
    ):
        """Initialize a tomography experiment.

        Args:
            circuit: the quantum process circuit. If not a quantum circuit
                it must be a class that can be appended to a quantum circuit.
            backend: The backend to run the experiment on.
            measurement_basis: Tomography basis for measurements.
            measurement_qubits: Optional, the qubits to be measured. These should refer
                to the logical qubits in the state circuit.
            preparation_basis: Tomography basis for measurements.
            preparation_qubits: Optional, the qubits to be prepared. These should refer
                to the logical qubits in the process circuit.
            basis_indices: Optional, the basis elements to be measured. If None
                All basis elements will be measured.
            qubits: Optional, the physical qubits for the initial state circuit.

        Raises:
            QiskitError: if input params are invalid.
        """
        # Initialize BaseExperiment
        if qubits is None:
            qubits = range(circuit.num_qubits)
        super().__init__(qubits, analysis=TomographyAnalysis(), backend=backend)

        # Get the target tomography circuit
        if isinstance(circuit, QuantumCircuit):
            target_circuit = circuit
        else:
            # Convert input to a circuit
            num_qubits = circuit.num_qubits
            target_circuit = QuantumCircuit(num_qubits)
            target_circuit.append(circuit, range(num_qubits))
        self._circuit = target_circuit

        # Measurement basis and qubits
        self._meas_circ_basis = measurement_basis
        if measurement_qubits:
            self._meas_qubits = tuple(measurement_qubits)
            for qubit in self._meas_qubits:
                if qubit not in range(self.num_qubits):
                    raise QiskitError(
                        f"measurement qubit ({qubit}) is outside the range"
                        f" of circuit qubits [0, {self.num_qubits})."
                    )
        else:
            self._meas_qubits = None

        # Preparation basis and qubits
        self._prep_circ_basis = preparation_basis
        if preparation_qubits:
            self._prep_qubits = tuple(preparation_qubits)
            for qubit in self._prep_qubits:
                if qubit not in range(self.num_qubits):
                    raise QiskitError(
                        f"preparation qubit ({qubit}) is outside the range"
                        f" of circuit qubits [0, {self.num_qubits})."
                    )
        else:
            self._prep_qubits = None

        # Configure experiment options
        if basis_indices:
            self.set_experiment_options(basis_indices=basis_indices)

        # Compute target state
        self._target = None
        if self._prep_circ_basis:
            self._target = self._target_quantum_channel(
                self._circuit,
                measurement_qubits=self._meas_qubits,
                preparation_qubits=self._prep_qubits,
            )
        else:
            self._target = self._target_quantum_state(
                self._circuit, measurement_qubits=self._meas_qubits
            )

        # Configure analysis basis options
        analysis_options = {}
        if measurement_basis:
            analysis_options["measurement_basis"] = measurement_basis
        if preparation_basis:
            analysis_options["preparation_basis"] = preparation_basis
        self.analysis.set_options(**analysis_options)

    def _metadata(self):
        metadata = super()._metadata()
        if self._target:
            metadata["target"] = copy.copy(self._target)
        return metadata

    def circuits(self):

        # Get qubits and clbits
        meas_qubits = self._meas_qubits or range(self.num_qubits)
        total_clbits = self._circuit.num_clbits + len(meas_qubits)
        circ_qubits = list(range(self._circuit.num_qubits))
        circ_clbits = list(range(self._circuit.num_clbits))
        meas_clbits = list(range(self._circuit.num_clbits, total_clbits))

        # Build circuits
        circuits = []
        for prep_element, meas_element in self._basis_indices():
            name = f"{self._type}_{meas_element}"
            metadata = {
                "experiment_type": self._type,
                "clbits": meas_clbits,
                "m_idx": list(meas_element),
            }
            if prep_element:
                name += f"_{prep_element}"
                metadata["p_idx"] = list(prep_element)

            circ = QuantumCircuit(self.num_qubits, total_clbits, name=name)

            if prep_element:
                # Add tomography preparation
                prep_qubits = self._prep_qubits or range(self.num_qubits)
                prep_circ = self._prep_circ_basis.circuit(prep_element)
                circ.reset(prep_qubits)
                circ.append(prep_circ, prep_qubits)
                circ.barrier(prep_qubits)

            # Add target circuit
            # Have to use compose since circuit.to_instruction has a bug
            # when circuit contains classical registers and conditionals
            circ = circ.compose(self._circuit, circ_qubits, circ_clbits)

            # Add tomography measurement
            meas_circ = self._meas_circ_basis.circuit(meas_element)
            circ.barrier(meas_qubits)
            circ.append(meas_circ, meas_qubits)
            circ.measure(meas_qubits, meas_clbits)

            # Add metadata
            circ.metadata = metadata
            circuits.append(circ)
        return circuits

    def _basis_indices(self):
        """Return list of basis element indices"""
        basis_indices = self.experiment_options.basis_indices
        if basis_indices is not None:
            return basis_indices

        meas_size = len(self._meas_circ_basis)
        num_meas = len(self._meas_qubits) if self._meas_qubits else self.num_qubits
        meas_elements = product(range(meas_size), repeat=num_meas)
        if self._prep_circ_basis:
            prep_size = len(self._prep_circ_basis)
            num_prep = len(self._prep_qubits) if self._prep_qubits else self.num_qubits
            prep_elements = product(range(prep_size), repeat=num_prep)
        else:
            prep_elements = [None]

        return product(prep_elements, meas_elements)

    @staticmethod
    def _permute_circuit(
        circuit: QuantumCircuit,
        measurement_qubits: Optional[Sequence[int]] = None,
        preparation_qubits: Optional[Sequence[int]] = None,
    ):
        """Permute circuit qubits.

        This permutes the circuit so that the specified preparation and measurement
        qubits correspond to input and output qubits [0, ..., N-1] respectively
        for the returned circuit.
        """
        if measurement_qubits is None and preparation_qubits is None:
            return circuit

        total_qubits = circuit.num_qubits
        total_clbits = circuit.num_clbits
        if total_clbits:
            perm_circ = QuantumCircuit(total_qubits, total_clbits)
        else:
            perm_circ = QuantumCircuit(total_qubits)

        # Apply permutation to put prep qubits as [0, ..., M-1]
        if preparation_qubits:
            prep_qargs = list(preparation_qubits)
            if len(preparation_qubits) != total_qubits:
                prep_qargs += [i for i in range(total_qubits) if i not in preparation_qubits]
            perm_circ.append(Permutation(total_qubits, prep_qargs).inverse(), range(total_qubits))

        # Apply original circuit
        if total_clbits:
            perm_circ = perm_circ.compose(circuit, range(total_qubits), range(total_clbits))
        else:
            perm_circ = perm_circ.compose(circuit, range(total_qubits))

        # Apply permutation to put meas qubits as [0, ..., M-1]
        if measurement_qubits:
            meas_qargs = list(measurement_qubits)
            if len(measurement_qubits) != total_qubits:
                meas_qargs += [i for i in range(total_qubits) if i not in measurement_qubits]
            perm_circ.append(Permutation(total_qubits, meas_qargs), range(total_qubits))

        return perm_circ

    @classmethod
    def _target_quantum_state(
        cls, circuit: QuantumCircuit, measurement_qubits: Optional[Sequence[int]] = None
    ):
        """Return the state tomography target"""
        # Check if circuit contains measure instructions
        # If so we cannot return target state
        circuit_ops = circuit.count_ops()
        if "measure" in circuit_ops:
            return None

        perm_circ = cls._permute_circuit(circuit, measurement_qubits=measurement_qubits)

        try:
            if "reset" in circuit_ops or "kraus" in circuit_ops or "superop" in circuit_ops:
                state = qi.DensityMatrix(perm_circ)
            else:
                state = qi.Statevector(perm_circ)
        except QiskitError:
            # Circuit couldn't be simulated
            return None

        total_qubits = circuit.num_qubits
        if measurement_qubits:
            num_meas = len(measurement_qubits)
        else:
            num_meas = total_qubits
        if num_meas == total_qubits:
            return state

        # Trace out non-measurement qubits
        tr_qargs = range(num_meas, total_qubits)
        return qi.partial_trace(state, tr_qargs)

    @classmethod
    def _target_quantum_channel(
        cls,
        circuit: QuantumCircuit,
        measurement_qubits: Optional[Sequence[int]] = None,
        preparation_qubits: Optional[Sequence[int]] = None,
    ):
        """Return the process tomography target"""
        # Check if circuit contains measure instructions
        # If so we cannot return target state
        circuit_ops = circuit.count_ops()
        if "measure" in circuit_ops:
            return None

        perm_circ = cls._permute_circuit(
            circuit, measurement_qubits=measurement_qubits, preparation_qubits=preparation_qubits
        )
        try:
            if "reset" in circuit_ops or "kraus" in circuit_ops or "superop" in circuit_ops:
                channel = qi.Choi(perm_circ)
            else:
                channel = qi.Operator(perm_circ)
        except QiskitError:
            # Circuit couldn't be simulated
            return None

        total_qubits = circuit.num_qubits
        if measurement_qubits:
            num_meas = len(measurement_qubits)
        else:
            num_meas = total_qubits
        if preparation_qubits:
            num_prep = len(preparation_qubits)
        else:
            num_prep = total_qubits

        if num_prep == total_qubits and num_meas == total_qubits:
            return channel

        # Trace out non-measurement subsystems
        tr_qargs = []
        if preparation_qubits:
            tr_qargs += list(range(num_prep, total_qubits))
        if measurement_qubits:
            tr_qargs += list(range(total_qubits + num_meas, 2 * total_qubits))

        chan_state = qi.Statevector(np.ravel(channel, order="F"))
        chan_state = qi.partial_trace(chan_state, tr_qargs) / 2 ** (total_qubits - num_meas)
        channel = qi.Choi(chan_state.data, input_dims=[2] * num_prep, output_dims=[2] * num_meas)
        return channel
