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
Batch Experiment class.
"""

from typing import List, Optional
from collections import OrderedDict

from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend
from .composite_experiment import CompositeExperiment, BaseExperiment


class BatchExperiment(CompositeExperiment):
    """Batch experiment class"""

    def __init__(self, experiments: List[BaseExperiment], backend: Optional[Backend] = None):
        """Initialize a batch experiment.

        Args:
            experiments: a list of experiments.
            backend: Optional, the backend to run the experiment on.
        """

        # Generate qubit map
        self._qubit_map = OrderedDict()
        logical_qubit = 0
        for expr in experiments:
            for physical_qubit in expr.physical_qubits:
                if physical_qubit not in self._qubit_map:
                    self._qubit_map[physical_qubit] = logical_qubit
                    logical_qubit += 1
        qubits = tuple(self._qubit_map.keys())
        super().__init__(experiments, qubits, backend=backend)

    def circuits(self):

        batch_circuits = []

        # Generate data for combination
        for index, expr in enumerate(self._experiments):
            if self.physical_qubits == expr.physical_qubits:
                qubit_mapping = None
            else:
                qubit_mapping = [self._qubit_map[qubit] for qubit in expr.physical_qubits]
            for circuit in expr.circuits():
                # Update metadata
                circuit.metadata = {
                    "experiment_type": self._type,
                    "composite_metadata": [circuit.metadata],
                    "composite_index": [index],
                }
                # Remap qubits if required
                if qubit_mapping:
                    circuit = self._remap_qubits(circuit, qubit_mapping)
                batch_circuits.append(circuit)
        return batch_circuits

    def _remap_qubits(self, circuit, qubit_mapping):
        """Remap qubits if physical qubit layout is different to batch layout"""
        num_qubits = self.num_qubits
        num_clbits = circuit.num_clbits
        new_circuit = QuantumCircuit(num_qubits, num_clbits, name="batch_" + circuit.name)
        new_circuit.metadata = circuit.metadata
        new_circuit.append(circuit, qubit_mapping, list(range(num_clbits)))
        return new_circuit
