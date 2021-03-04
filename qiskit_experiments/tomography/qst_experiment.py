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
Basic quantum state tomography experiment demo
"""

from qiskit.circuit import QuantumCircuit

# Temporarily import helper functions from ignis tomography module
from qiskit.ignis.verification.tomography import state_tomography_circuits

from qiskit_experiments.base_experiment import BaseExperiment
from .qst_analysis import QSTAnalysis


class QSTExperiment(BaseExperiment):
    """Basic quantum State Tomography experiment demonstration class"""

    __analysis_class__ = QSTAnalysis

    def __init__(self, initial_state, qubits=None, basis="Pauli"):
        """Initialize a state tomography experiment.

        Args:
            initial_state (QuantumCircuit or Gate or Operator or Statevector): the
                initial state circuit. If a Gate or Operator it will be
                appended to a quantum circuit.
            qubits (int or list or None): Optional, the qubits to be measured.
                If None all qubits will be measured.
        """
        # NOTE: this simple example only works if all qubits are measured
        num_qubits = initial_state.num_qubits
        if not qubits:
            qubits = num_qubits
        if isinstance(initial_state, QuantumCircuit):
            circuit = initial_state
        else:
            # Convert input to a circuit
            circuit = QuantumCircuit(num_qubits)
            circuit.append(initial_state, range(num_qubits))
        self._state_circuit = circuit

        super().__init__(qubits, circuit_options={"basis": basis})

    def circuits(self, backend=None, **circuit_options):

        clbits = list(
            range(self._state_circuit.num_clbits, self._state_circuit.num_clbits + self.num_qubits)
        )
        qubits = self.physical_qubits
        tomo_circs = state_tomography_circuits(self._state_circuit, list(range(self.num_qubits)))
        for circ in tomo_circs:
            # pylint: disable = eval-used
            circ.metadata = {
                "experiment_type": self._type,
                "meas_basis": eval(circ.name),
                "clbits": clbits,
                "qubits": qubits,
            }
        return tomo_circs
