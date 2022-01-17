# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Utility functions for readout mitigation experiments
"""
from qiskit import QuantumCircuit


def calibration_circuit(num_qubits: int, label: str) -> QuantumCircuit:
    """Return a calibration circuit.

    This is an N-qubit circuit where N is the length of the label.
    The circuit consists of X-gates on qubits with label bits equal to 1,
    and measurements of all qubits.
    """
    circ = QuantumCircuit(num_qubits, name="meas_mit_cal_" + label)
    for i, val in enumerate(reversed(label)):
        if val == "1":
            circ.x(i)
    circ.measure_all()
    circ.metadata = {"label": label}
    return circ
