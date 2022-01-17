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