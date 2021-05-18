from typing import Optional
from functools import lru_cache
from numpy.random import Generator, default_rng
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import SdgGate, HGate
from qiskit.quantum_info import Clifford, random_clifford

class VGate(Gate):
    """V Gate used in Clifford synthesis."""
    def __init__(self):
        """Create new V Gate."""
        super().__init__('v', 1, [])

    def _define(self):
        """V Gate definition."""
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q)
        qc.data = [(SdgGate(), [q[0]], []), (HGate(), [q[0]], [])]
        self.definition = qc

def v(self, q):
    """Apply V to q."""
    return self.append(VGate(), [q], [])
QuantumCircuit.v = v

class CliffordUtils():
    NUM_CLIFFORD_1_QUBIT = 24
    NUM_CLIFFORD_2_QUBIT = 11520
    CLIFFORD_1_QUBIT_SIG = (2, 3, 4)
    CLIFFORD_2_QUBIT_SIGS = [(2, 2, 3, 3, 4, 4),
                             (2, 2, 3, 3, 3, 3, 4, 4),
                             (2, 2, 3, 3, 3, 3, 4, 4),
                             (2, 2, 3, 3, 4, 4)]

    def clifford_1_qubit(self, num):
        return Clifford(self.clifford_1_qubit_circuit(num))

    def clifford_2_qubit(self, num):
        return Clifford(self.clifford_2_qubit_circuit(num))

    def random_cliffords(self, num_qubits: int, size: int = 1,
                                  rng: Optional[Generator] = None):
        """Generate a list of random clifford circuits"""
        if num_qubits > 2:
            return random_clifford(num_qubits, seed=rng)

        if rng is None:
            rng = default_rng()

        if num_qubits == 1:
            samples = rng.integers(24, size=size)
            return [Clifford(self.clifford_1_qubit_circuit(i)) for i in samples]
        else:
            samples = rng.integers(11520, size=size)
            return [Clifford(self.clifford_2_qubit_circuit(i)) for i in samples]

    def random_clifford_circuits(self, num_qubits: int, size: int = 1,
                                  rng: Optional[Generator] = None):
        """Generate a list of random clifford circuits"""
        if num_qubits > 2:
            return [random_clifford(num_qubits, seed=rng).to_circuit()
                    for _ in range(size)]

        if rng is None:
            rng = default_rng()

        if num_qubits == 1:
            samples = rng.integers(24, size=size)
            return [self.clifford_1_qubit_circuit(i) for i in
                    samples]
        else:
            samples = rng.integers(11520, size=size)
            return [self.clifford_2_qubit_circuit(i) for i in
                    samples]

    @lru_cache(maxsize=24)
    def clifford_1_qubit_circuit(self, num):
        (i, j, p) = self._unpack_num(num, self.CLIFFORD_1_QUBIT_SIG)
        qc = QuantumCircuit(1)
        if i == 1:
            qc.h(0)
        if j == 1:
            qc.v(0)
        if j == 2:
            qc.v(0)
            qc.v(0)
        if p == 1:
            qc.x(0)
        if p == 2:
            qc.y(0)
        if p == 3:
            qc.z(0)
        return qc

    @lru_cache(maxsize=11520)
    def clifford_2_qubit_circuit(self, num):
        vals = self._unpack_num_multi_sigs(num, self.CLIFFORD_2_QUBIT_SIGS)
        qc = QuantumCircuit(2)
        if vals[0] == 0 or vals[0] == 3:
            (form, i0, i1, j0, j1, p0, p1) = vals
        else:
            (form, i0, i1, j0, j1, k0, k1, p0, p1) = vals
        if i0 == 1:
            qc.h(0)
        if i1 == 1:
            qc.h(1)
        if j0 == 1:
            qc.v(0)
        if j0 == 2:
            qc.v(0)
            qc.v(0)
        if j1 == 1:
            qc.v(1)
        if j1 == 2:
            qc.v(1)
            qc.v(1)
        if form == 1 or form == 2 or form == 3:
            qc.cx(0, 1)
        if form == 2 or form == 3:
            qc.cx(1, 0)
        if form == 3:
            qc.cx(0, 1)
        if form == 1 or form == 2:
            if k0 == 1:
                qc.v(0)
            if k0 == 2:
                qc.v(0)
                qc.v(0)
            if k1 == 1:
                qc.v(1)
            if k1 == 2:
                qc.v(1)
                qc.v(1)
        if p0 == 1:
            qc.x(0)
        if p0 == 2:
            qc.y(0)
        if p0 == 3:
            qc.z(0)
        if p1 == 1:
            qc.x(1)
        if p1 == 2:
            qc.y(1)
        if p1 == 3:
            qc.z(1)
        return qc

    def _unpack_num(self, num, sig):
        res = []
        for k in sig:
            res.append(num % k)
            num //= k
        return res

    def _unpack_num_multi_sigs(self, num, sigs):
        for i, sig in enumerate(sigs):
            sig_size = 1
            for k in sig:
                sig_size *= k
            if num < sig_size:
                return [i] + self._unpack_num(num, sig)
            num -= sig_size
        return None