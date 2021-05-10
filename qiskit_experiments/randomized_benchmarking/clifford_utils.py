import random
import json
import os
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Clifford

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

    def __init__(self):
        self._cliffs_2_qubit_dict = None

    def _load_cliffs_2_qubit_dict(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path,'cliff_2.json')) as f:
            self._cliffs_2_qubit_dict = json.load(f)

    def clifford_1_qubit(self, num):
        cliffs_1_qubit_dict = [[["+Z"], ["+X"]], [["+X"], ["+Z"]],
                               [["+X"], ["+Y"]], [["+Y"], ["+X"]],
                               [["+Y"], ["+Z"]], [["+Z"], ["+Y"]],
                               [["-Z"], ["+X"]], [["+X"], ["-Z"]],
                               [["+X"], ["-Y"]], [["-Y"], ["+X"]],
                               [["-Y"], ["-Z"]], [["-Z"], ["-Y"]],
                               [["-Z"], ["-X"]], [["-X"], ["-Z"]],
                               [["-X"], ["+Y"]], [["+Y"], ["-X"]],
                               [["+Y"], ["-Z"]], [["-Z"], ["+Y"]],
                               [["+Z"], ["-X"]], [["-X"], ["+Z"]],
                               [["-X"], ["-Y"]], [["-Y"], ["-X"]],
                               [["-Y"], ["+Z"]], [["+Z"], ["-Y"]]]
        stabilizer, destabilizer = cliffs_1_qubit_dict[num]
        return Clifford.from_dict({"stabilizer": stabilizer,
                                   "destabilizer": destabilizer})

    def clifford_2_qubit(self, num):
        if self._cliffs_2_qubit_dict is None:
            self._load_cliffs_2_qubit_dict()
        stabilizer, destabilizer = self._cliffs_2_qubit_dict[num]
        return Clifford.from_dict({"stabilizer": stabilizer,
                                   "destabilizer": destabilizer})

    def random_clifford(self, num_qubits, seed):
        if num_qubits == 1:
            return self.random_clifford_1_qubit()
        elif num_qubits == 2:
            return self.random_clifford_2_qubit()
        else:
            return None

    def random_clifford_1_qubit(self):
        num = random.randint(1, self.NUM_CLIFFORD_1_QUBIT) - 1
        return (self.clifford_1_qubit(num),
                self.clifford_1_qubit_circuit(num))

    def random_clifford_2_qubit(self):
        num = random.randint(1, self.NUM_CLIFFORD_2_QUBIT) - 1
        return (self.clifford_2_qubit(num),
                self.clifford_2_qubit_circuit(num))

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