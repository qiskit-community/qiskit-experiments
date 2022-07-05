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
Utilities for using the Clifford group in randomized benchmarking
"""

from typing import Optional, Union, List
from functools import lru_cache
from math import isclose
import numpy as np
from numpy.random import Generator, default_rng

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Instruction
from qiskit.circuit.library import SdgGate, HGate, SGate, SXdgGate
from qiskit.quantum_info import Clifford, random_clifford
from qiskit.compiler import transpile
from qiskit.providers.aer import AerSimulator
from qiskit.exceptions import QiskitError
from .clifford_data import CLIFF_COMPOSE_DATA

class VGate(Gate):
    """V Gate used in Clifford synthesis."""

    def __init__(self):
        """Create new V Gate."""
        super().__init__("v", 1, [])

    def _define(self):
        """V Gate definition."""
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q)
        qc.data = [(SdgGate(), [q[0]], []), (HGate(), [q[0]], [])]
        self.definition = qc


class WGate(Gate):
    """W Gate used in Clifford synthesis."""

    def __init__(self):
        """Create new W Gate."""
        super().__init__("w", 1, [])

    def _define(self):
        """W Gate definition."""
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q)
        qc.data = [(HGate(), [q[0]], []), (SGate(), [q[0]], [])]
        self.definition = qc


class CliffordUtils:
    """Utilities for generating 1 and 2 qubit clifford circuits and elements"""

    NUM_CLIFFORD_1_QUBIT = 24
    NUM_CLIFFORD_2_QUBIT = 11520
    CLIFFORD_1_QUBIT_SIG = (2, 3, 4)
    CLIFFORD_2_QUBIT_SIGS = [
        (2, 2, 3, 3, 4, 4),
        (2, 2, 3, 3, 3, 3, 4, 4),
        (2, 2, 3, 3, 3, 3, 4, 4),
        (2, 2, 3, 3, 4, 4),
    ]
    GENERAL_CLIFF_LIST = ["id", "h", "sdg", "s", "x", "sx", "sxdg", "y", "z", "cx"]
    TRANSPILED_CLIFF_LIST = ["sx", "rz", "cx"]
    NUM_SINGLE_GATE_1_QUBIT_CLIFF = 9

    @classmethod
    def clifford_1_qubit(cls, num):
        """Return the 1-qubit clifford element corresponding to `num`
        where `num` is between 0 and 23.
        """
        return Clifford(cls.clifford_1_qubit_circuit(num), validate=False)

    @classmethod
    def clifford_2_qubit(cls, num):
        """Return the 2-qubit clifford element corresponding to `num`
        where `num` is between 0 and 11519.
        """
        return Clifford(cls.clifford_2_qubit_circuit(num), validate=False)

    @classmethod
    def random_cliffords(
        cls, num_qubits: int, size: int = 1, rng: Optional[Union[int, Generator]] = None
    ):
        """Generate a list of random clifford elements"""
        if num_qubits > 2:
            return random_clifford(num_qubits, seed=rng)

        if rng is None:
            rng = default_rng()

        if isinstance(rng, int):
            rng = default_rng(rng)

        if num_qubits == 1:
            samples = rng.integers(cls.NUM_CLIFFORD_1_QUBIT, size=size)
            return [Clifford(cls.clifford_1_qubit_circuit(i), validate=False) for i in samples]
        else:
            samples = rng.integers(11520, size=size)
            return [Clifford(cls.clifford_2_qubit_circuit(i), validate=False) for i in samples]

    @classmethod
    def random_clifford_circuits(
        cls, num_qubits: int, size: int = 1, rng: Optional[Union[int, Generator]] = None
    ):
        """Generate a list of random clifford circuits"""
        if num_qubits > 2:
            return [random_clifford(num_qubits, seed=rng).to_circuit() for _ in range(size)]

        if rng is None:
            rng = default_rng()

        if isinstance(rng, int):
            rng = default_rng(rng)

        if num_qubits == 1:
            samples = rng.integers(cls.NUM_CLIFFORD_1_QUBIT, size=size)
            return [cls.clifford_1_qubit_circuit(i) for i in samples]
        else:
            samples = rng.integers(11520, size=size)
            return [cls.clifford_2_qubit_circuit(i) for i in samples]

    @classmethod
    @lru_cache(maxsize=24)
    def clifford_1_qubit_circuit(cls, num):
        """Return the 1-qubit clifford circuit corresponding to `num`
        where `num` is between 0 and 23.
        """
        # pylint: disable=unbalanced-tuple-unpacking
        # This is safe since `_unpack_num` returns list the size of the sig
        (i, j, p) = cls._unpack_num(num, cls.CLIFFORD_1_QUBIT_SIG)
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        if i == 1:
            qc.h(0)
        if j == 1:
            qc._append(SXdgGate(), [qr[0]], [])
        if j == 2:
            qc._append(SGate(), [qr[0]], [])
        if p == 1:
            qc.x(0)
        if p == 2:
            qc.y(0)
        if p == 3:
            qc.z(0)
        return qc

    @classmethod
    @lru_cache(maxsize=11520)
    def clifford_2_qubit_circuit(cls, num):
        """Return the 2-qubit clifford circuit corresponding to `num`
        where `num` is between 0 and 11519.
        """
        vals = cls._unpack_num_multi_sigs(num, cls.CLIFFORD_2_QUBIT_SIGS)
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        if vals[0] == 0 or vals[0] == 3:
            (form, i0, i1, j0, j1, p0, p1) = vals
        else:
            (form, i0, i1, j0, j1, k0, k1, p0, p1) = vals
        if i0 == 1:
            qc.h(0)
        if i1 == 1:
            qc.h(1)
        if j0 == 1:
            qc.sxdg(0)
        if j0 == 2:
            qc.s(0)
        if j1 == 1:
            qc.sxdg(1)
        if j1 == 2:
            qc.s(1)
        if form in (1, 2, 3):
            qc.cx(0, 1)
        if form in (2, 3):
            qc.cx(1, 0)
        if form == 3:
            qc.cx(0, 1)
        if form in (1, 2):
            if k0 == 1:
                qc._append(VGate(), [qr[0]], [])
            if k0 == 2:
                qc._append(WGate(), [qr[0]], [])
            if k1 == 1:
                qc._append(VGate(), [qr[1]], [])
            if k1 == 2:
                qc._append(VGate(), [qr[1]], [])
                qc._append(VGate(), [qr[1]], [])
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

    @classmethod
    def _unpack_num(cls, num, sig):
        r"""Returns a tuple :math:`(a_1, \ldots, a_n)` where
        :math:`0 \le a_i \le \sigma_i` where
        sig=:math:`(\sigma_1, \ldots, \sigma_n)` and num is the sequential
        number of the tuple
        """
        res = []
        for k in sig:
            res.append(num % k)
            num //= k
        return res

    @classmethod
    def _unpack_num_multi_sigs(cls, num, sigs):
        """Returns the result of `_unpack_num` on one of the
        signatures in `sigs`
        """
        for i, sig in enumerate(sigs):
            sig_size = 1
            for k in sig:
                sig_size *= k
            if num < sig_size:
                return [i] + cls._unpack_num(num, sig)
            num -= sig_size
        return None

    @classmethod
    def transpile_single_clifford(cls, cliff_circ: QuantumCircuit, basis_gates: List[str]):
        """Transpile a single clifford circuit using basis_gates."""
        backend = AerSimulator()
        return transpile(cliff_circ, backend, optimization_level=1, basis_gates=basis_gates)

    @classmethod
    def generate_1q_transpiled_clifford_circuits(cls, basis_gates: List[str]):
        """Generate all transpiled clifford circuits"""
        transpiled_circs = []
        for num in range(0, cls.NUM_CLIFFORD_1_QUBIT):
            circ = cls.clifford_1_qubit_circuit(num=num)
            transpiled_circ = cls.transpile_single_clifford(circ, basis_gates)
            transpiled_circs.append(transpiled_circ)
        return transpiled_circs

    @classmethod
    def num_from_1_qubit_clifford_single_gate(cls, inst: Instruction,
                                              basis_gates: List[str]) -> int:
        """
        This method does the reverse of clifford_1_qubit_circuit -
        given a clifford, it returns the corresponding integer, with the mapping
        defined in the above method.
        The mapping is in the context of the basis_gates. Therefore, we define here
        the possible supersets of basis gates, and verify that the given inst belong to
        one of these sets.
        """
        name = inst.name

        gates_with_delay = basis_gates.copy()
        gates_with_delay.append("delay")
        if not name in gates_with_delay:
            raise QiskitError("Instruction {} is not in the basis gates".format(inst.name))
        if set(basis_gates).issubset(set(cls.GENERAL_CLIFF_LIST)):
            num_dict = {
                "id": 0,
                "h": 1,
                "sxdg": 2,
                "s": 4,
                "x": 6,
                "sx": 8,
                "y": 12,
                "z": 18,
                "sdg": 22,
                "delay": 0,
            }
            return num_dict[name]

        if set(basis_gates).issubset(set(cls.TRANSPILED_CLIFF_LIST)):
            if name == "sx":
                return 8
            if name == "delay":
                return 0
            if name == "rz":
                # The next two are identical up to a phase, which makes no difference
                # for the associated Cliffords
                if isclose(inst.params[0], np.pi) or isclose(inst.params[0], -np.pi):
                    return 18
                if isclose(inst.params[0], np.pi / 2):
                    return 4
                if isclose(inst.params[0], -np.pi / 2):
                    return 22
                else:
                    raise QiskitError("wrong param {} for rz in clifford".format(inst.params[0]))

        raise QiskitError("Instruction {} could not be converted to Clifford gate".format(name))

    @classmethod
    def compose_num_with_clifford(
        cls, composed_num: int, qc: QuantumCircuit, basis_gates: List[str]
    ) -> int:
        """Compose a number that represents a Clifford, with a single-gate Clifford, and return the
        number that represents the resulting Clifford."""

        # The numbers corresponding to single gate Cliffords are not in sequence -
        # see num_from_1_qubit_clifford_single_gate. To compute the index in
        # the array CLIFF_COMPOSE_DATA, we map the numbers to [0, 8].
        map_clifford_num_to_array_index = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 8: 5, 12: 6, 18: 7, 22: 8}
        for inst in qc:
            num = cls.num_from_1_qubit_clifford_single_gate(inst=inst[0], basis_gates=basis_gates)
            index = (
                cls.NUM_SINGLE_GATE_1_QUBIT_CLIFF * composed_num
                + map_clifford_num_to_array_index[num]
            )
            composed_num = CLIFF_COMPOSE_DATA[index]
        return composed_num
